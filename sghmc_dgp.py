import os
import tensorflow as tf
import numpy as np

from sghmc_base import BaseModel
import conditionals
from scipy.cluster.vq import kmeans2
from tensorflow.core.protobuf import rewriter_config_pb2




class Layer(object):
    def __init__(self, kern, outputs, Z, mean=None, ltype='Plain'):
        self.outputs, self.kernel = outputs, kern
        self.M = Z.shape[0]
        self.mean = mean

        self.Z = tf.Variable(Z, dtype=tf.float64, name='Z')
        self.mean = mean
        self.U = tf.Variable(np.zeros((self.M, self.outputs)), dtype=tf.float64, trainable=False, name='U')

        self.Lz = tf.placeholder_with_default(self._compute_Lz(self.Z),
                shape=[Z.shape[0], Z.shape[0]])

        self.ltype = ltype

    def _compute_Lz(self, Z):
        M = tf.shape(Z)[0]
        Kmm = self.kernel.Kzz(Z) + tf.eye(M, dtype=tf.float64) * 1e-3
        return tf.cholesky(Kmm)

    def cacheable_params(self):
        return [self.Lz]

    def conditional(self, X):
        Kmn = self.kernel.Kzx(self.Z, X)

        Knn = self.kernel.Kdiag(X)

        mean, var = conditionals.base_conditional(Kmn, self.Lz, Knn, self.U)

        if self.mean is not None:
            mean += tf.matmul(X, tf.cast(self.mean, tf.float64))
        return mean, var

    def prior(self):
        return -tf.reduce_sum(tf.square(self.U)) / 2.0


class DGP(BaseModel):
    def __init__(self, X, Y, layers, likelihood, minibatch_size, window_size,
                 adam_lr=0.01, epsilon=0.01, mdecay=0.05):
        self.likelihood = likelihood
        self.minibatch_size = minibatch_size
        self.window_size = window_size

        self.layers = layers
        N = X.shape[0]

        super().__init__(X, Y, [l.U for l in self.layers], minibatch_size, window_size) #TODO: the l.U isn't trainable
        self.f, self.fmeans, self.fvars = self.propagate(self.X_placeholder)
        self.y_mean, self.y_var = self.likelihood.predict_mean_and_var(self.fmeans[-1], self.fvars[-1])

        self.prior = tf.add_n([l.prior() for l in self.layers])
        self.log_likelihood = self.likelihood.predict_density(self.fmeans[-1], self.fvars[-1], self.Y_placeholder)

        self.nll = - tf.reduce_sum(self.log_likelihood) / tf.cast(tf.shape(self.X_placeholder)[0], tf.float64) \
                   - (self.prior / N)

        self.generate_update_step(self.nll, epsilon, mdecay)

        global_step = tf.compat.v1.train.create_global_step()
        lr = tf.maximum(tf.compat.v1.train.exponential_decay(learning_rate=adam_lr, global_step=global_step,
            decay_rate=0.1, staircase=True, decay_steps=1000), 1e-5)
        self.adam = tf.compat.v1.train.AdamOptimizer(lr)
        self.hyper_train_op = self.adam.minimize(self.nll, global_step=global_step)
        #self.train_step = self.adam.minimize(self.nll, global_step=global_step)

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        off = rewriter_config_pb2.RewriterConfig.OFF
        config.graph_options.rewrite_options.arithmetic_optimization = off

        self.session = tf.Session(config=config)
        init_op = tf.global_variables_initializer()
        self.session.run(init_op)

        self._saver = tf.compat.v1.train.Saver(
                var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES))
        # define tb summary
        self.tb_mll = self._define_mll_tb()
        self.tb_layers_grad = self._define_layers_grad_tb()

    def save(self, save_dir, name = None):
        if name is None:
            name = ''
        else:
            name = '_' + name
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, 'model') + name
        print('Saving model ....')
        self._saver.save(self.session, save_path)

    def propagate(self, X):
        Fs = [X, ]
        Fds = []
        Fmeans, Fvars = [], []
        Fdmeans, Fdvars = [], []
        downsample = False

        for layer in self.layers:
            if layer.ltype == 'Residual':
                print('Acces residual layer ', layer.ltype)
                mean, var = layer.conditional(Fs[-1])
                #assert mean.shape == Fdmeans[-1].shape, 'means shape sin\'t correct in propagate'
                #mean += Fs[-1] # old mean update
                mean += Fmeans[-1]
                #var += Fvars[-1] + layer.kernel.Kzx(layer.Z, Fs[-1]) # variance update
                #k_fx = self.kernel.K(F,Fs[-1])
                #var = var + Fvars[-1] + k_fx + k_fx.T
            elif layer.ltype == 'Residua-1':
                print('Acces residual-1 layer ', layer.ltype)
                mean, var = layer.conditional(Fs[-1])
                eps = tf.random_normal(tf.shape(mean), dtype=tf.float64)
                F = mean + eps * tf.sqrt(var)
                Fs.append(F)
                Fmeans.append(mean)
                Fvars.append(var)

            elif layer.ltype == 'Residua-2':
                print('Acces residual-2 layer ', layer.ltype)
                mean, var = layer.conditional(Fs[-1])

                if downsample:
                    mean += Fdmeans[-1]
                else:
                    mean += Fmeans[-2]
                downsample = False

            elif layer.ltype == 'downsample':
                print('Acces downsample layer ', layer.ltype)
                downsample = True
                mean, var = layer.conditional(Fs[-1])
                eps = tf.random_normal(tf.shape(mean), dtype=tf.float64)
                F = mean + eps * tf.sqrt(var)
                Fds.append(F)
                Fdmeans.append(mean)
                Fdvars.append(var)

            else:
                mean, var = layer.conditional(Fs[-1])
            print('meand shape ', mean.shape)
            eps = tf.random_normal(tf.shape(mean), dtype=tf.float64)
            F = mean + eps * tf.sqrt(var)
            Fs.append(F)
            Fmeans.append(mean)
            Fvars.append(var)

        return Fs[1:], Fmeans, Fvars

    def predict_y(self, X, S):
        assert S <= len(self.posterior_samples)
        ms, vs = [], []
        for i in range(S):
            feed_dict = {self.X_placeholder: X}
            feed_dict.update(self.posterior_samples[i])
            m, v = self.session.run((self.y_mean, self.y_var), feed_dict=feed_dict)
            ms.append(m)
            vs.append(v)
        return np.stack(ms, 0), np.stack(vs, 0)

    def _define_layers_grad_tb(self):
        grads_and_vars = self.adam.compute_gradients(self.nll)
        for g, v in grads_and_vars:
            tf.compat.v1.summary.histogram(v.name[:-2], v)
            tf.compat.v1.summary.histogram(v.name[:-2] + '_grad', g)

        merged = tf.compat.v1.summary.merge_all()
        return merged

    def _define_mll_tb(self):
        mll = tf.math.reduce_mean(self.log_likelihood)
        tb_mll = tf.compat.v1.summary.scalar('mll', mll)
        return tb_mll
