# Credit to GPFlow.

import tensorflow as tf
import numpy as np

class Gaussian(object):
    def logdensity(self, x, mu, var):
        return -0.5 * (np.log(2 * np.pi) + tf.log(var) + tf.square(mu-x) / var)

    def __init__(self, variance=1.0, **kwargs):
        self.variance = tf.exp(tf.Variable(np.log(variance), dtype=tf.float64, name='lik_log_variance'))

    def logp(self, F, Y):
        return self.logdensity(Y, F, self.variance)

    def conditional_mean(self, F):
        return tf.identity(F)

    def conditional_variance(self, F):
        return tf.fill(tf.shape(F), tf.squeeze(self.variance))

    def predict_mean_and_var(self, Fmu, Fvar):
        return tf.identity(Fmu), Fvar + self.variance

    def predict_density(self, Fmu, Fvar, Y):
        return self.logdensity(Y, Fmu, Fvar + self.variance)

    def variational_expectations(self, Fmu, Fvar, Y):
        return -0.5 * np.log(2 * np.pi) - 0.5 * tf.log(self.variance) \
               - 0.5 * (tf.square(Y - Fmu) + Fvar) / self.variance

class MultiClassInvLink(object):
    # From https://github.com/gpflow/gpflow
    def __init__(self, num_classes):
        self.epsilon = 1e-3
        self.num_classes = num_classes
        self.epsilon_k1 = self.epsilon / (self.num_classes - 1.0)

    def __call__(self, Fmu):
        return tf.one_hot(Fmu.argmax(axis=1), self.num_classes, 1 - self.epsilon, self.epsilon)

    def prob_is_largest(self, Y, mu, var, gh_x, gh_w):
        float_type = mu.dtype

        Y = tf.cast(Y, tf.int64)
        # work out what the mean and variance is of the indicated latent function.
        oh_on = tf.cast(tf.one_hot(tf.reshape(Y, (-1,)), self.num_classes, 1., 0.), float_type)
        mu_selected = tf.reduce_sum(oh_on * mu, 1)
        var_selected = tf.reduce_sum(oh_on * var, 1)

        # generate Gauss Hermite grid
        X = tf.reshape(mu_selected, (-1, 1)) + gh_x * tf.reshape(
            tf.sqrt(tf.clip_by_value(2. * var_selected, 1e-10, np.inf)), (-1, 1))

        # compute the CDF of the Gaussian between the latent functions and the grid (including the selected function)
        dist = (tf.expand_dims(X, 1) - tf.expand_dims(mu, 2)) / tf.expand_dims(
            tf.sqrt(tf.clip_by_value(var, 1e-10, np.inf)), 2)
        cdfs = 0.5 * (1.0 + tf.erf(dist / np.sqrt(2.0)))

        cdfs = cdfs * (1 - 2e-4) + 1e-4

        # blank out all the distances on the selected latent function
        oh_off = tf.cast(tf.one_hot(tf.reshape(Y, (-1,)), self.num_classes, 0., 1.), float_type)
        cdfs = cdfs * tf.expand_dims(oh_off, 2) + tf.expand_dims(oh_on, 2)

        # take the product over the latent functions, and the sum over the GH grid.
        return tf.matmul(tf.reduce_prod(cdfs, reduction_indices=[1]), tf.reshape(gh_w / np.sqrt(np.pi), (-1, 1)))


class MultiClass(object):
    # Modified from https://github.com/gpflow/gpflow
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.gauss_points = 20
        self.invlink = MultiClassInvLink(num_classes)

    def logp(self, F, Y):
        # F: N x D
        # Y: N x 1
        y_hat = F.argmax(axis=1)
        correct = tf.equal(y_hat[:, None], Y)
        Y_shape = tf.shape(Y)
        ones = tf.ones(Y_shape, dtype=tf.float64) - self.invlink.epsilon
        zeros = tf.zeros(Y_shape, dtype=tf.float64) + self.invlink.epsilon_k1
        p = tf.where(correct, ones, zeros)
        return tf.log(p)

    def _predict_mean(self, Fmu, Fvar):
        possible_outputs = [tf.fill(tf.stack([tf.shape(Fmu)[0], 1]), np.array(i, dtype=np.int64)) for i in
                            range(self.num_classes)]
        ps = [self._density(Fmu, Fvar, po) for po in possible_outputs]
        ps = tf.transpose(tf.stack([tf.reshape(p, (-1,)) for p in ps]))
        return ps

    def _density(self, Fmu, Fvar, Y):
        gauss_points, gauss_weights = np.polynomial.hermite.hermgauss(self.gauss_points)
        p = self.invlink.prob_is_largest(Y, Fmu, Fvar, gauss_points, gauss_weights)
        return p * (1.0 - self.invlink.epsilon) + (1.0 - p) * self.invlink.epsilon_k1

    def predict_density(self, Fmu, Fvar, Y):
        return tf.log(self._density(Fmu, Fvar, Y))

    def conditional_mean(self, Fmu):
        return self.invlink(Fmu)

    def predict_mean_and_var(self, Fmu, Fvar):
        mean = self._predict_mean(Fmu, Fvar)
        return mean, mean - tf.square(mean)

