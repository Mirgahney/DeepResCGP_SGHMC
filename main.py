import os
import numpy as np
import pandas as pd
from sklearn import cluster
import tensorflow as tf

from models import RegressionModel, ClassificationModel
from sghmc_dgp import DGP, Layer
from conv.layer import ConvLayer, PatchExtractor
from conv.kernels import ConvKernel, AdditivePatchKernel
import kernels
from likelihoods import MultiClass
from conv import utils as conv_utils

import argparse
import observations

parser = argparse.ArgumentParser()
parser.add_argument('--feature_maps', default=10, type=int)
parser.add_argument('-M', default=64, type=int)
parser.add_argument('--batch-size', default=128, type=int)
parser.add_argument('--iterations', default=35000, type=int)
parser.add_argument('--cifar', action='store_true')
parser.add_argument('--layers', default=3, type=int)
parser.add_argument('--lr', default=1e-4, type=float)
parser.add_argument('--load', type=str)
parser.add_argument('--out', default='results', type=str)

flags = parser.parse_args()

def load_data():
    if flags.cifar:
        (Xtrain, Ytrain), (Xtest, Ytest) = observations.cifar10('/tmp/cifar')
        Xtrain = np.transpose(Xtrain, [0, 2, 3, 1])
        Xtest = np.transpose(Xtest, [0, 2, 3, 1])
        mean = Xtrain.mean((0, 1, 2))
        std = Xtrain.std((0, 1, 2))
        Xtrain = (Xtrain - mean) / std
        Xtest = (Xtest - mean) / std
    else:
        (Xtrain, Ytrain), (Xtest, Ytest) = observations.mnist('/tmp/mnist')
        mean = Xtrain.mean(axis=0)
        std = Xtrain.std()
        Xtrain = (Xtrain - mean) / std
        Xtest = (Xtest - mean) / std
        Xtrain = Xtrain.reshape(-1, 28, 28, 1)
        Xtest = Xtest.reshape(-1, 28, 28, 1)
    return (Xtrain, Ytrain), (Xtest, Ytest)

(Xtrain, Ytrain), (Xtest, Ytest) = load_data()

def compute_z_inner(X, M, feature_maps_out):
    filter_matrix = np.zeros((5, 5, X.shape[3], feature_maps_out))
    filter_matrix[2, 2, :, :] = 1.0
    convolution = tf.nn.conv2d(X, filter_matrix,
            [1, 2, 2, 1],
            "VALID")

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        filtered = sess.run(convolution)

    return conv_utils.cluster_patches(filtered, M, 5)

layers = []
input_size = Xtrain.shape[1:]

Z_inner = compute_z_inner(Xtrain, flags.M, flags.feature_maps)
patches = conv_utils.cluster_patches(Xtrain, flags.M, 10)

strides = (2, 1, 1, 1)
filters = (5, 3, 5, 5)
for layer in range(0, flags.layers):
    if layer == 0:
        Z = patches
    else:
        Z = Z_inner
    filter_size = filters[layer]
    stride = strides[layer]
    if layer != flags.layers-1:

        base_kernel = kernels.SquaredExponential(input_dim=filter_size*filter_size*input_size[2], lengthscales=2.0)
        print('filter_size ', filter_size)
        if filter_size == 3:
            pad = 'SAME'
            ltype = 'Residual'
        else:
            pad = 'VALID'
            ltype = 'Plain'
        layer = ConvLayer(input_size, patch_size=filter_size, stride=stride, base_kernel=base_kernel, Z=Z, feature_maps_out=flags.feature_maps, pad=pad, ltype =ltype)


        input_size = (layer.patch_extractor.out_image_height, layer.patch_extractor.out_image_width, flags.feature_maps)
    else:
        rbf = kernels.SquaredExponential(input_dim=filter_size*filter_size*flags.feature_maps, lengthscales=2.0)
        patch_extractor = PatchExtractor(input_size, filter_size=filter_size, feature_maps=10, stride=stride)
        conv_kernel = ConvKernel(rbf, patch_extractor)
        layer = Layer(conv_kernel, 10, Z)

    layers.append(layer)


model = DGP(Xtrain.reshape(Xtrain.shape[0], np.prod(Xtrain.shape[1:])),
        Ytrain.reshape(Ytrain.shape[0], 1),
        layers=layers,
        likelihood=MultiClass(10),
        minibatch_size=flags.batch_size,
        window_size=100,
        adam_lr=flags.lr)

if flags.load is not None:
    print("Loading parameters")
    checkpoint = tf.train.latest_checkpoint(flags.load)
    model._saver.restore(model.session, checkpoint)

POSTERIOR_SAMPLES = 25

def sample_performance_acc(model):
    model.collect_samples(POSTERIOR_SAMPLES, 200)
    X_batch, Y_batch = model.get_minibatch()
    batch_size = 32
    batches = X_batch.shape[0] // batch_size
    correct = 0
    for i in range(batches + 1):
        slicer = slice(i * batch_size, (i+1) * batch_size)
        X = X_batch[slicer]
        Y = Y_batch[slicer]
        mean, var = model.predict_y(X.reshape(X.shape[0], np.prod(X.shape[1:])), POSTERIOR_SAMPLES)
        prediction = mean.mean(axis=0).argmax(axis=1)
        correct += (prediction == Y).sum()
    return correct / Y_batch.shape[0]

POSTERIOR_SAMPLES = 25

def measure_accuracy(model):
    model.collect_samples(POSTERIOR_SAMPLES, 200)
    batch_size = 32
    batches = Xtest.shape[0] // batch_size
    correct = 0
    for i in range(batches + 1):
        slicer = slice(i * batch_size, (i+1) * batch_size)
        X = Xtest[slicer]
        Y = Ytest[slicer]
        mean, var = model.predict_y(X.reshape(X.shape[0], np.prod(X.shape[1:])), POSTERIOR_SAMPLES)
        prediction = mean.mean(axis=0).argmax(axis=1)
        correct += (prediction == Y).sum()
    return correct / Ytest.shape[0]

# create a data frame to save intermediate resulr
result_df = pd.DataFrame(columns=['step', 'mll'])#, 'accuracy'])

# progress bar information
tdqm = conv_utils.TqdmExtraFormat
mll_max = -np.inf
accuracy_list = [0]
mll_list = [mll_max]

for i in tdqm(
      range(flags.iterations), ascii=" .oO0",
      bar_format="{total_time}: {percentage:.0f}%|{bar}{r_bar}"):
    model.sghmc_step()
    model.train_hypers()
    print("Iteration", i, end='\r')
    if i % 500 == 1:
        print("Iteration {}".format(i))
        mll = model.print_sample_performance()
        if i >= 17500:
            if np.round(mll - mll_max, decimals = 12) > 0:
                accuracy = measure_accuracy(model)
                mll_max = mll

                print('MLL increased ({:.7f} --> {:.7f}). Updating values ....'.format(mll_list[-1], mll_max))
                print('Update accuracy ({:.4f} --> {:.4f}). Updating values ....'.format(accuracy_list[-1], accuracy))

                accuracy_list.append(accuracy)
                mll_list.append(mll)

                model_name = str(i) + '_' + str(accuracy) + '_' + str(mll)
                model.save(flags.out, name = model_name)                

        result_df = result_df.append({'step': i, 'mll': mll}, ignore_index=True)

    if i % 10000 == 0:
        model.save(flags.out, name = str(i))

# append the final accuracy
# accuracy = sample_performance_acc(model)
mll = model.print_sample_performance()
result_df = result_df.append({'step': flags.iterations, 'mll': mll}, ignore_index=True)


def save_result(result_df, save_dir, name = None):
    # add file extention
    if name is None:
        name = '.csv'
    else:
        name = name + '.csv'

    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, 'metrics') + name
    result_df.to_csv(save_path, index=False)


model.save(flags.out)

accuracy = measure_accuracy(model)
acc_ind = np.argmax(accuracy_list)

print("Model Test accuracy:", accuracy)
print("Model Best Test accuracy: {:.5f} got with mll: {:.7f}".format(np.max(accuracy_list), mll_list[acc_ind]))

result_df = result_df.append({'step': flags.iterations, 'mll': mll}, ignore_index=True)
save_result(result_df, flags.out)

acc_mll_df = pd.DataFrame(accuracy_list, columns=['accuracy'])
acc_mll_df['mll'] = mll_list
save_result(acc_mll_df, flags.out, name = '_mll_accuracy')
