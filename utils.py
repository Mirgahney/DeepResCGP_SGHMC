import os
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf

import kernels
from conv import utils as conv_utils

import observations


def load_data(dataset, train_pct=1.0, root_dir:str='/home/mirgahney/Projects/datasets'):
    if dataset == "cifar":
        (Xtrain, Ytrain), (Xtest, Ytest) = observations.cifar10(f'{root_dir}/cifar')
        Xtrain = np.transpose(Xtrain, [0, 2, 3, 1])
        Xtest = np.transpose(Xtest, [0, 2, 3, 1])
        mean = Xtrain.mean((0, 1, 2))
        std = Xtrain.std((0, 1, 2))
        Xtrain = (Xtrain - mean) / std
        Xtest = (Xtest - mean) / std
        if train_pct < 1.0:
            Xvalid, Xtrain , Yvalid, Ytrain = train_test_split(Xtrain, Ytrain, stratify=Ytrain, test_size=train_pct)
        print(Xtrain.shape)

    elif dataset == "fashion_mnist":
        (Xtrain, Ytrain), (Xtest, Ytest) = observations.fashion_mnist(f'{root_dir}/fashion_mnist')
        mean = Xtrain.mean(axis=0)
        std = Xtrain.std()
        Xtrain = (Xtrain - mean) / std
        Xtest = (Xtest - mean) / std
        Xtrain = Xtrain.reshape(-1, 28, 28, 1)
        Xtest = Xtest.reshape(-1, 28, 28, 1)
        if train_pct < 1.0:
            Xvalid, Xtrain , Yvalid, Ytrain = train_test_split(Xtrain, Ytrain, stratify=Ytrain, test_size=train_pct)
        print(Xtrain.shape)

    else:
        (Xtrain, Ytrain), (Xtest, Ytest) = observations.mnist(f'{root_dir}/mnist')
        mean = Xtrain.mean(axis=0)
        std = Xtrain.std()
        Xtrain = (Xtrain - mean) / std
        Xtest = (Xtest - mean) / std
        Xtrain = Xtrain.reshape(-1, 28, 28, 1)
        Xtest = Xtest.reshape(-1, 28, 28, 1)
        Xvalid, Xtrain , Ytrain_2, Yvalid = train_test_split(Xtrain, Ytrain, stratify=Ytrain, test_size=train_pct)
        print(Xtrain.shape)

    return (Xtrain, Ytrain), (Xvalid, Yvalid), (Xtest, Ytest)

def get_kernel(kernel_name):
    if kernel_name == 'rbf':
        kernel = kernels.SquaredExponential
    elif kernel_name == 'matern12':
        kernel = kernels.Matern12
    elif kernel_name == 'matern32':
        kernel = kernels.Matern32
    elif kernel_name == 'matern52':
        kernel = kernels.Matern52
    else:
        raise NotImplementedError
    return kernel

def compute_z_inner(X, M, feature_maps_out):
    filter_matrix = np.zeros((5, 5, X.shape[3], feature_maps_out))
    filter_matrix[2, 2, :, :] = 1.0
    convolution = tf.nn.conv2d(X, filter_matrix, [1, 2, 2, 1], "VALID")
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.compat.v1.Session(config=config) as sess:
        filtered = sess.run(convolution)

    return conv_utils.cluster_patches(filtered, M, 5)

def sample_performance_acc(model, POSTERIOR_SAMPLES=25):
    #model.collect_samples(POSTERIOR_SAMPLES, 200)
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


def measure_accuracy(model, Xtest, Ytest, POSTERIOR_SAMPLES=25):
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

def save_result(result_df, save_dir, name = None):
    # add file extention
    if name is None:
        name = '.csv'
    else:
        name = name + '.csv'

    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, 'metrics') + name
    result_df.to_csv(save_path, index=False)