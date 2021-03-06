import os
import numpy as np
import pandas as pd
from collections import deque
from sklearn.model_selection import train_test_split
import tensorflow as tf

from sghmc_dgp import DGP, Layer
from conv.layer import ConvLayer, PatchExtractor
from conv.kernels import ConvKernel
import kernels
from likelihoods import MultiClass
from conv import utils as conv_utils

import argparse
import observations

#TODO:
# 1. try variance update equation and see it's effect
# 2. try different kernels
# 3. go deeper!!
# 4. try challanging datasets ImageNet 32, [corrupted dataset]
# 5. try different train step for sghmc_step and train_hypers *****

parser = argparse.ArgumentParser()
parser.add_argument('--feature_maps', default=10, type=int)
parser.add_argument('-M', default=64, type=int)
parser.add_argument('--batch-size', default=128, type=int)
parser.add_argument('--iterations', default=100000, type=int)
parser.add_argument('--dataset', default = "mnist", choices=['mnist', 'fashion_mnist', 'cifar'], type=str)
parser.add_argument('--layers', default=3, type=int)
parser.add_argument('--lr', default=1e-4, type=float)
parser.add_argument('--load', type=str)
parser.add_argument('--out', default='results', type=str)
parser.add_argument('--arch', default='plain', type=str)
parser.add_argument('--kernel', default='rbf', choices=['rbf', 'matern12', 'matern32', 'matern52'], type=str)
parser.add_argument('--train-pct', default=1.0, type=float)

flags = parser.parse_args()

def load_data():
    if flags.dataset == "cifar":
        (Xtrain, Ytrain), (Xtest, Ytest) = observations.cifar10('/home/mirgahney/Projects/datasets/cifar')
        Xtrain = np.transpose(Xtrain, [0, 2, 3, 1])
        Xtest = np.transpose(Xtest, [0, 2, 3, 1])
        mean = Xtrain.mean((0, 1, 2))
        std = Xtrain.std((0, 1, 2))
        Xtrain = (Xtrain - mean) / std
        Xtest = (Xtest - mean) / std
        Xtrain_2, Xtrain , Ytrain_2, Ytrain = train_test_split(Xtrain, Ytrain, stratify=Ytrain, test_size=flags.train_pct)
        print(Xtrain.shape)

    elif flags.dataset == "fashion_mnist":
        (Xtrain, Ytrain), (Xtest, Ytest) = observations.fashion_mnist('/home/mirgahney/Projects/datasets/fashion_mnist')
        mean = Xtrain.mean(axis=0)
        std = Xtrain.std()
        Xtrain = (Xtrain - mean) / std
        Xtest = (Xtest - mean) / std
        Xtrain = Xtrain.reshape(-1, 28, 28, 1)
        Xtest = Xtest.reshape(-1, 28, 28, 1)
        if flags.train_pct < 1.0:
            Xtrain_2, Xtrain , Ytrain_2, Ytrain = train_test_split(Xtrain, Ytrain, stratify=Ytrain, test_size=flags.train_pct)
        print(Xtrain.shape)

    else:
        (Xtrain, Ytrain), (Xtest, Ytest) = observations.mnist('/home/mirgahney/Projects/datasets/mnist')
        mean = Xtrain.mean(axis=0)
        std = Xtrain.std()
        Xtrain = (Xtrain - mean) / std
        Xtest = (Xtest - mean) / std
        Xtrain = Xtrain.reshape(-1, 28, 28, 1)
        Xtest = Xtest.reshape(-1, 28, 28, 1)
        Xtrain_2, Xtrain , Ytrain_2, Ytrain = train_test_split(Xtrain, Ytrain, stratify=Ytrain, test_size=flags.train_pct)
        print(Xtrain.shape)

    return (Xtrain, Ytrain), (Xtest, Ytest)

(Xtrain, Ytrain), (Xtest, Ytest) = load_data()

def compute_z_inner(X, M, feature_maps_out):
    filter_matrix = np.zeros((5, 5, X.shape[3], feature_maps_out))
    filter_matrix[2, 2, :, :] = 1.0
    convolution = tf.nn.conv2d(X, filter_matrix,
            [1, 2, 2, 1],
            "VALID")

    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.compat.v1.Session(config=config) as sess:
        filtered = sess.run(convolution)

    return conv_utils.cluster_patches(filtered, M, 5)

if flags.kernel == 'rbf':
    kernel = kernels.SquaredExponential
elif flags.kernel == 'matern12':
    kernel = kernels.Matern12
elif flags.kernel == 'matern32':
    kernel = kernels.Matern32
elif flags.kernel == 'matern52':
    kernel = kernels.Matern52
else:
    raise NotImplementedError

save_dir = f'run/{flags.dataset}/l{flags.layers}_fm_{flags.feature_maps}_M{flags.M}_K{kernel}_lr{flags.lr}'

layers = []
input_size = Xtrain.shape[1:]

Z_inner = compute_z_inner(Xtrain, flags.M, flags.feature_maps)
patches = conv_utils.cluster_patches(Xtrain, flags.M, 10)

if flags.layers == 3:
    strides = (2, 1, 1)
    filters = (5, 3, 5)
elif flags.layers == 6:
    strides = (2, 1, 1, 1, 1, 1)
    filters = (5, 3, 3, 3, 3, 5)
elif flags.layers == 12: #TODO: add non-residual layers with 3 filter size replacing 5x5, we can scall up the layers to 18
    strides = (2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1)
    filters = (5, 3, 3, 3, 3, 5, 3, 3, 3, 3, 5, 5)
else:
    raise Exception("undefined number of layers")

for layer in range(0, flags.layers):
    if layer == 0:
        Z = patches
    else:
        Z = Z_inner
    filter_size = filters[layer]
    stride = strides[layer]
    if layer != flags.layers-1:

        base_kernel = kernel(input_dim=filter_size*filter_size*input_size[2], lengthscales=2.0)
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
        rbf = kernel(input_dim=filter_size*filter_size*flags.feature_maps, lengthscales=2.0)
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

def Basic_Block(input_size, inplanes = None, planes = 10, stride = 1, Z=None, downsample = None):
    expansion = 1
    __constants__ = ['downsample']

    layers = []

    if downsample is not None:
        layers.append(downsample)

    base_kernel = kernels.SquaredExponential(input_dim=3*3*input_size[2], lengthscales=2.0)
    layer = ConvLayer(input_size, patch_size=3, stride=stride, base_kernel=base_kernel, Z=Z, feature_maps_out=planes, pad='SAME', ltype ='Residua-1')
    input_size = (layer.patch_extractor.out_image_height, layer.patch_extractor.out_image_width, planes)
    layers.append(layer)

    base_kernel = kernels.SquaredExponential(input_dim=3*3*input_size[2], lengthscales=2.0)
    layer = ConvLayer(input_size, patch_size=3, stride=1, base_kernel=base_kernel, Z=Z, feature_maps_out=planes, pad='SAME', ltype ='Residual-2')
    input_size = (layer.patch_extractor.out_image_height, layer.patch_extractor.out_image_width, planes)
    layers.append(layer)

    return layers, input_size

class ResCGPNet():
    def __init__(self, block, layers, num_classes=1000, input_size=input_size, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(ResCGPNet, self).__init__()
        
#         if norm_layer is None:
#             norm_layer = nn.BatchNorm2d
        
        self._norm_layer = norm_layer

        self.inplanes = 8
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        input_size = Xtrain.shape[1:]
        self.Reslayers = []
        # current impelemtation with fixed output feature maps for all alyers to user inputed argument 10 
        # need to replace it with self.inplanes but that requirs artucheture search which isn't valid for now
        Z = conv_utils.cluster_patches(Xtrain, flags.M, self.inplanes)
        base_kernel = kernels.SquaredExponential(input_dim=7*7*input_size[2], lengthscales=2.0)
        layer = ConvLayer(input_size, patch_size=7, stride=2, base_kernel=base_kernel, Z=Z, feature_maps_out=self.inplanes, pad=3, ltype ='Plain') # change stride 2-> 1 for cifar-10
        input_size = (layer.patch_extractor.out_image_height, layer.patch_extractor.out_image_width, self.inplanes)
        self.Reslayers.append(layer)

        #self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1) # need to compensate for the pooling calulate the size and adjuest the conGP acoordingly 
        Z = compute_z_inner(Xtrain, flags.M, 8)
        layers_, input_size = self._make_layer(input_size, block, 8, layers[0], Z)
        self.Reslayers += layers_
#         set_trace()
        Z = compute_z_inner(Xtrain, flags.M, 16)
        layers_, input_size = self._make_layer(input_size, block, 16, layers[1], Z, stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.Reslayers += layers_
        Z = compute_z_inner(Xtrain, flags.M, 32)
        layers_, input_size = self._make_layer(input_size, block, 32, layers[2], Z, stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.Reslayers += layers_
        Z = compute_z_inner(Xtrain, flags.M, 64)
        layers_, input_size = self._make_layer(input_size, block, 64, layers[3],Z,  stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.Reslayers += layers_
#         Z = compute_z_inner(Xtrain, flags.M, 16)
        rbf = kernels.SquaredExponential(input_dim=input_size[0]*input_size[1]*flags.feature_maps, lengthscales=2.0) # filter_size is equal to all input size to memic the Linear layer
        patch_extractor = PatchExtractor(input_size, filter_size=input_size[0], feature_maps=num_classes, stride=stride)
        conv_kernel = ConvKernel(rbf, patch_extractor)
        layer = Layer(conv_kernel, num_classes, Z)
        self.Reslayers.append(layer)
#         set_trace()
    def _make_layer(self, input_size, block, planes, blocks, Z, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * 1: #block.expansion: stop expnation for now and set it to 1

            base_kernel = kernels.SquaredExponential(input_dim=1*1*input_size[2], lengthscales=2.0)
            downsample = ConvLayer(input_size, patch_size=1, stride=stride, base_kernel=base_kernel, Z=Z, feature_maps_out=planes, pad='VALID', ltype ='downsample')
            input_size = (downsample.patch_extractor.out_image_height, downsample.patch_extractor.out_image_width, planes)

        layers = []

        layers_block, input_size = block(input_size, self.inplanes, planes, stride, Z, downsample)

        layers += layers_block

        self.inplanes = planes * 1 #block.expansion stop expnation for now and set it to 1
        
        for _ in range(1, blocks):
            layers_block, input_size = block(input_size, self.inplanes, planes, stride, Z, downsample)
            layers += layers_block

        return layers, input_size
    
    def get_model(self):
        return DGP(Xtrain.reshape(Xtrain.shape[0], np.prod(Xtrain.shape[1:])),
                        Ytrain.reshape(Ytrain.shape[0], 1),
                        layers=self.Reslayers,
                        likelihood=MultiClass(10),
                        minibatch_size=flags.batch_size,
                        window_size=100,
                        adam_lr=flags.lr)

if flags.arch == 'ResNet':
    model = ResCGPNet(Basic_Block,[2,2,2,2], num_classes=10).get_model()
    
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
accuracy_list = []
mll_list_que = deque([mll_max, mll_max, mll_max])
best_model_que = deque([model, model, model])

writer = tf.compat.v1.summary.FileWriter(f'{save_dir}', tf.get_default_graph())

for i in tdqm(
      range(flags.iterations), ascii=" .oO0",
      bar_format="{total_time}: {percentage:.0f}%|{bar}{r_bar}"):
    if i == 0:
        model.sghmc_step()
        summary = model.train_hypers(tb=True)
        print("Iteration", i, end='\r')
        mll, sum_mll = model.print_sample_performance(tb=True)
        print(f"MLL: {mll}")
        # mll = struct.unpack('fff', sum_mll)[2]
        # set_trace()
        writer.add_summary(summary, global_step=i)
        writer.add_summary(sum_mll, global_step=i)
    else:
        model.sghmc_step()
        model.train_hypers()
        print("Iteration", i, end='\r')
        mll = model.print_sample_performance()
        print(f"MLL: {mll}")

    if i >= 1000:#17500:
        if np.round(mll - mll_max, decimals=5) > 0:
            # accuracy = measure_accuracy(model)
            mll_max = mll

            print('MLL increased ({:.7f} --> {:.7f}). Updating values ....'.format(mll_list_que[-1], mll_max))
            mll_list_que.append(mll)
            best_model_que.append(model)  # append best model so far
            best_model_que.popleft()  # remove worst model so far
            mll_list_que.popleft()

# save best model
model_name = '_bestmodel_' + str(mll_list_que[-1])
best_model_que[-1].save(save_dir, name=model_name)

# append the final accuracy
# accuracy = sample_performance_acc(model)
#mll = model.print_sample_performance()
#result_df = result_df.append({'step': flags.iterations, 'mll': mll}, ignore_index=True)


def save_result(result_df, save_dir, name = None):
    # add file extention
    if name is None:
        name = '.csv'
    else:
        name = name + '.csv'

    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, 'metrics') + name
    result_df.to_csv(save_path, index=False)

# save final model
model_name = str(i) + '_' + str(mll)
model.save(save_dir, name=model_name)

#result_df = result_df.append({'step': flags.iterations, 'mll': mll}, ignore_index=True)
#save_result(result_df, flags.out)
#set_trace()
accuracy = measure_accuracy(model)
# loop over model
for m in tdqm(best_model_que):
    acc = measure_accuracy(m)
    accuracy_list.append(acc)

acc_ind = np.argmax(accuracy_list)

print("Model Test accuracy:", accuracy)
print("Model Best Test accuracy: {:.5f} got with mll: {:.7f}".format(np.max(accuracy_list), mll_list_que[acc_ind]))



acc_mll_df = pd.DataFrame(accuracy_list, columns=['accuracy'])
acc_mll_df['mll'] = mll_list_que
save_result(acc_mll_df, save_dir, name = '_mll_accuracy')

# send finish email
#os.system('python3 SendEmail.py --acc {:.4f} --mll {:.5f}'.format(np.max(accuracy_list), mll_list[-3:][acc_ind]))
