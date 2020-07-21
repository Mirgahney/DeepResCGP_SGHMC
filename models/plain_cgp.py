import numpy as np
import tensorflow as tf

from sghmc_dgp import DGP, Layer
from conv.layer import ConvLayer, PatchExtractor
from conv.kernels import ConvKernel
import kernels
from likelihoods import MultiClass
from conv import utils as conv_utils
from utils import get_kernel, compute_z_inner

class PlainCGPNet(DGP):
    def __init__(self, X, Y, num_classes:int=10, layers_strcut=[2, 2, 2], window_size:int=100, expansion_factor:int=1,
                 M:int= 384, kernel:str= 'rbf', batch_size:int= 128, lr:float=1e-3, weight_decay:float=2e-3, feature_maps:int=10):
        self.feature_maps = feature_maps
        self.M = M
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.lr = lr
        self.kernel = get_kernel(kernel)
        self.window_size = window_size  #TODO: checkout what this parameter for
        self.expansion_factor = expansion_factor  # additive expansion factor
        self.Reslayers = []
        patches = conv_utils.cluster_patches(X, self.M, 7)
        Z_inner = compute_z_inner(X, self.M, self.feature_maps, 3)
        input_size = X.shape[1:]

        input_size = self._add_first_layer(patches, input_size)
        input_size = self._add_layers(input_size, Z_inner, layers_strcut, X)
        Z = compute_z_inner(X, self.M, input_size[2], input_size[0])
        self._add_last_layer(input_size, Z)

        super().__init__(X.reshape(X.shape[0], np.prod(X.shape[1:])),
                   Y.reshape(Y.shape[0], 1),
                   layers=self.Reslayers,
                   likelihood=MultiClass(self.num_classes),
                   minibatch_size=self.batch_size,
                   window_size=self.window_size,
                   adam_lr=self.lr)
    # total number of layers = len(layers_strcut) + sum(layers_strcut) + 2 (first and last layer)
    # total number of layers = l + n + 2, = n + 5 (l=3)
    def _add_layers(self, input_size, Z, layers_strcut, X):
        k = 1
        for i in layers_strcut:
            for j in range(i):
                base_kernel = self.kernel(input_dim=3 * 3 * input_size[2], lengthscales=2.0)
                print('filter_size 3/1')
                layer = ConvLayer(input_size, patch_size=3, stride=1, base_kernel=base_kernel, Z=Z,
                                  feature_maps_out=input_size[2], pad='SAME', ltype='Plain')
                input_size = (layer.patch_extractor.out_image_height, layer.patch_extractor.out_image_width,
                              input_size[2])
                self.Reslayers.append(layer)

            k = k + self.expansion_factor
            base_kernel = self.kernel(input_dim=1 * 1 * input_size[2], lengthscales=2.0)
            print(f'filter_size 1/2-{k}')
            output_featuers = k*self.feature_maps
            Z = compute_z_inner(X, self.M, output_featuers, 3) if self.expansion_factor >= 1 else Z
            layer = ConvLayer(input_size, patch_size=1, stride=2, base_kernel=base_kernel, Z=Z,
                              feature_maps_out=output_featuers, pad='VALID', ltype='Plain')
            input_size = (layer.patch_extractor.out_image_height, layer.patch_extractor.out_image_width,
                          output_featuers)
            self.Reslayers.append(layer)

        return input_size

    def _add_first_layer(self, Z, input_size):
        base_kernel = self.kernel(input_dim=7 * 7 * input_size[2], lengthscales=2.0)
        print('filter_size 7/1')
        layer = ConvLayer(input_size, patch_size=7, stride=1, base_kernel=base_kernel, Z=Z,
                          feature_maps_out=self.feature_maps, pad='VALID', ltype='Plain')
        self.Reslayers.append(layer)
        input_size = (layer.patch_extractor.out_image_height, layer.patch_extractor.out_image_width, self.feature_maps)
        return input_size

    def _add_last_layer(self, input_size, Z):
        rbf = self.kernel(input_dim=input_size[0] * input_size[1] * input_size[2], lengthscales=2.0)
        patch_extractor = PatchExtractor(input_size, filter_size=input_size[0], feature_maps=input_size[2], stride=1)
        conv_kernel = ConvKernel(rbf, patch_extractor)
        layer = Layer(conv_kernel, self.num_classes, Z)
        self.Reslayers.append(layer)

    def propagate(self, X):
        Fs = [X, ]
        Fmeans, Fvars = [], []

        for layer in self.layers:
            mean, var = layer.conditional(Fs[-1])
            print('meand shape ', mean.shape)
            eps = tf.random_normal(tf.shape(mean), dtype=tf.float64)
            F = mean + eps * tf.sqrt(var)
            Fs.append(F)
            Fmeans.append(mean)
            Fvars.append(var)

        return Fs[1:], Fmeans, Fvars

def PlainCGPNet6(cfg, Xtrain, Ytrain, num_classes=10, window_size=100,
                pretrained=False, progress=True, **kwargs):
    r"""PlainCGPNet-6 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    #layers_strcut=[0, 2, 0] -> PlainCGPNet7
    return PlainCGPNet(cfg, Xtrain, Ytrain, num_classes=num_classes, layers_strcut=[0, 1, 0], window_size=window_size,
                   **kwargs)

def PlainCGPNet8(cfg, Xtrain, Ytrain, num_classes=10, window_size=100,
                pretrained=False, progress=True, **kwargs):
    r"""PlainCGPNet-8 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    # layers_strcut=[2, 1, 0] -> PlainCGPNet8 #TODO: try this cobination
    return PlainCGPNet(cfg, Xtrain, Ytrain, num_classes=num_classes, layers_strcut=[1, 1, 1], window_size=window_size,
                   **kwargs)

def PlainCGPNet11(cfg, Xtrain, Ytrain, num_classes=10, window_size=100,
                pretrained=False, progress=True, **kwargs):
    r"""PlainCGPNet-11 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return PlainCGPNet(cfg, Xtrain, Ytrain, num_classes=num_classes, layers_strcut=[2, 2, 2], window_size=window_size,
                   **kwargs)

def PlainCGPNet17(cfg, Xtrain, Ytrain, num_classes=10, window_size=100,
                pretrained=False, progress=True, **kwargs):
    r"""PlainCGPNet-17 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return PlainCGPNet(cfg, Xtrain, Ytrain, num_classes=num_classes, layers_strcut=[3, 6, 3], window_size=window_size,
                   **kwargs)