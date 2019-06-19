import numpy as np
import tensorflow as tf
from sghmc_dgp import Layer
import conditionals

jitter = 1e-3

class PatchExtractor(object):
    """Extracts patches including color channels from images."""
    def __init__(self, input_size, filter_size, feature_maps, stride=1, pad = 'VALID'):
        self.input_size = list(input_size)
        self.stride = stride
        self.pad = pad
        self.dilation = 1
        self.filter_size = filter_size
        self.feature_maps = feature_maps
        self.patch_shape = [filter_size, filter_size]
        self.pad_value = self._padd_with()
        self.patch_count = self._patch_count()
        self.patch_length = self._patch_length()
        self.out_image_height, self.out_image_width = self._out_image_size()

    def _extract_image_patches(self, NHWC_X):
        # returns: N x H x W x C * P
        return tf.extract_image_patches(NHWC_X,
                [1, self.filter_size, self.filter_size, 1],
                [1, self.stride, self.stride, 1],
                [1, self.dilation, self.dilation, 1],
                "VALID")

    def patches_PNL(self, NHWC_X):
        N = tf.shape(NHWC_X)[0]
        NHWK_patches = self._extract_image_patches(NHWC_X)
        NPL_patches = tf.reshape(NHWK_patches, [N, self.patch_count, self.patch_length])
        return tf.transpose(NPL_patches, [1, 0, 2])

    def patches(self, NHWC_X):
        """extract_patches

        :param X: N x height x width x feature_maps
        :returns N x patch_count x patch_length
        """
        N = tf.shape(NHWC_X)[0]
        NHWK_patches = self._extract_image_patches(NHWC_X)
        return tf.reshape(NHWK_patches, [N, self.patch_count, self.patch_length])

    def _patch_length(self):
        """The number of elements in a patch."""
        return self.feature_maps * np.prod(self.patch_shape)

    def _patch_count(self):
        """The amount of patches in one image."""
        height, width = self._out_image_size()
        return height * width

    def _out_image_size(self):
        height = (self.input_size[0] - self.patch_shape[0] + 2*self.pad_value) // self.stride + 1
        width = (self.input_size[1] - self.patch_shape[1] + 2*self.pad_value) // self.stride + 1
        return height, width

    def _padd_with(self):

    	if self.pad == 'SAME':
    		pad = self.filter_size -2 
    	else:
    		pad = 0
    	return pad
   
class MultiOutputConvKernel(object):
    def __init__(self, base_kernel, patch_count):
        self.base_kernel = base_kernel
        self.patch_count = patch_count

    def Kzz(self, ML_Z):
        M = tf.shape(ML_Z)[0]
        return self.base_kernel.K(ML_Z) + tf.eye(M, dtype=tf.float64) * jitter

    def Kzf(self, ML_Z, PNL_patches):
        """ Returns covariance between inducing points and input.
        Output shape: patch_count x M x N
        """
        def patch_covariance(NL_patches):
            # Returns covariance matrix of size M x N.
            return self.base_kernel.K(ML_Z, NL_patches)

        PMN_Kzx = tf.map_fn(patch_covariance, PNL_patches, parallel_iterations=self.patch_count)
        return PMN_Kzx

    def Kff(self, PNL_patches):
        """Kff returns auto covariance of the input.
        :return: O (== P) x N x N covariance matrices.
        """
        def patch_auto_covariance(NL_patches):
            # Returns covariance matrix of size N x N.
            return self.base_kernel.K(NL_patches)
        return tf.map_fn(patch_auto_covariance, PNL_patches, parallel_iterations=self.patch_count)

    def Kdiag(self, PNL_patches):
        """
        :return: O X N diagonals of the covariance matrices.
        """
        def Kdiag(NL_patch):
            ":return: N diagonal of covariance matrix."
            return self.base_kernel.Kdiag(NL_patch)
        return tf.map_fn(Kdiag, PNL_patches, parallel_iterations=self.patch_count)

class ConvLayer(Layer):
    def __init__(self, input_size, patch_size, stride, base_kernel, Z, feature_maps_out=1, pad=0):
        self.Z = tf.Variable(Z, dtype=tf.float64, name='Z')
        self.base_kernel = base_kernel

        self.patch_extractor = PatchExtractor(input_size, patch_size, input_size[2], stride=stride, pad=pad)

        self.feature_maps_in = input_size[2]
        self.feature_maps_out = feature_maps_out
        self.pad = pad

        self.patch_count = self.patch_extractor.patch_count
        self.patch_length = self.patch_extractor.patch_length
        self.num_outputs = self.patch_count * self.feature_maps_out

        self.conv_kernel = MultiOutputConvKernel(base_kernel, patch_count=self.patch_count)

        self.num_inducing = Z.shape[0]

        self.U = tf.Variable(np.zeros((self.num_inducing, feature_maps_out)), dtype=tf.float64, trainable=False, name='U')

        self.Lz = tf.placeholder_with_default(self._compute_Lu(self.Z),
                shape=[Z.shape[0], Z.shape[0]])

    def _compute_Lu(self, Z):
        MM_Kzz =  self.conv_kernel.Kzz(self.Z)
        return tf.cholesky(MM_Kzz)

    def cacheable_params(self):
        return [self.Lz]

    def conditional(self, ND_X, full_cov=False):
        """Computes the conditional distribution. Returns a tensor of size N x D."""
        N = tf.shape(ND_X)[0]
        NHWC_X = tf.reshape(ND_X, [N,
            self.patch_extractor.input_size[0],
            self.patch_extractor.input_size[1],
            self.feature_maps_in])
        print('NHWC_X without pad ', NHWC_X.shape)
        if self.pad_value != 0:
            # npad = ((0,0),(1,1),(1,1),(0,0))
            # NHWC_X = np.pad(NHWC_X, pad_width=npad, mode='constant', constant_values=0) 
            npad = tf.constant([[0,0],[self.pad_value, self.pad_value],[self.pad_value, self.pad_value],[0,0]])
            NHWC_X = tf.pad(NHWC_X, npad, mode='CONSTANT')
            print('NHWC_X after pad ', NHWC_X.shape)
        
        PNL_patches = self.patch_extractor.patches_PNL(NHWC_X)

        PMN_Kzf = self.conv_kernel.Kzf(self.Z, PNL_patches)

        if full_cov:
            Knn = self.conv_kernel.Kff(PNL_patches)
        else:
            Knn = self.conv_kernel.Kdiag(PNL_patches)

        mean, var = conditionals.multiple_output_conditional(PMN_Kzf, self.Lz, Knn, self.U)

        if full_cov:
            # var: R x P x N x N
            var = tf.transpose(var, [2, 3, 1, 0])
            var = tf.reshape(var, [N, N, self.num_outputs])
        else:
            # var: R x P x N
            var = tf.transpose(var, [2, 1, 0])
            var = tf.reshape(var, [N, self.num_outputs])

        mean = tf.reshape(mean, [N, self.num_outputs])
        return mean, var



