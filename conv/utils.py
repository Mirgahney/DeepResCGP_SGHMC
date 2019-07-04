import numpy as np
from sklearn import cluster
from tqdm import tqdm

def _sample_patches(HW_image, N, patch_size, patch_length):
    out = np.zeros((N, patch_length))
    for i in range(N):
        patch_y = np.random.randint(0, HW_image.shape[0] - patch_size)
        patch_x = np.random.randint(0, HW_image.shape[1] - patch_size)
        out[i] = HW_image[patch_y:patch_y+patch_size, patch_x:patch_x+patch_size].reshape(patch_length)
    return out

def _sample(tensor, count):
    chosen_indices = np.random.choice(np.arange(tensor.shape[0]), count)
    return tensor[chosen_indices]

def cluster_patches(NHWC_X, M, patch_size):
    NHWC = NHWC_X.shape
    patch_length = patch_size ** 2 * NHWC[3]
    # Randomly sample images and patches.
    patches = np.zeros((M, patch_length), dtype=NHWC_X.dtype)
    patches_per_image = 1
    samples_per_inducing_point = 1000
    for i in range(M * samples_per_inducing_point // patches_per_image):
        # Sample a random image, compute the patches and sample some random patches.
        image = _sample(NHWC_X, 1)[0]
        sampled_patches = _sample_patches(image, patches_per_image,
                patch_size, patch_length)
        patches[i*patches_per_image:(i+1)*patches_per_image] = sampled_patches

    k_means = cluster.KMeans(n_clusters=M, n_jobs=-1)
    k_means.fit(patches)
    return k_means.cluster_centers_

class TqdmExtraFormat(tqdm):
    """Provides a `total_time` format parameter"""
    @property
    def format_dict(self):
        d = super(TqdmExtraFormat, self).format_dict
        total_time = d["elapsed"] * (d["total"] or 0) / max(d["n"], 1)
        d.update(total_time=self.format_interval(total_time) + " in total")
        return d
