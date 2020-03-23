import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from gap_statistic import OptimalK
from sklearn import cluster
from sklearn.cluster import MeanShift
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
    patches_per_image = 1
    # samples_per_inducing_point = 1000
    samples_per_inducing_point = 100
    patches = np.zeros((M * samples_per_inducing_point, patch_length), dtype=NHWC_X.dtype)
    for i in range(M * samples_per_inducing_point // patches_per_image):
        # Sample a random image, compute the patches and sample some random patches.
        image = _sample(NHWC_X, 1)[0]
        sampled_patches = _sample_patches(image, patches_per_image,
                patch_size, patch_length)
        assert sampled_patches[0].shape == (patch_size, patch_size), f'patches of size {sampled_patches[0].shape} are different size in cluster patches {patch_size}'
        patches[i*patches_per_image:(i+1)*patches_per_image] = sampled_patches

    assert patch_size * patch_size > M, 'number of pixels per patch is less than the number of inducing points'

    k_means = cluster.KMeans(n_clusters=M, n_jobs=-1)
    k_means.fit(patches)
    return k_means.cluster_centers_


def compute_inertia(a, X):
    W = [np.mean(pairwise_distances(X[a == c, :])) for c in np.unique(a)]
    return np.mean(W)


def compute_gap(clustering, data, k_min =200 ,k_max=380, k_incerement = 100 ,n_references=5):
    if len(data.shape) == 1:
        data = data.reshape(-1, 1)
    reference = np.random.rand(*data.shape)
    reference_inertia = []
    for k in range(k_min, k_max + 1, k_incerement):
        local_inertia = []
        for _ in range(n_references):
            clustering.n_clusters = k
            assignments = clustering.fit_predict(reference)
            local_inertia.append(compute_inertia(assignments, reference))
        reference_inertia.append(np.mean(local_inertia))

    ondata_inertia = []
    for k in range(k_min, k_max + 1, k_incerement):
        clustering.n_clusters = k
        assignments = clustering.fit_predict(data)
        ondata_inertia.append(compute_inertia(assignments, data))

    gap = np.log(reference_inertia) - np.log(ondata_inertia)
    return gap, np.log(reference_inertia), np.log(ondata_inertia)

def test_cluster(data, k_min=200, k_max=380, k_incerement = 100, n_references=5 ):
    gap, reference_inertia, ondata_inertia = compute_gap(KMeans(), data, k_min=k_min,k_max=k_max,
                                                         k_incerement = k_incerement, n_references=n_references)

    plt.plot(range(1, k_max + 1), reference_inertia,
           '-o', label='reference')
    plt.plot(range(1, k_max + 1), ondata_inertia,
             '-o', label='data')
    plt.xlabel('k')
    plt.ylabel('log(inertia)')
    plt.show()
    plt.savefig('gap_clustering.jpg')

    # Define the OptimalK instance, but pass in our own clustering function
    optimalk = OptimalK(clusterer=special_clustering_func)
    # Use the callable instance as normal.
    n_clusters = optimalk(X, n_refs=3, cluster_array=range(k_min, k_max, k_incerement))



def special_clustering_func(X, k):
    """
    Special clustering function which uses the MeanShift
    model from sklearn.

    These user defined functions *must* take the X and a k
    and can take an arbitrary number of other kwargs, which can
    be pass with `clusterer_kwargs` when initializing OptimalK
    """

    # Here you can do whatever clustering algorithm you heart desires,
    # but we'll do a simple wrap of the MeanShift model in sklearn.

    m = MeanShift()
    m.fit(X)

    # Return the location of each cluster center,
    # and the labels for each point.
    return m.cluster_centers_, m.predict(X)


class TqdmExtraFormat(tqdm):
    """Provides a `total_time` format parameter"""
    @property
    def format_dict(self):
        d = super(TqdmExtraFormat, self).format_dict
        total_time = d["elapsed"] * (d["total"] or 0) / max(d["n"], 1)
        d.update(total_time=self.format_interval(total_time) + " in total")
        return d
