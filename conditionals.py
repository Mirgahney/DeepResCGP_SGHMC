# Credit to GPflow

import tensorflow as tf


def base_conditional(Kmn, Lm, Knn, f, *, full_cov=False, q_sqrt=None, white=False):
    """
    Given a g1 and g2, and distribution p and q such that
      p(g2) = N(g2;0,Kmm)
      p(g1) = N(g1;0,Knn)
      p(g1|g2) = N(g1;0,Knm)
    And
      q(g2) = N(g2;f,q_sqrt*q_sqrt^T)
    This method computes the mean and (co)variance of
      q(g1) = \int q(g2) p(g1|g2)
    :param Kmn: M x N
    :param Kmm: M x M
    :param Knn: N x N  or  N
    :param f: M x R
    :param full_cov: bool
    :param q_sqrt: None or R x M x M (lower triangular)
    :param white: bool
    :return: N x R  or R x N x N
    """
    # compute kernel stuff
    num_func = tf.shape(f)[1]  # R

    # Compute the projection matrix A
    A = tf.matrix_triangular_solve(Lm, Kmn, lower=True)

    # compute the covariance due to the conditioning
    if full_cov:
        fvar = Knn - tf.matmul(A, A, transpose_a=True)
        fvar = tf.tile(fvar[None, :, :], [num_func, 1, 1])  # R x N x N
    else:
        fvar = Knn - tf.reduce_sum(tf.square(A), 0)
        fvar = tf.tile(fvar[None, :], [num_func, 1])  # R x N

    # another backsubstitution in the unwhitened case
    if not white:
        A = tf.matrix_triangular_solve(tf.transpose(Lm), A, lower=False)

    # construct the conditional mean
    fmean = tf.matmul(A, f, transpose_a=True)

    if q_sqrt is not None:
        if q_sqrt.get_shape().ndims == 2:
            LTA = A * tf.expand_dims(tf.transpose(q_sqrt), 2)  # R x M x N
        elif q_sqrt.get_shape().ndims == 3:
            L = q_sqrt
            A_tiled = tf.tile(tf.expand_dims(A, 0), tf.stack([num_func, 1, 1]))
            LTA = tf.matmul(L, A_tiled, transpose_a=True)  # R x M x N
        else:  # pragma: no cover
            raise ValueError("Bad dimension for q_sqrt: %s" %
                             str(q_sqrt.get_shape().ndims))
        if full_cov:
            fvar = fvar + tf.matmul(LTA, LTA, transpose_a=True)  # R x N x N
        else:
            fvar = fvar + tf.reduce_sum(tf.square(LTA), 1)  # R x N

    if not full_cov:
        fvar = tf.transpose(fvar)  # N x R

    return fmean, fvar # N x R, R x N x N or N x R

def multiple_output_conditional(Kmn, Lm, Knn, u, full_cov=False, white=False):
    num_func = tf.shape(u)[1]  # R

    def solve_A(MN_Kmn): # TODO: check for numerical instability in solve
        A = tf.matrix_triangular_solve(Lm, MN_Kmn, lower=True)
        #assert tf.math.reduce_any(tf.math.less((tf.matmul(Lm, A) - MN_Kmn), 1e-3)), 'solve_A isn\'t within threshold of 1e-3 in multiple_output_conditional'
        return A # M x M @ M x N -> M x N
    A = tf.map_fn(solve_A, Kmn, parallel_iterations=100) # P x M x N

    # compute the covariance due to the conditioning
    if full_cov:
        fvar = Knn - tf.tensordot(A, A, [[1], [1]]) # P x N x N
        fvar = tf.tile(fvar[None, :, :, :], [num_func, 1, 1, 1])  # R x N x N
    else:
        fvar = Knn - tf.reduce_sum(tf.square(A), 1) # P x N
        fvar = tf.tile(fvar[None, :, :], [num_func, 1, 1])  # R x P x N

    # another backsubstitution in the unwhitened case
    if not white:
        def backsub(MN_A):
            return tf.matrix_triangular_solve(tf.transpose(Lm), MN_A, lower=False)
        A = tf.map_fn(backsub, A, parallel_iterations=100) # P x M x N

    # construct the conditional mean
    fmean = tf.tensordot(A, u, [[1], [0]]) # P x N x R #Residual connection makes sense here but the U isn't trainable
    fmean = tf.transpose(fmean, [1, 0, 2]) # N x P x R

    return fmean, fvar # N x P x R, R x P x N or R x P x N x N

