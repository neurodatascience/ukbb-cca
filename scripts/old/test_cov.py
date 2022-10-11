
print('first line')

import itertools
from datetime import datetime
import numpy as np

print('import statements done')

m = 40500 # number of rows
n = 675 # number of columns
na_fraction = 0.1
seed = 3791

# adapted from https://stackoverflow.com/questions/43238173/python-convert-matrix-to-positive-semi-definite/43244194#43244194
def nearest_spd(A):

    def is_symmetric(A, rtol=1e-05, atol=1e-08):
        return np.allclose(A, A.T, rtol=rtol, atol=atol)

    def is_pd(A):
        try:
            np.linalg.cholesky(A)
            return True
        except np.linalg.LinAlgError:
            return False

    # input validation
    if len(A.shape) != 2:
        raise ValueError('matrix must be 2D')

    n_rows, n_cols = A.shape

    if n_rows != n_cols:
        raise ValueError('matrix must be square')

    if is_symmetric(A) and is_pd(A):
        return A

    if n_rows == 1:
        return np.spacing(1)

    # symmetrize A into B
    B = (A + A.T) / 2

    # H is the symmetric polar factor of B
    _, s, Vt = np.linalg.svd(B)
    H = Vt.T @ np.diag(s) @ Vt

    A_hat = (B + H) / 2
    A_hat = (A_hat + A_hat.T) / 2 # make it symmetric

    k = 1
    I = np.eye(n_rows)
    spacing = np.spacing(np.linalg.norm(A_hat))
    while not is_pd(A_hat):
        min_eig = np.min(np.real(np.linalg.eigvals(A_hat)))
        A_hat += I * (-min_eig * k**2 + spacing)
        k += 1

    return A_hat

print('function definition done')

size = m * n
rng = np.random.default_rng(seed=seed)

na_coordinates = rng.choice(
    list(itertools.product(np.arange(m), np.arange(n))), 
    size=int(na_fraction * size), replace=False,
)
print('na_coordinates: ', len(na_coordinates))

big_X = rng.random((m, n))
print('big_X: ', big_X.shape)
big_X[na_coordinates[:, 0], na_coordinates[:, 1]] = np.nan
print('big_X NaNs: ', np.isnan(big_X).sum())

time_start = datetime.now()

big_cov = np.cov(big_X, rowvar=True)
print('big_cov: ', big_cov.shape)

time_cov1 = datetime.now()
print(time_cov1 - time_start)

small_cov1 = np.cov(big_X, rowvar=False)
print('small_cov1: ', small_cov1.shape)

time_cov2 = datetime.now()
print(time_cov2 - time_cov1)

big_X_demeaned = big_X - big_X.mean(axis=0)
small_cov2 = big_X_demeaned.T @ big_X_demeaned
print('small_cov2: ', small_cov2.shape)

time_cov3 = datetime.now()
print(time_cov3 - time_cov2)

big_cov_spd = nearest_spd(big_cov)
print('big_cov_spd: ', big_cov_spd.shape)

time_cov4 = datetime.now()
print(time_cov4 - time_cov3)
