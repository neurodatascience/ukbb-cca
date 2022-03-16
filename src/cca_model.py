
import numpy as np
from scipy.linalg import eigh
from cca_zoo.models import CCA
from sklearn.utils.extmath import svd_flip
from .utils import eig_flip

class FlippedCCA(CCA):
    def _setup_evp(self, views, **kwargs):
        Us, Ss, Vts = _flipped_pca_data(*views)
        self.Bs = [
            (1 - self.c[i]) * S * S / self.n + self.c[i] for i, S in enumerate(Ss)
        ]
        if len(views) == 2:
            self._two_view = True
            C, D = self._two_view_evp(Us, Ss)
        else:
            self._two_view = False
            C, D = self._multi_view_evp(Us, Ss)
        return Vts, C, D

    def _solve_evp(self, views, C, D=None, **kwargs):
        p = C.shape[0]
        if self._two_view:
            [eigvals, eigvecs] = eigh(C, subset_by_index=[p - self.latent_dims, p - 1])
            eigvecs = eig_flip(eigvecs)
            idx = np.argsort(eigvals, axis=0)[::-1][: self.latent_dims]
            eigvecs = eigvecs[:, idx].real
            w_y = views[1].T @ np.diag(1 / np.sqrt(self.Bs[1])) @ eigvecs
            w_x = (
                views[0].T
                @ np.diag(1 / self.Bs[0])
                @ self.R_12
                @ np.diag(1 / np.sqrt(self.Bs[1]))
                @ eigvecs
                / np.sqrt(eigvals[idx])
            )
            self.weights = [w_x, w_y]
        else:
            raise NotImplementedError('FlippedCCA only works for 2-view problems')
    
def _flipped_pca_data(*views):
    """
    Same as cca-zoo implementation but uses deterministic signs for SVD.
    """
    views_U = []
    views_S = []
    views_Vt = []
    for i, view in enumerate(views):
        U, S, Vt = np.linalg.svd(view, full_matrices=False)
        U, Vt = svd_flip(U, Vt)
        views_U.append(U)
        views_S.append(S)
        views_Vt.append(Vt)
    return views_U, views_S, views_Vt
