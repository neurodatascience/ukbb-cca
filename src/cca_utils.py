
import itertools
import numpy as np

def cca_correlations(transformed_views):
    '''Same as cca_zoo's _cca_base.correlations() except takes in already transformed views.'''

    n_views = len(transformed_views)
    n_latent_dims = transformed_views[0].shape[1]

    all_corrs = []
    for x, y in itertools.product(transformed_views, repeat=2):
        all_corrs.append(
            np.diag(np.corrcoef(x.T, y.T)[:n_latent_dims, n_latent_dims:])
        )

    all_corrs = np.array(all_corrs).reshape(
        (n_views, n_views, n_latent_dims)
    )

    return all_corrs

def cca_score(transformed_views):
    '''Same as cca_zoo's _cca_base.score() except takes in already transformed views.'''

    n_views = len(transformed_views)

    # by default return the average pairwise correlation in each dimension (for 2 views just the correlation)
    pair_corrs = cca_correlations(transformed_views)

    # sum all the pairwise correlations for each dimension. Subtract the self correlations. 
    # Divide by the number of views. Gives average correlation
    dim_corrs = (
        pair_corrs.sum(axis=tuple(range(pair_corrs.ndim - 1))) - n_views
    ) / (n_views ** 2 - n_views)
    return dim_corrs

def get_loadings(views, transformed_views, normalize=False):
    '''
    Same as cca_zoo's _cca_base.get_loadings() except 
    takes in both original views and transformed views.
    Also, handles missing data in original views.
    '''
    view = np.ma.masked_invalid(view)
    if normalize:
        loadings = [
            np.ma.corrcoef(view, transformed_view, rowvar=False)[
                : view.shape[1], view.shape[1] :
            ]
            for view, transformed_view in zip(views, transformed_views)
        ]
    else:
        loadings = [
            np.ma.dot(view.T, transformed_view)
            for view, transformed_view in zip(views, transformed_views)
        ]
    return loadings
