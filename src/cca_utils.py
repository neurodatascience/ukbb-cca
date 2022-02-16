
import numpy as np

def score_projections(projections1, projections2):
    '''Essentially the same as CCA.score() from cca-zoo but uses already transformed views (and for 2 views only).'''

    if projections1.shape != projections2.shape:
        raise ValueError('Projections must have the same shape')

    n_dims = projections1.shape[1]
    corrs = []
    
    for i_dim in range(n_dims):
        corrs.append(np.corrcoef(projections1[:, i_dim], projections2[:, i_dim])[0,1])

    return np.array(corrs)
