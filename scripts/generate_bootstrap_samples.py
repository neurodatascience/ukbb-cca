#!/usr/bin/env python
import click
import numpy as np

from src.sampling import BootstrapSamples
from src.subsets import SUBSET_FN_MAP
from src.utils import print_params

UPPER_BOUND_FRACTION = 0.5
N_FOLDS = 5
SEED = 3791
MAX_N_PCS = 300

@click.command()
@click.argument('n-bootstrap-repetitions', type=int)
@click.argument('n-sample-sizes', type=int)
@click.option('--dpath-data', required=True, envvar='DPATH_PROCESSED')
@click.option('--subset')
@click.option('--stratify/--no-stratify', default=False)
@click.option('--min', 'sample_size_min', type=int)
@click.option('--max', 'sample_size_max', type=int)
@click.option('--upper-bound-fraction', default=UPPER_BOUND_FRACTION)
@click.option('--match-val/--no-match-val', 'match_val_set_size', default=False)
@click.option('--n-folds', default=N_FOLDS)
@click.option('--seed', default=SEED)
@click.option('--max-n-pcs', 'max_n_PCs', default=MAX_N_PCS)
@click.option('--verbose/--quiet', default=True)
def generate_bootstrap_samples(
    n_bootstrap_repetitions,
    n_sample_sizes,
    dpath_data,
    subset,
    stratify,
    sample_size_min, 
    sample_size_max, 
    upper_bound_fraction,
    match_val_set_size,
    n_folds,
    seed,
    max_n_PCs,
    verbose,
):

    print_params(locals())

    if subset is not None:
        try:
            subset_fn = SUBSET_FN_MAP[subset]
        except KeyError:
            raise ValueError(
                f'Invalid subset name: {subset}. '
                f'Valid names are: {SUBSET_FN_MAP.keys()}'
            )
    else:
        subset_fn = None

    samples = BootstrapSamples(
        dpath=dpath_data,
        n_bootstrap_repetitions=n_bootstrap_repetitions,
        n_sample_sizes=n_sample_sizes,
        tag=subset,
        subset_fn=subset_fn,
        stratify=stratify,
        sample_size_min=sample_size_min,
        sample_size_max=sample_size_max,
        upper_bound_fraction=upper_bound_fraction,
        match_val_set_size=match_val_set_size,
        n_folds=n_folds,
        seed=seed,
        max_n_PCs=max_n_PCs,
        verbose=verbose,
    )

    print(samples)
    samples.save()

if __name__ == '__main__':
    np.set_printoptions(precision=4, suppress=True, linewidth=100, sign=' ')
    generate_bootstrap_samples()
