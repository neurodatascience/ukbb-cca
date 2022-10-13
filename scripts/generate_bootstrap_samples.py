#!/usr/bin/env python
import click
import numpy as np
from src.sampling import BootstrapSamples
from src.utils import print_params

VAL_SAMPLE_FRACTION = 0.5
N_FOLDS = 5
SEED = 3791
MAX_N_PCS = 100

@click.command()
@click.argument('n-bootstrap-repetitions', type=int)
@click.argument('n-sample-sizes', type=int)
@click.option('--dpath-data', required=True, envvar='DPATH_PROCESSED')
@click.option('--val-sample-fraction', default=VAL_SAMPLE_FRACTION)
@click.option('--n-folds', default=N_FOLDS)
@click.option('--seed', default=SEED)
@click.option('--max-n-pcs', 'max_n_PCs', default=MAX_N_PCS)
@click.option('--verbose/--quiet', default=True)
def generate_bootstrap_samples(
    dpath_data,
    n_bootstrap_repetitions,
    n_sample_sizes,
    val_sample_fraction,
    n_folds,
    seed,
    max_n_PCs,
    verbose,
):
    print_params(locals())
    samples = BootstrapSamples(
        dpath=dpath_data,
        n_bootstrap_repetitions=n_bootstrap_repetitions,
        n_sample_sizes=n_sample_sizes,
        val_sample_fraction=val_sample_fraction,
        n_folds=n_folds,
        seed=seed,
        max_n_PCs=max_n_PCs,
        verbose=verbose,
    )
    samples.generate()
    print(samples)
    samples.save()

if __name__ == '__main__':
    np.set_printoptions(precision=4, suppress=True, linewidth=100, sign=' ')
    generate_bootstrap_samples()
