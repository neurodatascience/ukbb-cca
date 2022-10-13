#!/usr/bin/env python
import click
import pandas as pd
from src.data_processing import write_subset
from src.utils import print_params

@click.command()
@click.option('--fpath-data', required=True, envvar='FPATH_TABULAR_MRI')
@click.option('--fpath-out', required=True, default='.', envvar='FPATH_TABULAR_MRI_FILTERED')
@click.option('--fpath-subjects', required=True, envvar='FPATH_SUBJECTS_WITHDRAWN')
@click.option('--chunksize', default=10000)
def remove_withdrawn_subjects(fpath_data, fpath_out, fpath_subjects, chunksize):
    print_params(locals())

    subjects_to_remove = pd.read_csv(fpath_subjects)['eid']
    print(f'Number of subjects to remove from dataset: {len(subjects_to_remove)}')

    n_rows, n_cols = write_subset(fpath_data, fpath_out, colnames=None, chunksize=chunksize,
        fn_to_apply=(lambda df: df.drop(index=subjects_to_remove, errors='ignore'))
    )
    print(f'Wrote {n_rows} rows and {n_cols} columns to {fpath_out}')

if __name__ == '__main__':
    remove_withdrawn_subjects()
