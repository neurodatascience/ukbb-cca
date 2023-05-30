#!/usr/bin/env python
import click
from src.database_helpers import DatabaseHelper
from src.data_processing import write_subset
from src.utils import print_params

@click.command()
@click.option('--fpath-raw', required=True, envvar='FPATH_TABULAR_RAW')
@click.option('--fpath-out', required=True, default='.', envvar='FPATH_TABULAR_MRI')
@click.option('--fpath-udis', required=True, envvar='FPATH_UDIS')
@click.option('--dpath-schema', required=True, envvar='DPATH_SCHEMA')
@click.option('--add-category', 'categories', multiple=True, default=[1014])
@click.option('--add-instance', 'instances', multiple=True, default=[2])
@click.option('--chunksize', default=10000)
def select_mri_subjects(fpath_raw, fpath_out, fpath_udis, dpath_schema, 
    categories, instances, chunksize, 
):
    print_params(locals())

    db_helper = DatabaseHelper(dpath_schema, fpath_udis)
    udis_brain = db_helper.get_udis_from_categories_and_fields(categories, instances=instances)

    n_rows, n_cols = write_subset(
        fpath_raw, fpath_out, colnames=None, chunksize=chunksize,
        fn_to_apply=(lambda df: df.dropna(axis='index', how='all', subset=udis_brain))
    )

    print(f'Wrote {n_rows} rows and {n_cols} columns to {fpath_out}')

if __name__ == '__main__':
    select_mri_subjects()
