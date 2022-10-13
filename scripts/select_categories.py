#!/usr/bin/env python
from pathlib import Path
import click
from src.database_helpers import DatabaseHelper
from src.data_processing import write_subset, generate_fname_data
from src.utils import print_params

CONFIGS = {
    'behavioural': {
        'categories': [100026], # [100026, 116]
        'title_substring': None,
        'title_substrings_reject': [],
        'instances': [2],
        'keep_instance': 'all',
    },
    'brain': {
        'categories': [135],
        # 'title_substring': 'Mean FA', # dMRI measure
        'title_substring': None,
        'title_substrings_reject': ['L1', 'L2', 'L3'],
        'instances': [2],
        'keep_instance': 'all',
    },
    'demographic': {
        'categories': [1001, 1002, 1006],
        'title_substring': None,
        'title_substrings_reject': [],
        'instances': [0, 2],
        'keep_instance': 'max',
    },
}

@click.command()
@click.argument('domain', required=True)
@click.argument('fpath-data', required=True, envvar='FPATH_TABULAR_MRI_FILTERED')
@click.argument('dpath-processed', required=True, envvar='DPATH_PROCESSED')
@click.option('--fpath-udis', required=True, envvar='FPATH_UDIS')
@click.option('--dpath-schema', required=True, envvar='DPATH_SCHEMA')
@click.option('--value-types', default='numeric')
@click.option('--chunksize', default=10000)
def select_categories(domain, fpath_data, dpath_processed, fpath_udis, dpath_schema, value_types, chunksize):

    try:
        config = CONFIGS[domain]
    except KeyError:
        raise ValueError(f'Invalid domain "{domain}". Accepted domains are: {CONFIGS.keys()}')

    categories = config['categories']
    title_substring = config['title_substring']
    title_substrings_reject = config['title_substrings_reject']
    instances = config['instances']
    keep_instance = config['keep_instance']
    fpath_out = Path(dpath_processed, generate_fname_data(domain))

    print_params(locals(), skip='config')

    db_helper = DatabaseHelper(dpath_schema, fpath_udis)

    udis = db_helper.get_udis_from_categories(categories, value_types=value_types, 
        title_substring=title_substring, title_substrings_reject=title_substrings_reject,
        instances=instances, keep_instance=keep_instance)

    print(f'Selected {len(udis)} UDIs')

    n_rows, n_cols = write_subset(fpath_data, fpath_out, colnames=udis, chunksize=chunksize)
    print(f'Wrote {n_rows} rows and {n_cols} columns to {fpath_data}')

if __name__ == '__main__':
    select_categories()
