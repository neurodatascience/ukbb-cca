#!/usr/bin/env python
from pathlib import Path
import click
from src.database_helpers import DatabaseHelper
from src.data_processing import write_subset, generate_fname_data
from src.utils import print_params, save_json, add_suffix

PREFIX_FNAME_CONFIG = 'select'

CONFIGS = {
    'behavioural': {
        'categories': [100026], # [100026, 116]
        'fields': [],
        'title_substring': None,
        'title_substrings_reject': [
            # reaction time
            'Index for card',
            # numeric memory
            'Completion status of numeric memory test',
            'Round of numeric memory test',
            'Number of digits to be memorised/recalled',
            'Target number to be',
            'Number entered by participant',
            'Time number displayed for',
            'Digits entered correctly',
            'Time first key touched',
            'Time elapsed',
            'Keystroke history',
            # fluid intelligence
            'Attempted fluid intelligence (FI) test',
            # trail making
            # matrix pattern completion
            'Item selected for each puzzle',
            # symbol digit substitution
            'Values entered',
            'Values wanted',
            'First code array presented',
            # tower rearranging
            'Value entered',
            # prospective memory
            'Test completion status',
            'PM: initial answer',
            'PM: final answer',
            'Final attempt correct',
            'History of attempts',
            'Duration screen displayed',
            'Time when initial screen shown',
            'Time screen exited',
            'Time to answer',
            # pairs matching
            'Number of columns displayed in round',
            'Number of rows displayed in round',
            'Screen layout',
        ],
        'instances': [2],
        'keep_instance': 'all',
    },
    'brain': {
        'categories': [135], # [135, 134]
        'fields': [],
        # 'title_substring': 'Mean FA', # dMRI measure
        'title_substring': None,
        'title_substrings_reject': ['L1', 'L2', 'L3'],#['L1', 'L2', 'L3'],
        'instances': [2],
        'keep_instance': 'all',
    },
    'demographic': {
        'categories': [1001, 100011], #[1001, 1002, 1006] #100011 blood pressure, 100007 arterial stiffness
        'fields': [4194, 50, 21002, 21, 1707], # 4194 pulse rate, 50 standing height, 21002 weight, 21 weight method, 1707 handedness
        'title_substring': None,
        'title_substrings_reject': [
            'Place of birth in UK',
            'Time since interview start at which blood pressure screen(s) shown',
        ],
        'instances': [0, 2],
        'keep_instance': 'max',
    },
    'disease': {
        'categories': [2002],
        'fields': [],
        'title_substring': 'ICD10',
        'title_substrings_reject': [],
        'instances': [0, 1, 2],
        'keep_instance': 'all',
    },
}

@click.command()
@click.argument('domain')
@click.option('--fpath-data', required=True, envvar='FPATH_TABULAR_MRI_FILTERED')
@click.option('--dpath-processed', required=True, default='.', envvar='DPATH_PROCESSED')
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
    fields = config['fields']
    title_substring = config['title_substring']
    title_substrings_reject = config['title_substrings_reject']
    instances = config['instances']
    keep_instance = config['keep_instance']
    fpath_out = Path(dpath_processed, generate_fname_data(domain))

    print_params(locals(), skip='config')

    db_helper = DatabaseHelper(dpath_schema, fpath_udis)

    udis = db_helper.get_udis_from_categories_and_fields(categories, fields=fields, value_types=value_types, 
        title_substring=title_substring, title_substrings_reject=title_substrings_reject,
        instances=instances, keep_instance=keep_instance)

    print(f'Selected {len(udis)} UDIs')

    n_rows, n_cols = write_subset(fpath_data, fpath_out, colnames=udis, chunksize=chunksize)
    print(f'Wrote {n_rows} rows and {n_cols} columns to {fpath_out}')

    # save config
    save_json(config, Path(dpath_processed, add_suffix(PREFIX_FNAME_CONFIG, domain)))

if __name__ == '__main__':
    select_categories()
