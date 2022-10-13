#!/usr/bin/env python
import click
from src.database_helpers import DatabaseHelper
from src.data_processing import clean_datasets

# process all 3 datasets at the same time
# so that they all keep/remove the same subjects
DOMAIN_DEMOGRAPHIC = 'demographic'
DOMAINS = ['behavioural', 'brain', DOMAIN_DEMOGRAPHIC]

@click.command()
@click.argument('dpath-data', required=True, default='.', envvar='DPATH_PROCESSED')
@click.argument('dpath-figs', required=True, default='.', envvar='DPATH_PREPROCESSING')
@click.option('--holdout', 'holdouts', multiple=True, default=[21003, 34]) # age, year of birth
@click.option('--square-conf', default=True)
@click.option('--threshold-na', default=0.5)
@click.option('--threshold-high-freq', default=0.95)
@click.option('--threshold-outliers', default=100)
@click.option('--dpath-schema', required=True, envvar='DPATH_SCHEMA')
@click.option('--fpath-udis', required=True, envvar='FPATH_UDIS')
def clean_data(dpath_data, dpath_figs, holdouts, square_conf, 
    threshold_na, threshold_high_freq, threshold_outliers,
    dpath_schema, fpath_udis
):
    db_helper = DatabaseHelper(dpath_schema, fpath_udis)
    clean_datasets(
        db_helper=db_helper,
        dpath_data=dpath_data,
        dpath_figs=dpath_figs,
        domains=DOMAINS,
        holdout_fields=holdouts,
        square_conf=square_conf,
        domains_square=[DOMAIN_DEMOGRAPHIC],
        threshold_na=threshold_na,
        threshold_high_freq=threshold_high_freq,
        threshold_outliers=threshold_outliers,
    )

if __name__ == '__main__':

    clean_data()

    # print('----- Parameters -----')
    # print(f'domains:\t{DOMAINS}')
    # print(f'holdout_fields:\t{holdout_fields}')
    # print(f'threshold_na:\t{threshold_na}')
    # print(f'threshold_high_freq:\t{threshold_high_freq}')
    # print(f'threshold_outliers:\t{threshold_outliers}')
    # print(f'fpath_holdout:\t{fpath_holdout}')
    # print(f'fpath_dropped_subjects:\t{fpath_dropped_subjects}')
    # print(f'fpath_dropped_udis:\t{fpath_dropped_udis}')
    # print(f'fig_prefix:\t{FIG_PREFIX}')
    # print(f'dpath_figs:\t{dpath_figs}')
    # print('----------------------')
