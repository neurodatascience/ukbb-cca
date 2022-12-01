#!/usr/bin/env python
import click
from pathlib import Path
from src.database_helpers import DatabaseHelper
from src.data_processing import clean_datasets
from src.utils import print_params, save_json

# default settings
DOMAIN_DEMOGRAPHIC = 'demographic'
DOMAINS = ['behavioural', 'brain', DOMAIN_DEMOGRAPHIC]
DOMAINS_TO_SQUARE = [DOMAIN_DEMOGRAPHIC]
DOMAINS_EXTRA = ['disease']
HOLDOUTS = [] #[21003, 34] # TODO do we still need to hold out these?
SQUARE_CONF = True
THRESHOLD_NA = 0.5
THRESHOLD_HIGH_FREQ = 0.95
THRESHOLD_OUTLIERS = 100
THRESHOLD_LARGE = 100000

FNAME_PARAMS = 'clean_data'

@click.command()
@click.option('--dpath-data', required=True, envvar='DPATH_PROCESSED')
@click.option('--dpath-figs', required=True, default='.', envvar='DPATH_PREPROCESSING')
@click.option('--domain', 'domains', multiple=True)
@click.option('--domain-to-square', 'domains_to_square', multiple=True)
@click.option('--domain-extra', 'domains_extra', multiple=True) # subset these based on clean data
@click.option('--holdout', 'holdouts', multiple=True, default=HOLDOUTS) # age, year of birth
@click.option('--square-conf', default=SQUARE_CONF)
@click.option('--threshold-na', default=THRESHOLD_NA)
@click.option('--threshold-high-freq', default=THRESHOLD_HIGH_FREQ)
@click.option('--threshold-outliers', default=THRESHOLD_OUTLIERS)
@click.option('--threshold-large', default=THRESHOLD_LARGE)
@click.option('--dpath-schema', required=True, envvar='DPATH_SCHEMA')
@click.option('--fpath-udis', required=True, envvar='FPATH_UDIS')
def clean_data(dpath_data, dpath_figs, domains, domains_to_square, domains_extra, holdouts, 
    square_conf, threshold_na, threshold_high_freq, threshold_outliers, threshold_large,
    dpath_schema, fpath_udis
):

    if len(domains) == 0:
        domains = DOMAINS
    if len(domains_to_square) == 0:
        domains_to_square = DOMAINS_TO_SQUARE
    if len(domains_extra) == 0:
        domains_extra = DOMAINS_EXTRA

    params = locals()
    print_params(params)
    save_json(params, Path(dpath_data) / FNAME_PARAMS)

    db_helper = DatabaseHelper(dpath_schema, fpath_udis)

    clean_datasets(
        db_helper=db_helper,
        dpath_data=dpath_data,
        dpath_figs=dpath_figs,
        domains=domains,
        domains_extra=domains_extra,
        holdout_fields=holdouts,
        square_conf=square_conf,
        domains_to_square=domains_to_square,
        threshold_na=threshold_na,
        threshold_high_freq=threshold_high_freq,
        threshold_outliers=threshold_outliers,
        threshold_large=threshold_large,
    )

if __name__ == '__main__':
    clean_data()
