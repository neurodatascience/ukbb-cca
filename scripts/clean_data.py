#!/usr/bin/env python
import click
from src.database_helpers import DatabaseHelper
from src.data_processing import clean_datasets
from src.utils import print_params

DOMAIN_DEMOGRAPHIC = 'demographic'
DOMAINS = ['behavioural', 'brain', DOMAIN_DEMOGRAPHIC]

@click.command()
@click.option('--dpath-data', required=True, envvar='DPATH_PROCESSED')
@click.option('--dpath-figs', required=True, default='.', envvar='DPATH_PREPROCESSING')
@click.option('--domain', 'domains', multiple=True)
@click.option('--domain-to-square', 'domains_to_square', multiple=True)
@click.option('--holdout', 'holdouts', multiple=True, default=[21003, 34]) # age, year of birth
@click.option('--square-conf', default=True)
@click.option('--threshold-na', default=0.5)
@click.option('--threshold-high-freq', default=0.95)
@click.option('--threshold-outliers', default=100)
@click.option('--dpath-schema', required=True, envvar='DPATH_SCHEMA')
@click.option('--fpath-udis', required=True, envvar='FPATH_UDIS')
def clean_data(dpath_data, dpath_figs, domains, domains_to_square, holdouts, square_conf, 
    threshold_na, threshold_high_freq, threshold_outliers,
    dpath_schema, fpath_udis
):

    db_helper = DatabaseHelper(dpath_schema, fpath_udis)
    if len(domains) == 0:
        domains = DOMAINS
    if len(domains_to_square) == 0:
        domains_to_square = [DOMAIN_DEMOGRAPHIC]

    print_params(locals())

    clean_datasets(
        db_helper=db_helper,
        dpath_data=dpath_data,
        dpath_figs=dpath_figs,
        domains=domains,
        holdout_fields=holdouts,
        square_conf=square_conf,
        domains_to_square=domains_to_square,
        threshold_na=threshold_na,
        threshold_high_freq=threshold_high_freq,
        threshold_outliers=threshold_outliers,
    )

if __name__ == '__main__':
    clean_data()
