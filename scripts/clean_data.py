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

# set HOLDOUTS to [] to keep age info
# set everything else as False/None (will be done with sklearn-style models later)
HOLDOUTS = []#[21003, 34] # NOTE: [] for subtract_age, [21003, 34] for add_age
SQUARE_CONF = False#False
THRESHOLD_NA = None#0.5#None
THRESHOLD_HIGH_FREQ = None#0.95#None
THRESHOLD_OUTLIERS = None#100#None
THRESHOLD_LARGE = None#10000#

FNAME_PARAMS = 'clean_data'

FIELDS_TO_AVERAGE = [398, 399, 400, 403, 404, 4255, 6333, 6770, 6771, 6772, 6773] # replace with mean/std
ONE_HOT_ENCODED_UDIS_KEEP = { # only keep these
    '4924-2.0': '4924-2.0_1',   # attempted fluid intelligence test (yes)
    '4935-2.0': '4935-2.0_15',  # Fluid intelligence Q1  (correct)
    '4946-2.0': '4946-2.0_987', # Fluid intelligence Q2  (correct)
    '4957-2.0': '4957-2.0_4',   # Fluid intelligence Q3  (correct)
    '4968-2.0': '4968-2.0_6',   # Fluid intelligence Q4  (correct)
    '4979-2.0': '4979-2.0_4',   # Fluid intelligence Q5  (correct)
    '4990-2.0': '4990-2.0_69',  # Fluid intelligence Q6  (correct)
    '5001-2.0': '5001-2.0_3',   # Fluid intelligence Q7  (correct)
    '5012-2.0': '5012-2.0_26',  # Fluid intelligence Q8  (correct)
    '5556-2.0': '5556-2.0_4',   # Fluid intelligence Q9  (correct)
    '5699-2.0': '5699-2.0_95',  # Fluid intelligence Q10 (correct)
    '5779-2.0': '5779-2.0_5',   # Fluid intelligence Q11 (correct)
    '5790-2.0': '5790-2.0_45',  # Fluid intelligence Q12 (correct)
    '5866-2.0': '5866-2.0_1',   # Fluid intelligence Q13 (correct)
    '20018-2.0': '20018-2.0_1', # prospective memory: prospective memory result (correct on first attempt)
}

@click.command()
@click.option('--dpath-data', required=True, envvar='DPATH_PROCESSED')
@click.option('--dpath-figs', required=True, default='.', envvar='DPATH_PREPROCESSING')
@click.option('--domain', 'domains', multiple=True)
@click.option('--domain-to-square', 'domains_to_square', multiple=True)
@click.option('--domain-extra', 'domains_extra', multiple=True) # subset these based on clean data
@click.option('--holdout', 'holdouts', multiple=True, default=HOLDOUTS) # age, year of birth
@click.option('--square-conf', default=SQUARE_CONF)
@click.option('--fields-to-average', multiple=True)
@click.option('--one-hot-encoded-udis-keep', multiple=True)
@click.option('--threshold-na', default=THRESHOLD_NA)
@click.option('--threshold-high-freq', default=THRESHOLD_HIGH_FREQ)
@click.option('--threshold-outliers', default=THRESHOLD_OUTLIERS)
@click.option('--threshold-large', default=THRESHOLD_LARGE)
@click.option('--dpath-schema', required=True, envvar='DPATH_SCHEMA')
@click.option('--fpath-udis', required=True, envvar='FPATH_UDIS')
def clean_data(dpath_data, dpath_figs, domains, domains_to_square, domains_extra, holdouts, 
    square_conf, fields_to_average, one_hot_encoded_udis_keep,
    threshold_na, threshold_high_freq, threshold_outliers, threshold_large,
    dpath_schema, fpath_udis
):

    if len(domains) == 0:
        domains = DOMAINS
    if len(domains_to_square) == 0:
        domains_to_square = DOMAINS_TO_SQUARE
    if len(domains_extra) == 0:
        domains_extra = DOMAINS_EXTRA
    if len(fields_to_average) == 0:
        fields_to_average = FIELDS_TO_AVERAGE
    if len(one_hot_encoded_udis_keep) == 0:
        one_hot_encoded_udis_keep = ONE_HOT_ENCODED_UDIS_KEEP

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
        fields_to_average=fields_to_average,
        one_hot_encoded_udis_keep=one_hot_encoded_udis_keep,
        threshold_na=threshold_na,
        threshold_high_freq=threshold_high_freq,
        threshold_outliers=threshold_outliers,
        threshold_large=threshold_large,
    )

if __name__ == '__main__':
    clean_data()
