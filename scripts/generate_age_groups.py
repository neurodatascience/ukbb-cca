#!/usr/bin/env python
import click
from src.data_processing import get_age_groups_from_holdouts, UDI_AGE
from src.utils import print_params

YEAR_STEP = 5
ALTERNATIVE_DATASET_NAME = 'demographic'

@click.command()
@click.option('--dpath-data', required=True, envvar='DPATH_PROCESSED')
@click.option('--alt-dataset', default=ALTERNATIVE_DATASET_NAME)
@click.option('--year-step', default=YEAR_STEP)
@click.option('--plot/--no-plot', default=True)
@click.option('--dpath-figs', default='.', envvar='DPATH_PREPROCESSING')
def generate_age_groups(dpath_data, alt_dataset, year_step, plot, dpath_figs):
    print_params(locals())
    get_age_groups_from_holdouts(
        dpath_data, 
        year_step=year_step,
        plot=plot,
        dpath_figs=dpath_figs,
        alternative_dataset=alt_dataset,
    )

if __name__ == '__main__':
    generate_age_groups()
