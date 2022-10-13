#!/usr/bin/env python
import click
from src.data_processing import get_age_groups_from_holdouts, UDI_AGE
from src.utils import print_params

YEAR_STEP = 5

@click.command()
@click.argument('dpath-data', required=True, envvar='DPATH_PROCESSED')
@click.option('--year-step', default=YEAR_STEP)
@click.option('--plot/--no-plot', default=True)
@click.option('--dpath-figs', default='.', envvar='DPATH_PREPROCESSING')
def generate_age_groups(dpath_data, year_step, plot, dpath_figs):
    print_params(locals())
    get_age_groups_from_holdouts(
        dpath_data, 
        year_step=year_step,
        plot=plot,
        dpath_figs=dpath_figs,
    )

if __name__ == '__main__':
    generate_age_groups()
