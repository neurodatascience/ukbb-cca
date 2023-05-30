#!/usr/bin/env python
import click
from src.data_processing import XyData, UDI_AGE, PREFIX_AGE_GROUP
from src.utils import print_params

DATASET_NAMES = ['behavioural', 'brain']
EXTRA_DATASET_NAMES = ['disease']
CONF_NAME = 'demographic'

@click.command()
@click.option('--dpath-data', required=True, envvar='DPATH_PROCESSED')
@click.option('--dataset-name', 'dataset_names', multiple=True)
@click.option('--extra-dataset-name', 'extra_dataset_names', multiple=True)
@click.option('--conf-name', default=CONF_NAME)
@click.option('--udi-holdout', default=UDI_AGE)
@click.option('--group-name', default=PREFIX_AGE_GROUP)
@click.option('--verbose/--quiet', default=True)
def build_dataset(dpath_data, dataset_names, extra_dataset_names, conf_name, 
                  udi_holdout, group_name, verbose):

    if len(dataset_names) == 0:
        dataset_names = DATASET_NAMES
    if len(extra_dataset_names) == 0:
        extra_dataset_names = EXTRA_DATASET_NAMES

    print_params(locals())

    Xy = XyData(
        dpath_data, 
        dataset_names, 
        extra_dataset_names=extra_dataset_names,
        conf_name=conf_name,
        udi_holdout=udi_holdout,
        group_name=group_name,
        verbose=verbose,
    )

    print('----------------------')
    print(Xy)

    Xy.save()


if __name__ == '__main__':

    build_dataset()
