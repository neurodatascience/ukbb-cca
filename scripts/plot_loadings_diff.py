#!/usr/bin/env python
import sys
from pathlib import Path

import click
import matplotlib.pyplot as plt
import pandas as pd

from src.base import NestedItems
from src.cca import CcaResultsSampleSize
from src.database_helpers import DatabaseHelper
from src.plotting import save_fig, plot_loadings_bar
from src.utils import print_params

DNAME_SUMMARY = 'summary'
SEP_TAGS = '-'

AX_WIDTH = 12
AX_HEIGHT_UNIT = 0.5

SET_NAME = 'learn'

@click.command()
@click.argument('n_PCs_all', nargs=-1, required=True)
@click.option('--dpath-cca', required=True, envvar='DPATH_CCA_SAMPLE_SIZE')
@click.option('--cca-suffix1')
@click.option('--cca-suffix2')
@click.option('--tag1')
@click.option('--tag2')
@click.option('--dpath-out', required=True, envvar='DPATH_CCA_COMPARISONS')
@click.option('--subset', default='all')
@click.option('--CA', 'i_component', default=1)
@click.option('--n', 'n_loadings', default=10)
@click.option('--dpath-schema', required=True, envvar='DPATH_SCHEMA')
@click.option('--fpath-udis', required=True, envvar='FPATH_UDIS')
def plot_loadings_diff(n_pcs_all, dpath_cca, cca_suffix1, cca_suffix2, tag1, tag2, 
                       dpath_out, subset, i_component, n_loadings, dpath_schema, fpath_udis):

    def get_tag(tag, cca_suffix):
        if tag is not None:
            return tag
        else:
            return cca_suffix

    print_params(locals())
    i_component = i_component - 1 # zero-indexing
    db_helper = DatabaseHelper(dpath_schema, fpath_udis)

    tag1 = get_tag(tag1, cca_suffix1)
    tag2 = get_tag(tag2, cca_suffix2)

    n_PCs_str = CcaResultsSampleSize.get_dname_PCs(n_pcs_all)
    dpath_figs = Path(dpath_out, n_PCs_str, subset, f'{tag1}{SEP_TAGS}{tag2}')
    dpath_cca1 = Path(f'{dpath_cca}-{cca_suffix1}') if cca_suffix1 is not None else Path(dpath_cca)
    dpath_cca2 = Path(f'{dpath_cca}-{cca_suffix2}') if cca_suffix2 is not None else Path(dpath_cca)
    fpath_summary1 = Path(dpath_cca1, n_PCs_str, DNAME_SUMMARY, subset)
    fpath_summary2 = Path(dpath_cca2, n_PCs_str, DNAME_SUMMARY, subset)
    summary1 = NestedItems.load_fpath(fpath_summary1)
    print(f'Loaded results summary 1: {summary1}')
    summary2 = NestedItems.load_fpath(fpath_summary2)
    print(f'Loaded results summary 2: {summary2}')

    # assume summary1 and summary2 have same dataset names, sample sizes and cca_types
    dataset_names = summary1.dataset_names

    # one figure per sample size
    for sample_size in summary1.levels['sample_size']:

        print(f'sample_size: {sample_size}')

        cca_types = summary1.levels['cca_type']

        n_rows = len(cca_types)
        n_cols = len(dataset_names)
        fig, axes = plt.subplots(
            nrows=n_rows, 
            ncols=n_cols,
            figsize=(n_cols*AX_WIDTH, n_rows*n_loadings*AX_HEIGHT_UNIT),
            squeeze=False,
        )

        for i_row, cca_type in enumerate(cca_types):

            print(f'cca_type: {cca_type}')

            loadings1 = summary1[sample_size, cca_type, SET_NAME, 'mean'].loadings
            loadings2 = summary2[sample_size, cca_type, SET_NAME, 'mean'].loadings

            loadings_diff = [
                (l1 - l2).iloc[:, i_component].dropna().sort_values()
                for l1, l2 in zip(loadings1, loadings2)
            ]
            axes_row = axes[i_row]
            
            plot_loadings_bar(
                loadings=loadings_diff,
                labels=[db_helper.udis_to_text(l.index) for l in loadings_diff],
                ax_titles=[
                    f'{tag1} - {tag2} loading differences ({cca_type}, {dataset_name})' 
                    for dataset_name in dataset_names
                ],
                axes=axes_row,
            )

        fpath_fig = dpath_figs / f'{subset}_{sample_size}-loadings_diff'
        save_fig(fig, fpath_fig)

if __name__ == '__main__':
    plot_loadings_diff()
