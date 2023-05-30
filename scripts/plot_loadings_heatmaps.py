#!/usr/bin/env python
import sys
from pathlib import Path

import click
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from src.base import NestedItems
from src.cca import CcaResultsSampleSize
from src.database_helpers import DatabaseHelper
from src.plotting import save_fig
from src.utils import print_params

DNAME_FIGS = 'figs'
DNAME_SUMMARY = 'summary'

AX_WIDTH_UNIT = 1.5
AX_HEIGHT = 20

SET_NAME = 'learn'

@click.command()
@click.argument('n_PCs_all', nargs=-1, required=True)
@click.option('--dpath-cca', required=True, envvar='DPATH_CCA_SAMPLE_SIZE')
@click.option('--subset', default='all')
@click.option('--CA', 'i_component', default=1)
@click.option('--dpath-schema', required=True, envvar='DPATH_SCHEMA')
@click.option('--fpath-udis', required=True, envvar='FPATH_UDIS')
def plot_loadings_heatmaps(n_pcs_all, dpath_cca, subset, i_component, dpath_schema, fpath_udis):

    print_params(locals())
    i_component = i_component - 1 # zero-indexing
    db_helper = DatabaseHelper(dpath_schema, fpath_udis)

    n_PCs_str = CcaResultsSampleSize.get_dname_PCs(n_pcs_all)
    dpath_subset = Path(dpath_cca, n_PCs_str, subset)

    if not dpath_subset.exists():
        print(f'[ERROR] Directory not found: {dpath_subset}')
        sys.exit(1)

    dpath_figs = Path(dpath_cca, DNAME_FIGS, n_PCs_str, subset)
    
    fpath_summary = Path(dpath_cca, n_PCs_str, DNAME_SUMMARY, subset)
    summary = NestedItems.load_fpath(fpath_summary)
    print(f'Loaded results summary: {summary}')

    dataset_names = summary.dataset_names
    cca_types = summary.levels['cca_type']
    sample_sizes = sorted([int(s) for s in summary.levels['sample_size']])
    # udis_datasets = summary.udis_datasets # TODO not good
    # labels = [db_helper.udis_to_text(udis.get_level_values(-1)) for udis in udis_datasets]

    n_rows = len(cca_types)
    n_cols = len(dataset_names)

    fig_mag, axes_mag = plt.subplots(
        nrows=n_rows, 
        ncols=n_cols,
        figsize=(AX_WIDTH_UNIT*len(sample_sizes), n_rows*AX_HEIGHT),
        squeeze=False,
    )
    
    fig_rank, axes_rank = plt.subplots(
        nrows=n_rows, 
        ncols=n_cols,
        figsize=(AX_WIDTH_UNIT*len(sample_sizes), n_rows*AX_HEIGHT),
        squeeze=False,
    )

    for i_row, cca_type in enumerate(cca_types):

        for i_col, dataset_name in enumerate(dataset_names):

            ax_mag = axes_mag[i_row][i_col]
            ax_rank = axes_rank[i_row][i_col]

            data_for_df_mag = []
            data_for_df_rank = []
            for sample_size in reversed(sample_sizes):

                loadings = summary[sample_size, cca_type, SET_NAME, 'mean'].loadings[i_col].iloc[:, i_component]
                # loadings.index = db_helper.udis_to_text(loadings.index)
                # labels = db_helper.udis_to_text(loadings.index) 

                # data_mag = pd.Series(
                #     data=loadings.iloc[:, i_component],
                #     index=labels,
                #     name=sample_size,
                # )

                data_mag = loadings
                data_mag.name = sample_size

                data_rank = pd.Series(
                    data=np.argsort(np.argsort(loadings)), # increasing
                    # index=loadings.index,
                    name=sample_size,
                )

                data_for_df_mag.append(data_mag)
                data_for_df_rank.append(data_rank)

            df_mag = pd.concat(data_for_df_mag, axis='columns').sort_values(sample_sizes[-1], ascending=False)
            df_mag.index = db_helper.udis_to_text(df_mag.index)
            df_rank = pd.concat(data_for_df_rank, axis='columns').sort_values(sample_sizes[-1], ascending=False)
            df_rank.index = db_helper.udis_to_text(df_rank.index)

            sns.heatmap(df_mag, ax=ax_mag)
            sns.heatmap(df_rank, ax=ax_rank)

            ax_mag.set_title(f'{dataset_name.capitalize()} ({cca_type})')
            ax_rank.set_title(f'{dataset_name.capitalize()} ({cca_type})')

    fig_mag.tight_layout()
    fig_rank.tight_layout()

    fpath_fig_rank = dpath_figs / f'{subset}-loadings_heatmap_rank'
    save_fig(fig_rank, fpath_fig_rank)

    fpath_fig_mag = dpath_figs / f'{subset}-loadings_heatmap_mag'
    save_fig(fig_mag, fpath_fig_mag)

if __name__ == '__main__':
    plot_loadings_heatmaps()
