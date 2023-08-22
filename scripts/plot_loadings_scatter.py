#!/usr/bin/env python
import sys
from pathlib import Path

import click
import matplotlib.pyplot as plt

from src.base import NestedItems
from src.cca import CcaResultsSampleSize
from src.database_helpers import DatabaseHelper
from src.plotting import save_fig, plot_loadings_scatter
from src.utils import print_params

DNAME_FIGS = 'figs'
DNAME_SUMMARY = 'summary'

AX_WIDTH = 12
AX_HEIGHT_UNIT = 0.5

SET_NAME = 'learn'
BOOTSTRAP_ALPHA = 0.05

@click.command()
@click.argument('n_PCs_all', nargs=-1, required=True)
@click.option('--dpath-cca', required=True, envvar='DPATH_CCA_SAMPLE_SIZE')
@click.option('--subset', default='all')
@click.option('--CA', 'i_component', default=1)
@click.option('--n', 'n_loadings', default=10)
@click.option('--dpath-schema', required=True, envvar='DPATH_SCHEMA')
@click.option('--fpath-udis', required=True, envvar='FPATH_UDIS')
@click.option('--largest-null/--no-largest-null', 'use_largest_null_model', default=True)
def plot_loadings(n_pcs_all, dpath_cca, subset, i_component, n_loadings, dpath_schema, fpath_udis, use_largest_null_model):

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

    fpath_summary_null = Path(dpath_cca, n_PCs_str, DNAME_SUMMARY, f'{subset}-null_model')
    try:
        summary_null = NestedItems.load_fpath(fpath_summary_null)
        print(f'Loaded null model summary: {fpath_summary_null}')
    except FileNotFoundError:
        summary_null = None
        print(f'Did not find null model summary: {fpath_summary_null}')

    dataset_names = summary.dataset_names
    # udis_datasets = summary.udis_datasets
    # labels = [db_helper.udis_to_text(udis.get_level_values(-1)) for udis in udis_datasets]

    largest_sample_size = sorted(summary.levels['sample_size'])[-1]

    # one figure per sample size
    for sample_size in summary.levels['sample_size']:

        print(f'sample_size: {sample_size}')

        cca_types = summary.levels['cca_type']

        n_rows = len(cca_types)
        n_cols = len(dataset_names)
        fig, axes = plt.subplots(
            nrows=n_rows, 
            ncols=n_cols,
            figsize=(n_cols*AX_WIDTH, n_rows*n_loadings*AX_HEIGHT_UNIT),
            squeeze=False,
            )
        
        if use_largest_null_model:
            sample_size_null_model = largest_sample_size
        else:
            sample_size_null_model = sample_size

        for i_row, cca_type in enumerate(cca_types):

            print(f'cca_type: {cca_type}')

            loadings = summary[sample_size, cca_type, SET_NAME, 'mean'].loadings
            labels = [db_helper.udis_to_text(loading.index) for loading in loadings]
            loadings_errs = summary[sample_size, cca_type, SET_NAME, 'std'].loadings
            axes_row = axes[i_row]

            try:
                loadings_null_low = summary_null[sample_size_null_model, cca_type, SET_NAME, f'quantile_{BOOTSTRAP_ALPHA/2}'].loadings
                loadings_null_high = summary_null[sample_size_null_model, cca_type, SET_NAME, f'quantile_{1-BOOTSTRAP_ALPHA/2}'].loadings

                loadings_null_low = [loading.iloc[:, i_component] for loading in loadings_null_low]
                loadings_null_high = [loading.iloc[:, i_component] for loading in loadings_null_high]
            except Exception:
                loadings_null_low = None
                loadings_null_high = None
            
            plot_loadings_scatter(
                loadings=[loading.iloc[:, i_component] for loading in loadings],
                loadings_null_low=loadings_null_low,
                loadings_null_high=loadings_null_high,
                labels=labels,
                ax_titles=[
                    f'{dataset_name.capitalize()} loadings ({cca_type}) ($\mathregular{{CA_{{{i_component+1}}}}}$)' 
                    for dataset_name in dataset_names
                ],
                errs=[err.iloc[:, i_component] for err in loadings_errs],
                axes=axes_row,
                n_loadings=n_loadings,
            )

        fpath_fig = dpath_figs / f'{subset}_{sample_size}-loadings'
        save_fig(fig, fpath_fig)

if __name__ == '__main__':
    plot_loadings()
