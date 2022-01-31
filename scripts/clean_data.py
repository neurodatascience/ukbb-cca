
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from paths import DPATHS, FPATHS
from src.plotting import plot_na_histograms

if __name__ == '__main__':

    # process all 3 datasets at the same time
    # so that they all keep/remove the same subjects
    domains = ['behavioural', 'brain', 'demographic']

    remove_constant_cols = True

    fig_prefix = 'hist_na'
    dpath_figs = DPATHS['preprocessing']

    print('----- Parameters -----')
    print(f'domains:\t{domains}')
    print(f'remove_constant_cols:\t{remove_constant_cols}')
    print(f'fig_prefix:\t{fig_prefix}')
    print(f'dpath_figs:\t{dpath_figs}')
    print('----------------------')

    dfs_data = {}
    subjects_to_drop = set()

    print('----- Loading unprocessed data -----')
    for domain in domains:

        fpath_data = FPATHS[f'data_{domain}']
        print(f'{domain}: {fpath_data}')

        # load data
        df_data = pd.read_csv(fpath_data, index_col='eid')
        print(f'\tDataframe shape: {df_data.shape}')

        # plot histograms of row-/column-wise NaN frequencies
        fig, freqs_na_row, freqs_na_col = plot_na_histograms(df_data, return_freqs=True)
        fig.subplots_adjust(top=0.85)
        fig.suptitle(f'{domain.capitalize()} dataset {df_data.shape}')

        # save figure
        fpath_fig = os.path.join(dpath_figs, f'{fig_prefix}_{domain}_before.png')
        fig.savefig(fpath_fig, dpi=300, bbox_inches='tight')
        print(f'\tFigure saved: {fpath_fig}')

        # add rows to drop later
        subjects_to_drop.update(freqs_na_row.loc[np.isclose(freqs_na_row, 1)].index)

        dfs_data[domain] = df_data

    print('----- Cleaning data -----')
    for domain in domains:

        print(f'{domain}')

        df_clean = dfs_data[domain].drop(index=subjects_to_drop)
        df_clean = df_clean.dropna(axis='columns', how='all')
        print(f'\tDataframe shape after removing empty rows/columns: {df_clean.shape}')

        # drop columns that only contain one unique value (excluding NaNs)
        if remove_constant_cols:
            nunique = df_clean.nunique()
            constant_cols = nunique[nunique == 1].index
            df_clean = df_clean.drop(columns=constant_cols)
            if len(constant_cols) > 0:
                print(f'\tDropped {len(constant_cols)} columns with constant values')

        # plot histograms of row-/column-wise NaN frequencies
        fig, freqs_na_row, freqs_na_col = plot_na_histograms(df_clean, return_freqs=True)
        fig.subplots_adjust(top=0.85)
        fig.suptitle(f'{domain.capitalize()} dataset {df_clean.shape}')

        # save figure
        fpath_fig = os.path.join(dpath_figs, f'{fig_prefix}_{domain}_after.png')
        fig.savefig(fpath_fig, dpi=300, bbox_inches='tight')
        print(f'\tFigure saved: {fpath_fig}')

        # report NaN count/percentage
        na_count = df_clean.isna().values.sum()
        print(f'\tNaN count: {na_count} ({100*na_count/df_clean.size:.0f}%)')

        # save
        fpath_out = FPATHS[f'data_{domain}_clean']
        df_clean.to_csv(fpath_out, header=True, index=True)
        print(f'\tSaved to {fpath_out}')
