
import os
import numpy as np
import pandas as pd
from paths import DPATHS, FPATHS
from src.database_helpers import DatabaseHelper
from src.data_processing import find_cols_with_missing, find_cols_with_high_freq, find_cols_with_outliers
from src.plotting import plot_na_histograms

# process all 3 datasets at the same time
# so that they all keep/remove the same subjects
domains = ['behavioural', 'brain', 'demographic']

threshold_na = 0.5
threshold_high_freq = 0.95
threshold_outliers = 100

fig_prefix = 'hist_na'
dpath_figs = DPATHS['preprocessing']
dpath_schema = DPATHS['schema']
fpath_udis = FPATHS['udis_tabular_raw']

def one_hot_encode(df):

    def fn_rename(colname):
        components = colname.split('.')
        if len(components) != 2:
            return '.'.join(components[:-1]) # remove trailing floating point
        else:
            return colname

    dfs_encoded = []

    for colname_original in df.columns:

        df_encoded = pd.get_dummies(df[colname_original], prefix=colname_original, prefix_sep='_')
        dfs_encoded.append(df_encoded.rename(columns=fn_rename))

    return pd.concat(dfs_encoded, axis='columns')

def square_df(df):
    df = np.square(df)
    df = df.rename(columns=(lambda x: f'{x}-squared')) # append 'squared' to column name
    return df

def remove_bad_cols(df, threshold_na=0.5, threshold_high_freq=0.95, threshold_outliers=100):

    # identify columns with too much missing data
    cols_with_missing = find_cols_with_missing(df, threshold=threshold_na)
    df = df.drop(columns=cols_with_missing)
    print(f'\t\tRemoved {len(cols_with_missing)} columns with too much missing data')

    # identify columns with not enough variability
    cols_without_variability = find_cols_with_high_freq(df, threshold=threshold_high_freq)
    df = df.drop(columns=cols_without_variability)
    print(f'\t\tRemoved {len(cols_without_variability)} columns with not enough variability')

    # identify columns with outliers
    cols_with_outliers = find_cols_with_outliers(df, threshold=threshold_outliers)
    print(f'\t\tRemoved {len(cols_with_outliers)} columns with outlier(s)')
    df = df.drop(columns=cols_with_outliers)

    return df

if __name__ == '__main__':

    print('----- Parameters -----')
    print(f'domains:\t{domains}')
    print(f'threshold_na:\t{threshold_na}')
    print(f'threshold_high_freq:\t{threshold_high_freq}')
    print(f'threshold_outliers:\t{threshold_outliers}')
    print(f'fig_prefix:\t{fig_prefix}')
    print(f'dpath_figs:\t{dpath_figs}')
    print('----------------------')

    db_helper = DatabaseHelper(dpath_schema, fpath_udis)

    dfs_data = {}
    subjects_to_drop = set()

    print('----- Loading unprocessed data -----')
    for domain in domains:

        fpath_data = FPATHS[f'data_{domain}']
        print(f'{domain}: {fpath_data}')

        # load data
        df_data = pd.read_csv(fpath_data, index_col='eid')
        print(f'\tDataframe shape: {df_data.shape}')

        # one-hot encode categorical variables
        udis = df_data.columns
        categorical_udis = db_helper.filter_udis_by_value_type(udis, 'categorical')
        if len(categorical_udis) > 0:
            print(f'\tOne-hot encoding {len(categorical_udis)} categorical UDIs')
            df_data = pd.concat([df_data, one_hot_encode(df_data[categorical_udis])], axis='columns')
            df_data = df_data.drop(columns=categorical_udis)
            print(f'\t\tShape after one-hot encoding: {df_data.shape}')

        # square confounders
        if domain == 'demographic':
            # only square non-categorical (i.e., integer/continuous) columns
            non_categorical_udis = db_helper.filter_udis_by_value_type(udis, [11, 31])
            print(f'\tSquaring {len(non_categorical_udis)} numerical, non-categorical columns')
            df_data = pd.concat([df_data, square_df(df_data[non_categorical_udis])], axis='columns')
            print(f'\t\tShape after squaring: {df_data.shape}')

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
    dfs_clean = {}
    while len(subjects_to_drop) != 0:
        subjects_to_drop_new = set()
        for domain in domains:

            print(f'{domain}')

            # remove bad rows
            print(f'\tRemoving {len(subjects_to_drop)} rows')
            df_clean = dfs_data[domain].drop(index=subjects_to_drop)

            # remove bad columns
            print('\tLooking for bad columns...')
            df_clean = remove_bad_cols(df_clean, 
                threshold_na=threshold_na, threshold_high_freq=threshold_high_freq, threshold_outliers=threshold_outliers)

            print(f'\tDataframe shape after removing bad rows/columns: {df_clean.shape}')

            # report NaN count/percentage
            na_count = df_clean.isna().values.sum()
            print(f'\tNaN count: {na_count} ({100*na_count/df_clean.size:.0f}%)')

            # plot histograms of row-/column-wise NaN frequencies
            fig, freqs_na_row, freqs_na_col = plot_na_histograms(df_clean, return_freqs=True)
            fig.subplots_adjust(top=0.85)
            fig.suptitle(f'{domain.capitalize()} dataset {df_clean.shape}')

            # add rows to drop later, if any
            subjects_to_drop_new.update(freqs_na_row.loc[np.isclose(freqs_na_row, 1)].index)

            # save figure
            fpath_fig = os.path.join(dpath_figs, f'{fig_prefix}_{domain}_after.png')
            fig.savefig(fpath_fig, dpi=300, bbox_inches='tight')
            print(f'\tFigure saved: {fpath_fig}')

            dfs_clean[domain] = df_clean # to be saved

        subjects_to_drop = subjects_to_drop_new
        print('-------------------------')

    # save
    for domain in domains:
        fpath_out = FPATHS[f'data_{domain}_clean']
        dfs_clean[domain].to_csv(fpath_out, header=True, index=True)
        print(f'\tSaved to {fpath_out}')
