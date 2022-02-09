
import os
import numpy as np
import pandas as pd
from paths import DPATHS, FPATHS
from src.database_helpers import DatabaseHelper
from src.data_processing import find_cols_with_missing, find_cols_with_high_freq, find_cols_with_outliers
from src.utils import load_data_df
from src.plotting import plot_na_histograms

# process all 3 datasets at the same time
# so that they all keep/remove the same subjects
domains = ['behavioural', 'brain', 'demographic']
holdout_fields = [21003, 34] # age, year of birth

square_conf = True

threshold_na = 0.5
threshold_high_freq = 0.95
threshold_outliers = 100

fpath_holdout = FPATHS['data_holdout_clean']
fpath_dropped_subjects = os.path.join(DPATHS['clean'], 'dropped_subjects.csv')
fpath_dropped_udis = os.path.join(DPATHS['clean'], 'dropped_udis.csv')

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

    dfs_encoded = {} # to be concatenated

    for colname_original in df.columns:

        df_encoded = pd.get_dummies(df[colname_original], prefix=colname_original, prefix_sep='_')
        dfs_encoded[colname_original] = df_encoded.rename(columns=fn_rename)

    return pd.concat(dfs_encoded, axis='columns', names=['udis', 'udis_encoded'])

def square_df(df):
    df = np.square(df)
    df = df.rename(columns=(lambda x: f'{x}_squared')) # append 'squared' to column name
    return df

def remove_bad_cols(df, threshold_na=0.5, threshold_high_freq=0.95, threshold_outliers=100, return_colnames=False):

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

    if return_colnames:
        bad_colnames = set(cols_with_missing).union(cols_without_variability).union(cols_with_outliers)
        return df, list(bad_colnames)

    return df

if __name__ == '__main__':

    print('----- Parameters -----')
    print(f'domains:\t{domains}')
    print(f'holdout_fields:\t{holdout_fields}')
    print(f'threshold_na:\t{threshold_na}')
    print(f'threshold_high_freq:\t{threshold_high_freq}')
    print(f'threshold_outliers:\t{threshold_outliers}')
    print(f'fpath_holdout:\t{fpath_holdout}')
    print(f'fpath_dropped_subjects:\t{fpath_dropped_subjects}')
    print(f'fpath_dropped_udis:\t{fpath_dropped_udis}')
    print(f'fig_prefix:\t{fig_prefix}')
    print(f'dpath_figs:\t{dpath_figs}')
    print('----------------------')

    db_helper = DatabaseHelper(dpath_schema, fpath_udis)

    dfs_data = {}
    dfs_holdout = [] # to be concatenated
    subjects_to_drop = set()

    print('----- Loading unprocessed data -----')
    for domain in domains:

        fpath_data = FPATHS[f'data_{domain}']
        print(f'{domain}: {fpath_data}')

        # load data
        df_data = load_data_df(fpath_data)
        print(f'\tDataframe shape: {df_data.shape}')

        # if data contains holdout UDIs, extract them and drop them from dataframe
        holdout_udis = db_helper.filter_udis_by_field(df_data.columns, holdout_fields)
        if len(holdout_udis) > 0:
            print(f'\tExtracting {len(holdout_udis)} holdout variables')
            dfs_holdout.append(df_data.loc[:, holdout_udis])
            df_data = df_data.drop(columns=holdout_udis)

        # one-hot encode categorical variables
        udis = df_data.columns
        categorical_udis = db_helper.filter_udis_by_value_type(udis, 'categorical')
        if len(categorical_udis) > 0:

            print(f'\tOne-hot encoding {len(categorical_udis)} categorical UDIs')
            df_encoded = one_hot_encode(df_data[categorical_udis]) # returns multiindexed df

            # drop original categorical columns
            df_data = df_data.drop(columns=categorical_udis)

            # add column level for compatibility with df_encoded
            df_data.columns = pd.MultiIndex.from_arrays(
                [df_data.columns, df_data.columns.map(lambda colname: f'{colname}_orig')],
                names=['udis', 'udis_encoded'],
            )
            
            df_data = pd.concat([df_data, df_encoded], axis='columns')
            print(f'\t\tShape after one-hot encoding: {df_data.shape}')

        # square confounders
        if square_conf and domain in ['demographic']:
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

    df_holdout = pd.concat(dfs_holdout, axis='columns')

    print('----- Cleaning data -----')
    mode = 'w'
    dfs_dropped_cols = []
    while len(subjects_to_drop) != 0:

        # write dropped subject IDs
        pd.Series(list(subjects_to_drop), name='eid').to_csv(fpath_dropped_subjects, 
            header=True, index=False, mode=mode)
        mode = 'a' # append for subsequent iterations

        # drop subjects from holdouts dataframe
        df_holdout = df_holdout.drop(index=subjects_to_drop)

        subjects_to_drop_new = set()
        for domain in domains:

            print(f'{domain}')

            # remove bad rows
            print(f'\tRemoving {len(subjects_to_drop)} rows')
            df_clean = dfs_data[domain].drop(index=subjects_to_drop)

            # remove bad columns
            print('\tLooking for bad columns...')
            df_clean, bad_cols = remove_bad_cols(df_clean, return_colnames=True,
                threshold_na=threshold_na, threshold_high_freq=threshold_high_freq, threshold_outliers=threshold_outliers)

            # save in df, to write later
            dfs_dropped_cols.append(pd.DataFrame({'uid':bad_cols, 'domain':domain}))

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

            dfs_data[domain] = df_clean # to be saved in csv file

        subjects_to_drop = subjects_to_drop_new
        print('-------------------------')

    print(f'Holdout dataframe shape: {df_holdout.shape}')

    # save cleaned data
    for domain in domains:
        fpath_out = FPATHS[f'data_{domain}_clean']
        dfs_data[domain].to_csv(fpath_out, header=True, index=True)
        print(f'Saved {domain} data to {fpath_out}')

    # save cleaned holdouts dataframe
    df_holdout.to_csv(fpath_holdout, header=True, index=True)

    # log dropped columns
    pd.concat(dfs_dropped_cols).to_csv(fpath_dropped_udis, header=True, index=False)
