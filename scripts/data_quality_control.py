
import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
from paths import DPATHS, FPATHS

def zscore_df(df):

    mean = df.mean(axis='index', skipna=True)
    std = df.std(axis='index', skipna=True)

    return (df - mean) / std

def plot_histograms(freq_na_col, freq_na_row, threshold_col, threshold_row):

    fig, (ax_cols, ax_rows) = plt.subplots(nrows=1, ncols=2, figsize=(10, 4))

    ax_cols.hist(freq_na_col, bins=20)
    ylims = ax_cols.get_ylim()
    ax_cols.vlines(threshold_col, *ylims, color='black', linestyles='--')
    ax_cols.set_ylim(*ylims)
    ax_cols.set_xlabel('Frequency')
    ax_cols.set_ylabel('Count')
    ax_cols.set_title('NaN frequencies in columns')

    ax_rows.hist(freq_na_row, bins=20)
    ylims = ax_rows.get_ylim()
    ax_rows.vlines(threshold_row, *ylims, color='black', linestyles='--')
    ax_rows.set_ylim(*ylims)
    ax_rows.set_xlabel('Frequency')
    ax_rows.set_ylabel('Count')
    ax_rows.set_title('NaN frequencies in rows (after removing columns)')

    fig.tight_layout()

    return fig

if __name__ == '__main__':

    threshold_col = 0.5 # maximum acceptable nan frequency
    threshold_row = 0.5 # maximum acceptable nan frequency
    normalize = True
    fill_na_with_mean = True

    configs = {
        'behavioural': {
            'fpath_data': FPATHS['data_behavioural'],
            'fpath_out': FPATHS['data_behavioural_qc'],
        },
        # 'brain': {
        #     'fpath_data': FPATHS['data_brain'],
        #     'fpath_out': FPATHS['data_brain_qc'],
        # },
        # 'demographic': {
        #     'fpath_data': FPATHS['data_demographic'],
        #     'fpath_out': FPATHS['data_brain_qc'],
        # },
    }

    if len(sys.argv) != 2:
        raise ValueError(f'Usage: {sys.argv[0]} <type>')

    type = sys.argv[1]

    try:
        config = configs[type]
    except KeyError:
        raise ValueError(f'Invalid type "{type}". Accepted types are: {configs.keys()}')

    fpath_data = config['fpath_data']
    fpath_out = config['fpath_out']
    fpath_fig_out = os.path.join(DPATHS['figures'], f'fig_qc_{type}.png')

    print('----- Parameters -----')
    print(f'fpath_data:\t\t{fpath_data}')
    print(f'fpath_out:\t\t{fpath_out}')
    print(f'fpath_fig_out:\t{fpath_fig_out}')
    print(f'threshold_col:\t{threshold_col}')
    print(f'threshold_row:\t{threshold_row}')
    print(f'normalize:\t{normalize}')
    print(f'fill_na_with_mean:\t{fill_na_with_mean}')
    print('----------------------')

    # load data
    df_data = pd.read_csv(fpath_data, index_col='eid')

    # columns
    freq_na_col = df_data.isna().mean(axis='index')
    cols_thresholded = freq_na_col.loc[freq_na_col <= threshold_col].index

    print(f'Keeping {len(cols_thresholded)} out of {len(freq_na_col)} columns')

    # remove columns from dataframe, then check rows
    df_data_filtered = df_data.loc[:, cols_thresholded]

    # rows
    freq_na_row = df_data_filtered.isna().mean(axis='columns')
    rows_thresholded = freq_na_row.loc[freq_na_row <= threshold_row].index

    print(f'Keeping {len(rows_thresholded)} out of {len(freq_na_row)} rows')

    # remove rows
    df_data_filtered = df_data_filtered.loc[rows_thresholded]

    # figure
    fig = plot_histograms(freq_na_col, freq_na_row, threshold_col, threshold_row)
    fig.savefig(fpath_fig_out, dpi=300, bbox_inches='tight')

    # drop columns that only contain one unique value (excluding NaNs)
    nunique = df_data_filtered.nunique()
    cols_to_drop = nunique[nunique == 1].index
    df_data_filtered = df_data_filtered.drop(columns=cols_to_drop)
    if len(cols_to_drop) > 0:
        print(f'Dropped {len(cols_to_drop)} columns that only contained one value')

    # normalize
    df_data_normalized = zscore_df(df_data_filtered)

    # save
    df_data_normalized.to_csv(fpath_out, header=True, index=True)
