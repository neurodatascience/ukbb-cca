import numpy as np
import matplotlib.pyplot as plt

def plot_na_histograms(df, return_freqs=False):

    freqs_na_row = df.isna().mean(axis='columns')
    freqs_na_col = df.isna().mean(axis='index')

    bins = np.arange(0, 1.06, 0.05) # 21 bins

    fig, (ax_rows, ax_cols) = plt.subplots(nrows=1, ncols=2, figsize=(10, 4))

    ax_rows.hist(freqs_na_row, bins=bins)
    ax_rows.set_xlabel('Frequency')
    ax_rows.set_ylabel('Count')
    ax_rows.set_title('NaN frequencies in rows')

    ax_cols.hist(freqs_na_col, bins=bins)
    ax_cols.set_xlabel('Frequency')
    ax_cols.set_ylabel('Count')
    ax_cols.set_title('NaN frequencies in columns')

    fig.tight_layout()

    if return_freqs:
        return fig, freqs_na_row, freqs_na_col
    else:
        return fig
