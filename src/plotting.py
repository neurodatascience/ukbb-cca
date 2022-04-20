
from math import ceil
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

def plot_corrs(corrs, labels, colors=None, errs=None, err_measure='error', 
        bootstrap_corrs=None, max_CAs=100, bootstrap_alpha=0.05,
        fmt='o', markersize=4):

    n_CAs_to_plot = min(len(corrs[0]), max_CAs)

    figsize = (max(n_CAs_to_plot//10, 4), 4)
    fig_corr, ax = plt.subplots(figsize=figsize)

    x = np.arange(n_CAs_to_plot) + 1

    # plot learning set correlations (with error bars)
    markersize = 4
    for i_data, (y, label) in enumerate(zip(corrs, labels)):
        y = y[:n_CAs_to_plot]
        try:
            color = colors[i_data]
        except:
            color = None
        if errs is not None:
            ax.errorbar(x, y, errs, fmt=fmt, markersize=markersize, ecolor='black', label=f'{label.capitalize()} (\u00B1 {err_measure})', color=color)
        else:
            ax.plot(x, y, fmt, markersize=markersize, label=label.capitalize(), color=color)

    # bootstrap confidence interval
    if bootstrap_corrs is not None:
        rbga_black = (0, 0, 0, 1)
        rbga_grey = (0.5, 0.5, 0.5, 0.3)
        bootstrap_ci = np.quantile(bootstrap_corrs[:n_CAs_to_plot], [bootstrap_alpha, 1-bootstrap_alpha], axis=1)
        ax.fill_between(
            x, bootstrap_ci[0], bootstrap_ci[1], 
            edgecolor=rbga_black, facecolor=rbga_grey,
            label=f'Bootstrapped null distribution\n({int(bootstrap_alpha*100)}th-{int((1-bootstrap_alpha)*100)}th percentiles)',
        )

        handles, labels = ax.get_legend_handles_labels()
        correct_order = [2, 0, 1] # TODO might not be correct
        handles_sorted = [handles[i] for i in correct_order]
        labels_sorted = [labels[i] for i in correct_order]
        ax.legend(handles_sorted, labels_sorted)
    else:
        ax.legend()

    ax.set_xlim(left=min(x)-1, right=max(x)+1)
    ax.set_xlabel('Canonical axes')
    ax.set_ylabel('Correlation')

    return fig_corr, n_CAs_to_plot

def plot_lr_results_comparison(scores_all, dims=None, ylabel=None, fmts=None, colors=None, max_components=50, alpha=0.9):

    if dims is not None:
        n_rows, n_cols = dims
    else:
        n_rows = 2
        n_cols = ceil(len(scores_all) / n_rows)

    ax_width = max(max_components//10, 4)
    ax_height = 4

    fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(n_cols*ax_width, n_rows*ax_height), sharey='all')
    if n_rows == 1:
        axes = np.array([axes])
    if n_cols == 1:
        axes = np.array([axes])

    i_ax = 0
    for component_type in scores_all.keys():
        ax = axes.ravel()[i_ax]
        for i_cum, cumulative_label in enumerate(scores_all[component_type].keys()):
            if fmts is not None:
                fmt = fmts[i_cum]
            else:
                fmt = None
            for i_set, set_name in enumerate(scores_all[component_type][cumulative_label].keys()):
                if colors is not None:
                    color = colors[i_set]
                else:
                    color = None
                data = scores_all[component_type][cumulative_label][set_name]
                n_components_to_plot = min(len(data), max_components)
                x = np.arange(n_components_to_plot) + 1
                ax.plot(x, data[:n_components_to_plot], fmt, color=color, label=f'{cumulative_label.capitalize()}, {set_name}', alpha=alpha)
        ax.set_title(component_type)
        ax.set_xlabel('Components')
        ax.set_ylabel(ylabel)
        ax.set_xlim(left=min(x)-1, right=max(x)+1)
        ax.legend()
        i_ax += 1

    while i_ax < axes.size:
        axes.ravel()[i_ax].set_axis_off()
        i_ax += 1

    fig.tight_layout()
    return fig, n_components_to_plot

def plot_scatter(x_data, y_data, c_data, x_label, y_label, ax_titles=None, texts_upper_left=None, texts_bottom_right=None, cbar_label=None, alpha=1):

    n_sets = len(x_data)
    n_cols = n_sets

    c_data_concatenated = np.concatenate([c.flatten() for c in c_data])
    vmin = c_data_concatenated.min()
    vmax = c_data_concatenated.max()

    fig, axes = plt.subplots(ncols=n_cols, figsize=(6.5*n_cols,4), sharey='all', sharex='all')

    for i_set, (ax, x, y, c) in enumerate(zip(axes, x_data, y_data, c_data)):

        if ax_titles is not None:
            ax_title = ax_titles[i_set]
        else:
            ax_title = None
        
        if texts_upper_left is not None:
            text_upper_left = texts_upper_left[i_set]
        else:
            text_upper_left = ''

        if texts_bottom_right is not None:
            text_bottom_right = texts_bottom_right[i_set]
        else:
            text_bottom_right = ''

        sc = ax.scatter(x, y, c=c,
            vmin=vmin, vmax=vmax,
            s=10,
            linewidths=0.3,
            edgecolors='black',
            alpha=alpha,
        )

        ax.set_xlabel(f'Canonical scores ({x_label} data)')
        ax.set_ylabel(f'Canonical scores ({y_label} data)')

        ax.set_title(f'{ax_title} ({len(c)} subjects)')

        ax.text(0.05, 0.95, text_upper_left,
            horizontalalignment='left',
            verticalalignment='top',
            transform = ax.transAxes,
        )

        ax.text(0.95, 0.05, text_bottom_right,
            horizontalalignment='right',
            verticalalignment='bottom',
            transform = ax.transAxes,
        )

        fig.colorbar(sc, ax=ax, label=cbar_label)

    fig.tight_layout()
    return fig
