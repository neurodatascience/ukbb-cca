from math import ceil
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from src.utils import make_parent_dir

def save_fig(fig: Figure, fpath, dpi=300, bbox_inches='tight', ext='.png', verbose=True):
    fpath = Path(fpath).with_suffix(ext)
    make_parent_dir(fpath)
    fig.savefig(fpath, dpi=dpi, bbox_inches=bbox_inches)
    if verbose:
        print(f'Figure saved to {fpath}')

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

def plot_group_histograms(data, bins):
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.hist(data, bins=bins)
    ax.set_title('Age distribution')
    ax.set_xlabel('Age')
    ax.set_ylabel('Count')
    return fig

def plot_corrs(corrs, labels, colors=None, errs=None, err_measure='error', 
        bootstrap_corrs=None, max_CAs=100, bootstrap_alpha=0.05,
        fmt='o', markersize=4, ax=None):

    n_CAs_to_plot = min(len(corrs[0]), max_CAs)

    figsize = (max(n_CAs_to_plot//10, 4), 4)
    if ax is None:
        fig_corr, ax = plt.subplots(figsize=figsize)
    else:
        fig_corr = ax.get_figure()

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
            ax.errorbar(x, y, errs[i_data], fmt=fmt, markersize=markersize, ecolor='black', label=f'{label.capitalize()} (\u00B1 {err_measure})', color=color)
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

def plot_lr_results_comparison(scores_all, max_values=None, dims=None, ylabel=None, fmts=None, colors=None, max_colors=['black', 'grey'], max_components=50, alpha=0.9):

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
        for i_fmt, cumulative_label in enumerate(scores_all[component_type].keys()):
            if fmts is not None:
                fmt = fmts[i_fmt]
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
                try:
                    max_value = max_values[component_type][cumulative_label][set_name]
                    ax.axhline(max_value, linestyle='--', color=max_colors[i_fmt], label=f'{cumulative_label.capitalize()}, {set_name} (max)', zorder=-10)
                except:
                    continue
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

def plot_scatter(x_data, y_data, c_data, x_label=None, y_label=None, ax_titles=None, texts_upper_left=None, texts_bottom_right=None, cbar_label=None, alpha=1, axes=None, x_errs=None, y_errs=None, with_colorbar=True):

    n_sets = len(x_data)
    n_cols = n_sets

    if c_data is None or any([c is None for c in c_data]):
        c_data = [None for _ in range(n_sets)]
        vmin = None
        vmax = None
        with_colorbar = False
    else:
        c_data_concatenated = np.concatenate([c.flatten() for c in c_data])
        vmin = c_data_concatenated.min()
        vmax = c_data_concatenated.max()

    if x_errs is None:
        x_errs = [None for _ in range(n_sets)]
    if y_errs is None:
        y_errs = [None for _ in range(n_sets)]

    if axes is None:
        fig, axes = plt.subplots(ncols=n_cols, figsize=(6.5*n_cols,4), sharey='all', sharex='all')
    else:
        fig = axes[0].get_figure()

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

        ax.errorbar(x, y, yerr=y_errs[i_set], xerr=x_errs[i_set], fmt='None', ecolor='black', zorder=-1)

        sc = ax.scatter(x, y, c=c,
            vmin=vmin, vmax=vmax,
            s=10,
            linewidths=0.3,
            edgecolors='black',
            alpha=alpha,
        )

        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)

        ax.set_title(ax_title)

        # if text_upper_left is not None:
        ax.text(0.05, 0.95, text_upper_left,
            horizontalalignment='left',
            verticalalignment='top',
            transform = ax.transAxes,
        )

        # if text_bottom_right is not None:
        ax.text(0.95, 0.05, text_bottom_right,
            horizontalalignment='right',
            verticalalignment='bottom',
            transform = ax.transAxes,
        )

        if with_colorbar:
            fig.colorbar(sc, ax=ax, label=cbar_label)

    fig.tight_layout()
    return fig

def plot_bar(x_data, y_data, c_data, x_label=None, y_label=None, ax_titles=None, axes=None, height=0.8, **kwargs):

    print(f'Ignoring args: {kwargs}')

    n_sets = len(x_data)
    n_cols = n_sets

    if c_data is None or any([c is None for c in c_data]):
        c_data = [None for _ in range(n_sets)]

    if axes is None:
        fig, axes = plt.subplots(ncols=n_cols, figsize=(6.5*n_cols,4), sharey='all', sharex='all')
    else:
        fig = axes[0].get_figure()

    for i_set, (ax, x, y, c) in enumerate(zip(axes, x_data, y_data, c_data)):

        if ax_titles is not None:
            ax_title = ax_titles[i_set]
        else:
            ax_title = None

        ax.barh(y=y, width=x, color=c, height=height)

        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)

        ax.set_title(ax_title)


def plot_CAs(x_data, y_data, c_data, x_label=None, y_label=None, ax_titles=None, texts_upper_left=None, texts_bottom_right=None, cbar_label=None, alpha=1, axes=None, x_errs=None, y_errs=None):

    return plot_scatter(
        x_data=x_data,
        y_data=y_data,
        c_data=c_data,
        x_label=f'Canonical scores ({x_label} data)',
        y_label=f'Canonical scores ({y_label} data)',
        ax_titles=[f'{ax_title} ({len(x)} subjects)' for x, ax_title in zip(x_data, ax_titles)],
        texts_upper_left=texts_upper_left,
        texts_bottom_right=texts_bottom_right,
        cbar_label=cbar_label,
        alpha=alpha,
        axes=axes,
        x_errs=x_errs,
        y_errs=y_errs,
        with_colorbar=True,
    )

def plot_loadings(plot_type, loadings, labels, ax_titles=None, n_loadings=10, colours=None, errs=None, ax_width=10, ax_height_unit=0.5, axes=None):
    
    n_datasets = len(loadings)
    n_loadings = [min(n_loadings, len(l)) for l in loadings]

    if colours is None:
        colours = [None for _ in range(n_datasets)]
    if errs is None:
        errs = [None for _ in range(n_datasets)]    

    # get top/bottom loadings
    idx_loadings_selected = []
    for loading, n in zip(loadings, n_loadings):
        idx_sorted = np.argsort(loading) # ascending order
        idx_loadings_selected.append(np.concatenate((
            idx_sorted[:n],     # bottom n
            idx_sorted[-n:],    # top n
        )))
    labels = [label[idx] for (label, idx) in zip(labels, idx_loadings_selected)]

    y_data = [np.arange(n*2) for n in n_loadings]

    n_cols = n_datasets
    if axes is not None:
        fig = axes.ravel()[0].get_figure()
    else:
        fig, axes = plt.subplots(ncols=n_cols, figsize=(ax_width, ax_height_unit*max(n_loadings)))

    x_data = [loading[idx] for (loading, idx) in zip(loadings, idx_loadings_selected)]
    c_data = [colour[idx] if colour is not None else None for (colour, idx) in zip(colours, idx_loadings_selected)]

    if plot_type == 'scatter':
        plot_scatter(
            x_data=x_data,
            y_data=y_data,
            c_data=c_data,
            x_errs=[err[idx] if err is not None else None for (err, idx) in zip(errs, idx_loadings_selected)],
            axes=axes,
            ax_titles=ax_titles,
        )
    elif plot_type == 'bar':
        plot_bar(
            x_data=x_data,
            y_data=y_data,
            c_data=c_data,
            axes=axes,
            ax_titles=ax_titles,
        )
    else:
        raise ValueError(f'Invalid plot_type: {plot_type}')

    for ax, label, y in zip(axes, labels, y_data):
        ax.set_yticks(y)
        ax.set_yticklabels(label)

    fig.tight_layout()

    return fig

def plot_loadings_scatter(loadings, labels, ax_titles=None, n_loadings=10, colours=None, errs=None, ax_width=10, ax_height_unit=0.5, axes=None):
    return plot_loadings(
        'scatter', loadings=loadings, labels=labels, ax_titles=ax_titles, 
        n_loadings=n_loadings, colours=colours, errs=errs, ax_width=ax_width, 
        ax_height_unit=ax_height_unit, axes=axes)

def plot_loadings_bar(loadings, labels, ax_titles=None, n_loadings=10, colours=None, errs=None, ax_width=10, ax_height_unit=0.5, axes=None):
    return plot_loadings(
        'bar', loadings=loadings, labels=labels, ax_titles=ax_titles, 
        n_loadings=n_loadings, colours=colours, errs=errs, ax_width=ax_width, 
        ax_height_unit=ax_height_unit, axes=axes)
