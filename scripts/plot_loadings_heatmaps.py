#!/usr/bin/env python
import pickle
import sys
from pathlib import Path

import click
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from scipy.ndimage import correlate

from src.base import NestedItems
from src.cca import CcaResultsSampleSize
from src.database_helpers import DatabaseHelper
from src.plotting import save_fig
from src.utils import print_params

DNAME_FIGS = 'figs'
DNAME_SUMMARY = 'summary'

AX_WIDTH_UNIT = 1.5
AX_HEIGHT = 20

BOOTSTRAP_ALPHA = 0.05

MASK_COLOR_LEVEL = 0.5 # 0 is white, 1 is black
MASK_ALPHA = 0.5 # transparency of black layer for masked regions
OUTLINE_COLOR = 'grey'
OUTLINE_LINEWIDTH = 0.5

SET_NAME = 'learn'

@click.command()
@click.argument('n_PCs_all', nargs=-1, required=True)
@click.option('--dpath-cca', required=True, envvar='DPATH_CCA_SAMPLE_SIZE')
@click.option('--subset', default='all')
@click.option('--CA', 'i_component', default=1)
@click.option('--dpath-schema', required=True, envvar='DPATH_SCHEMA')
@click.option('--fpath-udis', required=True, envvar='FPATH_UDIS')
@click.option('--largest-null/--no-largest-null', 'use_largest_null_model', default=True)
def plot_loadings_heatmaps(n_pcs_all, dpath_cca, subset, i_component, dpath_schema, fpath_udis, use_largest_null_model):

    print_params(locals())
    i_component = i_component - 1 # zero-indexing
    db_helper = DatabaseHelper(dpath_schema, fpath_udis)

    n_PCs_str = CcaResultsSampleSize.get_dname_PCs(n_pcs_all)
    dpath_subset = Path(dpath_cca, n_PCs_str, subset)

    if not dpath_subset.exists():
        print(f'[ERROR] Directory not found: {dpath_subset}')
        sys.exit(1)

    dpath_figs = Path(dpath_cca, DNAME_FIGS)
    dpath_figs_heatmaps = Path(dpath_figs, n_PCs_str, subset)
    
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
    cca_types = summary.levels['cca_type']
    sample_sizes = sorted([int(s) for s in summary.levels['sample_size']])
    largest_sample_size = sample_sizes[-1]

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
            data_for_df_mask = []
            data_for_df_std = [] # TODO
            for sample_size in reversed(sample_sizes):

                loadings = summary[sample_size, cca_type, SET_NAME, 'mean'].loadings[i_col].iloc[:, i_component]
                loadings_std = summary[sample_size, cca_type, SET_NAME, 'std'].loadings[i_col].iloc[:, i_component]

                if use_largest_null_model:
                    sample_size_null_model = largest_sample_size
                else:
                    sample_size_null_model = sample_size

                if summary_null is not None:
                    try:
                        # print(f'{sample_size_null_model}\t{cca_type}\t{SET_NAME}\t{BOOTSTRAP_ALPHA/2}')
                        loadings_null_low = summary_null[sample_size_null_model, cca_type, SET_NAME, f'quantile_{BOOTSTRAP_ALPHA/2}'].loadings[i_col].iloc[:, i_component]
                        # print(f'{sample_size_null_model}\t{cca_type}\t{SET_NAME}\t{1-BOOTSTRAP_ALPHA/2}')
                        loadings_null_high = summary_null[sample_size_null_model, cca_type, SET_NAME, f'quantile_{1-BOOTSTRAP_ALPHA/2}'].loadings[i_col].iloc[:, i_component]

                        # the null model loadings might have different variables than the normal loadings
                        # so we align them by label with pd.concat before computing the mask
                        df_tmp = pd.concat(
                            {
                                'loadings': loadings,
                                'low': loadings_null_low,
                                'high': loadings_null_high,
                            },
                            axis='columns',
                        )
                        
                        # if null loading has no variance, pretend it is missing
                        df_tmp.loc[df_tmp['low'] == df_tmp['high'], ['low', 'high']] = np.nan

                        # impute missing values in mask
                        for col in ['low', 'high']:
                            df_tmp[col] = df_tmp[col].replace(np.nan, df_tmp[col].mean())

                        # print(df_tmp)
                        mask = (df_tmp['loadings'] > df_tmp['low']) & (df_tmp['loadings'] < df_tmp['high'])
                        # print(mask)
                        # return

                    except Exception as ex:
                        raise ex
                else:
                    mask = pd.Series(np.zeros_like(loadings, dtype=bool), index=loadings.index)

                data_mag = loadings
                data_mag.name = sample_size

                data_rank = pd.Series(
                    data=np.argsort(np.argsort(loadings)), # increasing
                    name=sample_size,
                )

                data_mask = pd.Series(mask, name=sample_size)

                data_std = loadings_std
                data_std.name = sample_size

                data_for_df_mag.append(data_mag)
                data_for_df_rank.append(data_rank)
                data_for_df_mask.append(data_mask)
                data_for_df_std.append(data_std)

            df_mag = pd.concat(data_for_df_mag, axis='columns').sort_values(sample_sizes[-1], ascending=False)
            df_rank = pd.concat(data_for_df_rank, axis='columns').sort_values(sample_sizes[-1], ascending=False)
            df_mask: pd.DataFrame = pd.concat(data_for_df_mask, axis='columns').loc[df_mag.index].fillna(False)
            df_std = pd.concat(data_for_df_std, axis='columns').loc[df_mag.index]

            # # re-order based on significance
            # df_mask_pos = df_mask.loc[df_mag[sample_sizes[-1]] >= 0]
            # df_mask_neg = df_mask.loc[(df_mag[sample_sizes[-1]] < 0) | (df_mag[sample_sizes[-1]].isna())]
            # df_mask_pos_sorted = df_mask_pos.sort_values(sample_sizes[-1], ascending=True, kind='stable')
            # df_mask_neg_sorted = df_mask_neg.sort_values(sample_sizes[-1], ascending=False, kind='stable')
            # index_sorted = df_mask_pos_sorted.index.tolist() + df_mask_neg_sorted.index.tolist()
            index_sorted = df_mag[sample_sizes[-1]].sort_values(ascending=True).index.tolist()
            df_mag = df_mag.loc[index_sorted]
            df_rank = df_rank.loc[index_sorted]
            df_mask = df_mask.loc[index_sorted]
            df_std = df_std.loc[index_sorted]

            # human-readable labels
            df_mag.index = db_helper.udis_to_text(df_mag.index, encoded=True, prepend_category=True)
            df_rank.index = db_helper.udis_to_text(df_rank.index, encoded=True, prepend_category=True)
            df_mask.index = db_helper.udis_to_text(df_mask.index, encoded=True, prepend_category=True)
            df_std.index = db_helper.udis_to_text(df_std.index, encoded=True, prepend_category=True)

            sns.heatmap(df_mag, ax=ax_mag)
            sns.heatmap(df_rank, ax=ax_rank)

            if summary_null is not None:
                add_mask_to_heatmap(df_mask, ax_mag)
                add_mask_to_heatmap(df_mask, ax_rank)

            ax_mag.set_title(f'{dataset_name.capitalize()} ({cca_type})')
            ax_rank.set_title(f'{dataset_name.capitalize()} ({cca_type})')

            to_save = {'mag': df_mag, 'rank': df_rank, 'mask': df_mask, 'std': df_std}
            fpath_out = dpath_figs / f'loadings-{subset}-{cca_type}-{SET_NAME}-{dataset_name}-CA{i_component+1}.pkl'
            with fpath_out.open('wb') as file_out:
                pickle.dump(to_save, file_out)
            print(f'Heatmap plot data saved to: {fpath_out}')

    fig_mag.tight_layout()
    fig_rank.tight_layout()

    fpath_fig_mag = dpath_figs_heatmaps / f'{subset}-loadings_heatmap_mag'
    save_fig(fig_mag, fpath_fig_mag)

    fpath_fig_rank = dpath_figs_heatmaps / f'{subset}-loadings_heatmap_rank'
    save_fig(fig_rank, fpath_fig_rank)

def add_mask_to_heatmap(df_mask: pd.DataFrame, ax: plt.Axes):
    
    # add the grey mask
    df_mask_tmp = df_mask.applymap(lambda x: MASK_COLOR_LEVEL if (x and not np.isnan(x)) else np.nan)
    ax.pcolormesh(df_mask_tmp, cmap='Greys', vmin=0, vmax=1, alpha=MASK_ALPHA, linewidth=0)

    # add the contours
    outlines = create_cluster_contour(df_mask.fillna(True))
    for x_line, y_line in outlines:
        ax.plot(x_line, y_line, color=OUTLINE_COLOR, linewidth=OUTLINE_LINEWIDTH)

def create_cluster_contour(mask):
    # taken from https://github.com/mmagnuski/sarna/blob/4bac30274e16fb434f7e3b71c233ca1e4d49edff/mypy/viz.py (MIT license)

    orig_mask_shape = mask.shape
    mask_int = np.pad(mask.astype('int'), ((1, 1), (1, 1)), 'constant')
    kernels = {'upper': np.array([[-1], [1], [0]]),
               'lower': np.array([[0], [1], [-1]]),
               'left': np.array([[-1, 1, 0]]),
               'right': np.array([[0, 1, -1]])}
    lines = {k: (correlate(mask_int, v) == 1).astype('int')
             for k, v in kernels.items()}

    search_order = {'upper': ['right', 'left', 'upper'],
                    'right': ['lower', 'upper', 'right'],
                    'lower': ['left', 'right', 'lower'],
                    'left': ['upper', 'lower', 'left']}
    movement_direction = {'upper': [0, 1], 'right': [1, 0],
                          'lower': [0, -1], 'left': [-1, 0]}
    search_modifiers = {'upper_left': [-1, 1], 'right_upper': [1, 1],
                        'lower_right': [1, -1], 'left_lower': [-1, -1]}
    finish_modifiers = {'upper': [-0.5, 0.5], 'right': [0.5, 0.5],
                        'lower': [0.5, -0.5], 'left': [-0.5, -0.5]}

    # current index - upmost upper line
    upper_lines = np.where(lines['upper'])
    outlines = list()

    while len(upper_lines[0]) > 0:
        current_index = np.array([x[0] for x in upper_lines])
        closed_shape = False
        current_edge = 'upper'
        edge_points = [tuple(current_index + [-0.5, -0.5])]
        direction = movement_direction[current_edge]

        while not closed_shape:
            new_edge = None
            ind = tuple(current_index)

            # check the next edge
            for edge in search_order[current_edge]:
                modifier = '_'.join([current_edge, edge])
                has_modifier = modifier in search_modifiers
                if has_modifier:
                    modifier_value = search_modifiers[modifier]
                    test_ind = tuple(current_index + modifier_value)
                else:
                    test_ind = ind

                if lines[edge][test_ind] == 1:
                    new_edge = edge
                    lines[current_edge][ind] = -1
                    break
                elif lines[edge][test_ind] == -1: # -1 means 'visited'
                    closed_shape = True
                    new_edge = 'finish'
                    lines[current_edge][ind] = -1
                    break

            if not new_edge == current_edge:
                edge_points.append(tuple(
                    current_index + finish_modifiers[current_edge]))
                direction = modifier_value if has_modifier else [0, 0]
                current_edge = new_edge
            else:
                direction = movement_direction[current_edge]

            current_index += direction
        x = np.array([l[1] for l in edge_points])
        y = np.array([l[0] for l in edge_points])
        outlines.append([x, y])
        upper_lines = np.where(lines['upper'] > 0)
    _correct_all_outlines(outlines, orig_mask_shape, extent=[0, orig_mask_shape[1], 0, orig_mask_shape[0]])
    return outlines

def _correct_all_outlines(outlines, orig_mask_shape, extent=None):
    # taken from https://github.com/mmagnuski/sarna/blob/4bac30274e16fb434f7e3b71c233ca1e4d49edff/mypy/viz.py (MIT license)
    if extent is not None:
        orig_ext = [-0.5, orig_mask_shape[1] - 0.5,
                    -0.5, orig_mask_shape[0] - 0.5]
        orig_ranges = [orig_ext[1] - orig_ext[0],
                       orig_ext[3] - orig_ext[2]]
        ext_ranges = [extent[1] - extent[0],
                       extent[3] - extent[2]]
        scales = [ext_ranges[0] / orig_ranges[0],
                  ext_ranges[1] / orig_ranges[1]]

    def find_successive(vec):
        vec = vec.astype('int')
        two_consec = np.where((vec[:-1] + vec[1:]) == 2)[0]
        return two_consec

    for current_outlines in outlines:
        x_lim = (0, orig_mask_shape[1])
        y_lim = (0, orig_mask_shape[0])

        x_above = current_outlines[0] > x_lim[1]
        x_below = current_outlines[0] < x_lim[0]
        y_above = current_outlines[1] > y_lim[1]
        y_below = current_outlines[1] < y_lim[0]

        x_ind, y_ind = list(), list()
        for x in [x_above, x_below]:
            x_ind.append(find_successive(x))
        for y in [y_above, y_below]:
            y_ind.append(find_successive(y))

        all_ind = np.concatenate(x_ind + y_ind)

        if len(all_ind) > 0:
            current_outlines[1] = np.insert(current_outlines[1],
                                            all_ind + 1, np.nan)
            current_outlines[0] = np.insert(current_outlines[0],
                                            all_ind + 1, np.nan)
        # compensate for padding
        current_outlines[0] = current_outlines[0] - 1.
        current_outlines[1] = current_outlines[1] - 1.

        if extent is not None:
            current_outlines[0] = ((current_outlines[0] + 0.5) * scales[0]
                                   + extent[0])
            current_outlines[1] = ((current_outlines[1] + 0.5) * scales[1]
                                   + extent[2])

if __name__ == '__main__':
    plot_loadings_heatmaps()
