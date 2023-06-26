#!/usr/bin/env python
import sys
from collections import defaultdict
from pathlib import Path

import click
import numpy as np
import matplotlib.pyplot as plt

from src.base import NestedItems
from src.cca import CcaResultsSampleSize, CcaResultsCombined
from src.plotting import save_fig, plot_corrs
from src.utils import print_params

DNAME_FIGS = 'figs'
DNAME_OUT = 'summary'

# HIGH_VAL_CORR_THRESHOLD = 0.9
# DROP_HIGH_VAL_CORR = ['without_cv']

AX_WIDTH_CORRS = 8
AX_WIDTH_LOADINGS = 4
AX_WIDTH_CAS = 4
AX_HEIGHT = 4

CA_X_INDEX = 1 # brain
CA_Y_INDEX = 0 # behavioural

BOOTSTRAP_ALPHA = 0.05
N_TEST = 30

@click.command()
@click.argument('n_PCs_all', nargs=-1, required=True)
@click.option('--dpath-cca', required=True, envvar='DPATH_CCA_SAMPLE_SIZE')
@click.option('--subset', default='all')
@click.option('--CA', 'i_component', default=1)
@click.option('--test/--no-test', 'is_test', default=False)
def aggregate_sample_size_results(n_pcs_all, dpath_cca, subset, i_component, is_test):

    print_params(locals())

    def find_fpaths_results(dpath):
        fpaths = [
            p for p in Path(dpath).glob('**/*')
            if (p.is_file() and p.suffix == '.pkl')
        ]

        # TODO check if order is really necessary
        func_sort = (lambda x: -int("".join([i for i in str(x) if i.isdigit()])))
        fpaths = sorted(fpaths, key=func_sort)

        # if testing, don't use all the results
        if is_test:
            fpaths = fpaths[:N_TEST]

        return fpaths
    
    def combine_results(fpaths_results):

        results_combined = None
        top_loading_indices_all = {}

        for fpath_results in fpaths_results:
            print(f'fpath_results: {fpath_results}')

            try:
                results = CcaResultsSampleSize.load_fpath(fpath_results)
            except Exception as ex:
                print(ex)
                continue

            if results_combined is None:
                results_combined = CcaResultsCombined(
                    data=results,
                    levels=['sample_size', 'cca_type', 'set_name'],
                )

            if len(results.dataset_names) > 2:
                raise NotImplementedError(
                    'Not implemented for more than 2 datasets'
                )
            
            sample_size = results.sample_size

            for cca_type in results.method_names:

                flip_ref = 'learn'
                i_dataset_ref = 1 # pick brain dataset because it has no missing data

                # find the indices of the loadings with largest magnitude (in one of the datasets)
                
                # print(f'cca_type: {cca_type}\tsample_size: {sample_size}')
                loadings_dataset = results[cca_type][flip_ref].loadings[i_dataset_ref]
                try:
                    top_loading_indices = top_loading_indices_all[cca_type]#[sample_size]
                    # print(f'top_loading_indices: {top_loading_indices}')
                except KeyError:
                    top_loading_indices = np.argmax(np.abs(loadings_dataset), axis=0)
                    # top_loading_indices_all[cca_type][sample_size] = top_loading_indices
                    top_loading_indices_all[cca_type] = top_loading_indices
                    # print(f'\tRecomputing top_loading_indices: {top_loading_indices}')
                    # signs = np.sign(u[max_abs_cols, range(u.shape[1])])
                    
                # figure out which ICs need to be flipped
                signs = np.sign(loadings_dataset[top_loading_indices, range(loadings_dataset.shape[1])])
                # print(f'\tMaximum values are: {loadings_dataset[top_loading_indices, range(loadings_dataset.shape[1])]}')
                # print(f'\tsigns: {signs}')
                
                for set_name in results[cca_type].set_names:
                    try:
                        single_result = results[cca_type][set_name]

                        if len(single_result.corrs) != n_components_expected:
                            print('====================')
                            print(f'Unexpected array shapes in {fpath_results}')
                            print(single_result)
                            print('====================')
                            continue

                        single_result.multiply_CAs(signs)
                        single_result.multiply_loadings(signs)

                        # print(f'NA count: {[np.isnan(loadings).sum() for loadings in single_result.loadings]}')
                        
                        single_result.loadings_to_df(results.udis_datasets)

                        results_combined.append(
                            (sample_size, cca_type, set_name),
                            single_result,
                        )

                    except KeyError:
                        continue

        return results_combined

    i_component = i_component - 1 # zero-indexing
    n_components_expected = min([int(n) for n in n_pcs_all])

    n_PCs_str = CcaResultsSampleSize.get_dname_PCs(n_pcs_all)
    dpath_PCs = Path(dpath_cca, n_PCs_str)

    if not dpath_PCs.exists():
        print(f'[ERROR] Directory not found: {dpath_PCs}')
        sys.exit(1)

    dpath_figs = Path(dpath_cca, DNAME_FIGS, n_PCs_str, subset)

    fpaths_results = find_fpaths_results(dpath_PCs/subset)
    print(f'Found {len(fpaths_results)} result files for subset {subset}')

    fpaths_results_null = find_fpaths_results(dpath_PCs/f'{subset}-null_model')
    print(f'Found {len(fpaths_results_null)} result files for corresponding null model')


    results_combined = combine_results(fpaths_results)
    results_combined_null = combine_results(fpaths_results_null)

    # aggregate results
    results_summary = results_combined.aggregate()
    results_summary_null = results_combined_null.aggregate({
        f'quantile_{quantile}': (lambda data, quantile=quantile:
            CcaResultsCombined.agg_func_helper(
                data,
                'quantile',
                np.quantile,
                kwargs_pd={'q': quantile},
                kwargs_np={'q': quantile},
            )
        )
        for quantile in [BOOTSTRAP_ALPHA, 1 - BOOTSTRAP_ALPHA]
    })

    print(results_summary)
    print(results_summary_null)

    dpath_out = Path(dpath_cca, n_PCs_str, DNAME_OUT)
    results_summary.fname = f'{subset}.pkl'
    results_summary.save(dpath=dpath_out)

    dpath_out_null = Path(dpath_cca, n_PCs_str, DNAME_OUT)
    results_summary_null.fname = f'{subset}-null_model.pkl'
    results_summary_null.save(dpath=dpath_out_null)

    # return

    # plot
    for sample_size in results_summary.levels['sample_size']:

        print(f'sample_size: {sample_size}')
        
        results_summary_page = results_summary[sample_size]
        cca_types = results_summary_page.levels['cca_type']
        set_names = results_summary_page.levels['set_name']
        n_rows = len(cca_types)
        n_cols = len(set_names) # only for CAs/loadings

        fig_corrs, axes_corrs = plt.subplots(
            nrows=n_rows,
            figsize=(AX_WIDTH_CORRS, AX_HEIGHT*n_rows),
            sharey='row',
        )

        # fig_CAs, axes_CAs = plt.subplots(
        #     nrows=n_rows,
        #     ncols=n_cols,
        #     figsize=(AX_WIDTH_CAS*n_cols, AX_HEIGHT*n_rows),
        #     sharex='all',
        #     sharey='all',
        #     squeeze=False,
        # )

        # TODO cols should be dataset_names instead of set here
        # fig_loadings, axes_loadings = plt.subplots(
        #     nrows=n_rows,
        #     ncols=n_cols,
        #     figsize=(AX_WIDTH_CAS, AX_HEIGHT*n_rows),
        #     sharex='all',
        #     sharey='all',
        #     squeeze=False,
        # )

        for i_row, cca_type in enumerate(cca_types):

            # TODO check if null model results exist for this sample size and CCA type
            # then pass them to plotting function as bootstrap_corrs=

            results_for_row = results_summary_page[cca_type]
            ax_corrs = axes_corrs[i_row]

            if results_summary_null is not None:
                try:
                    bootstrap_corrs = [
                        results_summary_null[sample_size][cca_type]['val'][f'quantile_{quantile}'].corrs
                        for quantile in [BOOTSTRAP_ALPHA, 1 - BOOTSTRAP_ALPHA]
                    ]
                except Exception:
                    bootstrap_corrs = None
            else:
                bootstrap_corrs = None

            plot_corrs(
                corrs=[results_for_row[set_name]['mean'].corrs for set_name in set_names],
                labels=set_names,
                errs=[results_for_row[set_name]['std'].corrs for set_name in set_names],
                err_measure='1 SD',
                ax=ax_corrs,
                bootstrap_corrs=bootstrap_corrs,
                bootstrap_alpha=BOOTSTRAP_ALPHA,
            )
            ax_corrs.set_title(cca_type)

            # axes_CAs_row = axes_CAs[i_row]
            # plot_scatter(
            #     x_data=[results_for_row[set_name]['mean'].CAs[CA_X_INDEX][:, i_component] for set_name in set_names],
            #     y_data=[results_for_row[set_name]['mean'].CAs[CA_Y_INDEX][:, i_component] for set_name in set_names],
            #     c_data=None,
            #     x_label=results_summary_page.dataset_names[CA_X_INDEX],
            #     y_label=results_summary_page.dataset_names[CA_Y_INDEX],
            #     ax_titles=[f'{cca_type}, {set_name}' for set_name in set_names],
            #     texts_upper_left=[
            #         f'$\mathregular{{CA_{{{i_component+1}}} = {results_for_row[set_name]["mean"].corrs[i_component]:.3f}}}$'
            #         for set_name in set_names
            #     ],
            #     axes=axes_CAs_row,
            #     # TODO makes no sense to average over subjects...
            #     # x_errs=[results_for_row[set_name]['std'].CAs[CA_X_INDEX][:, i_component] for set_name in set_names],
            #     # y_errs=[results_for_row[set_name]['std'].CAs[CA_Y_INDEX][:, i_component] for set_name in set_names],
            # )

        fig_corrs.tight_layout()
        # fig_CAs.tight_layout()

        # also save figures
        fpath_fig_corrs = dpath_figs / f'{subset}_{sample_size}-corrs'
        save_fig(fig_corrs, fpath_fig_corrs)

    # dpath_out = Path(dpath_cca, n_PCs_str, DNAME_OUT)
    # results_summary.fname = f'{subset}.pkl'
    # results_summary.save(dpath=dpath_out)

    # dpath_out_null = Path(dpath_cca, n_PCs_str, DNAME_OUT)
    # results_summary_null.fname = f'{subset}-null_model.pkl'
    # results_summary_null.save(dpath=dpath_out_null)

if __name__ == '__main__':
    aggregate_sample_size_results()
