#!/usr/bin/env python
import sys
from pathlib import Path

import click
import numpy as np
import matplotlib.pyplot as plt

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

@click.command()
@click.argument('n_PCs_all', nargs=-1, required=True)
@click.option('--dpath-cca', required=True, envvar='DPATH_CCA_SAMPLE_SIZE')
@click.option('--subset', default='all')
@click.option('--CA', 'i_component', default=1)
def aggregate_sample_size_results(n_pcs_all, dpath_cca, subset, i_component):

    print_params(locals())
    i_component = i_component - 1 # zero-indexing
    n_components_expected = min([int(n) for n in n_pcs_all])

    n_PCs_str = CcaResultsSampleSize.get_dname_PCs(n_pcs_all)
    dpath_PCs = Path(dpath_cca, n_PCs_str)

    if not dpath_PCs.exists():
        print(f'[ERROR] Directory not found: {dpath_PCs}')
        sys.exit(1)

    dpath_figs = Path(dpath_cca, DNAME_FIGS, n_PCs_str, subset)

    results_combined = None # sample size, cca_type, set_name

    fpaths_results = [
        p 
        for p in (dpath_PCs/subset).glob('**/*') 
        if (p.is_file() and p.suffix == '.pkl')
    ]

    print(f'Found {len(fpaths_results)} result files for subset {subset}')

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

        for cca_type in results.method_names:

            flip_values = None
            flip_ref = 'learn'

            # if np.any(results[cca_type]['val'].corrs >= HIGH_VAL_CORR_THRESHOLD) and cca_type in DROP_HIGH_VAL_CORR:
            #     print(f'Dropping bad repetition: {fpath_results}')
            #     continue

            if ('repeated' in cca_type) and (flip_ref in results[cca_type].set_names):
                flip_values = np.where(results[cca_type][flip_ref].corrs < 0, -1, 1)
            
            for set_name in results[cca_type].set_names:
                try:
                    sample_size = results.sample_size
                    single_result = results[cca_type][set_name]

                    if len(single_result.corrs) != n_components_expected:
                        print('====================')
                        print(f'Unexpected array shapes in {fpath_results}')
                        print(single_result)
                        print('====================')
                        continue

                    if flip_values is not None:
                        single_result.multiply_corrs(flip_values)
                        single_result.multiply_CAs(flip_values)
                        single_result.multiply_loadings(flip_values)
                    
                    single_result.loadings_to_df(results.udis_datasets)

                    results_combined.append(
                        (sample_size, cca_type, set_name),
                        single_result,
                    )

                except KeyError:
                    continue

    # aggregate results
    results_summary = results_combined.aggregate()

    # TODO check for aggregated null results

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
            plot_corrs(
                corrs=[results_for_row[set_name]['mean'].corrs for set_name in set_names],
                labels=set_names,
                errs=[results_for_row[set_name]['std'].corrs for set_name in set_names],
                err_measure='1 SD',
                ax=ax_corrs,
                bootstrap_corrs=None,
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

    dpath_out = Path(dpath_cca, n_PCs_str, DNAME_OUT)
    results_summary.fname = f'{subset}.pkl'
    results_summary.save(dpath=dpath_out)

if __name__ == '__main__':
    aggregate_sample_size_results()
