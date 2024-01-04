#!/usr/bin/env python
import re
import sys
from pathlib import Path

import click
import pandas as pd

from src.base import NestedItems
from src.cca import CcaResultsSampleSize
from src.database_helpers import DatabaseHelper
from src.utils import print_params, add_suffix
from aggregate_sample_size_results import find_fpaths_results, combine_results, RE_SAMPLE_SIZE

DNAME_FIGS = 'figs'
DNAME_SUMMARY = 'summary'

CCA_RESULT_TYPES = ['loadings', 'corrs'] # do not support CAs
VALID_RESULT_TYPES = ['summary'] + CCA_RESULT_TYPES

@click.command()
@click.argument('n_PCs_all', nargs=-1, required=True)
@click.option('--dpath-cca', required=True, envvar='DPATH_CCA_SAMPLE_SIZE')
@click.option('--CA', 'i_component', default=1)
@click.option('--subset', default='all')
@click.option('--set-name', default='learn')
@click.option('--cca-type', default='repeated_cv')
@click.option('--sample_size')
@click.option('--what', 'requested_result_types', default=['summary'], multiple=True, type=click.Choice(VALID_RESULT_TYPES))
@click.option('--dpath-schema', required=True, envvar='DPATH_SCHEMA')
@click.option('--fpath-udis', required=True, envvar='FPATH_UDIS')
def get_plot_data(n_pcs_all, dpath_cca, i_component, subset, set_name, cca_type, sample_size, requested_result_types, dpath_schema, fpath_udis):

    requested_result_types = set(requested_result_types)
    print_params(locals())

    i_component = i_component - 1 # zero-indexing
    n_PCs_str = CcaResultsSampleSize.get_dname_PCs(n_pcs_all)
    dpath_subset = Path(dpath_cca, n_PCs_str, subset)
    db_helper = DatabaseHelper(dpath_schema, fpath_udis)

    if not dpath_subset.exists():
        print(f'[ERROR] Directory not found: {dpath_subset}')
        sys.exit(1)

    dpath_out = Path(dpath_cca, DNAME_FIGS)

    for requested_result_type in requested_result_types:
        if requested_result_type == 'summary':
        
            fpath_summary = Path(dpath_cca, n_PCs_str, DNAME_SUMMARY, subset)
            summary = NestedItems.load_fpath(fpath_summary)
            print(f'Loaded results summary: {summary}')

            if sample_size is None:
                sample_size = sorted(summary.levels['sample_size'])[-1]
                print(f'Using largest sample size: {sample_size}')

            data = summary[sample_size]

            for cca_type in data.levels['cca_type']:
                for set_name in data.levels['set_name']:
                    for func in data.levels['func']:
                        for i_dataset in range(len(data.dataset_names)):
                            data[cca_type, set_name, func].loadings[i_dataset].index = db_helper.udis_to_text(data[cca_type, set_name, func].loadings[i_dataset].index, encoded=True, prepend_category=True)

            data.fname = add_suffix(requested_result_type, add_suffix(summary.fname, sample_size))
            data.save(dpath=dpath_out, verbose=True)

        else:

            dpath_subset = Path(dpath_cca, n_PCs_str, subset)
            if sample_size is None:
                sample_size = max([int(re.match(RE_SAMPLE_SIZE, dname.name).group(1)) for dname in dpath_subset.iterdir() if dname.is_dir()])
                print(f'Using largest sample size: {sample_size}')

            dpath_results = dpath_subset / f'sample_size_{sample_size}'
            fpaths_results = find_fpaths_results(dpath_results)

            print(f'Found {len(fpaths_results)} result files in {dpath_results}')

            results_combined = combine_results(fpaths_results, n_components_expected=min([int(n) for n in n_pcs_all]), verbose=False)
            print(results_combined)

            data = {} # to save as pickle
            for i_dataset, dataset_name in enumerate(results_combined.dataset_names):
                data_for_df = []
                for result in results_combined[(sample_size, cca_type, set_name)]:
                    if requested_result_type == 'corrs':
                        requested_result = {f'CA{i_component+1}': getattr(result, requested_result_type)[i_component]}
                    else:
                        dataset_results = getattr(result, requested_result_type)[i_dataset]
                        if isinstance(dataset_results, pd.DataFrame):
                            requested_result = dataset_results.iloc[:, i_component]
                        else:
                            requested_result = dataset_results[:, i_component]
                    data_for_df.append(requested_result)

                df = pd.DataFrame(data_for_df)
                
                if requested_result_type == 'loadings':
                    df.columns = db_helper.udis_to_text(df.columns, encoded=True, prepend_category=True)
                
                fpath_out = dpath_out / f'{"-".join([requested_result_type, subset, str(sample_size), cca_type, set_name, f"CA{i_component+1}", dataset_name])}.csv'
                df.to_csv(fpath_out, index=False, header=True)
                print(f'Saved to {fpath_out}')

                # data[f'df_{dataset_name}'] = df

if __name__ == '__main__':
    get_plot_data()
