#!/usr/bin/env python
import sys
from pathlib import Path

import click
import pandas as pd
import seaborn as sns

from src.cca import CcaResultsSampleSize
from src.plotting import save_fig
from src.utils import make_parent_dir, print_params

DNAME_FIGS = 'figs'

@click.command()
@click.argument('n_PCs_all', nargs=-1, required=True)
@click.option('--dpath-cca', required=True, envvar='DPATH_CCA_SAMPLE_SIZE')
@click.option('--CA', 'i_component', default=1)
def plot_sample_size_results(n_pcs_all, dpath_cca, i_component):

    print_params(locals())

    n_PCs_str = CcaResultsSampleSize.get_dname_PCs(n_pcs_all)
    i_component = i_component - 1 # zero-indexing
    dpath_PCs = Path(dpath_cca, n_PCs_str)

    if not dpath_PCs.exists():
        print(f'[ERROR] Directory not found: {dpath_PCs}')
        sys.exit(1)

    subsets = sorted([p.name for p in dpath_PCs.iterdir()])
    print(f'Found {len(subsets)} subsets: {[str(subset) for subset in subsets]}')

    data_for_df = []
    for subset in subsets:

        fpaths_results = [
            p 
            for p in (dpath_PCs/subset).glob('**/*') 
            if (p.is_file() and p.suffix == '.pkl')
        ]

        print(f'Found {len(fpaths_results)} result files for subset {subset}')

        for fpath_results in fpaths_results:

            results = CcaResultsSampleSize.load_fpath(fpath_results)

            # # convert to new type
            # results = CcaResultsSampleSize.load_and_cast(fpath_results)
            # results.save(verbose=False)

            for cca_type in results.method_names:
                try:
                    corr_learn = results[cca_type]['learn'].corrs[i_component]
                    corr_val = results[cca_type]['val'].corrs[i_component]
                    if 'repeated' in cca_type:
                        if corr_learn < 0:
                            corr_learn = -corr_learn
                            corr_val = -corr_val
                    data_for_df.append({
                        'sample_size': results.sample_size,
                        'i_bootstrap_repetition': results.i_bootstrap_repetition,
                        'subset': subset,
                        'cca_type': cca_type,
                        'corr_learn': corr_learn,
                        'corr_val': corr_val,
                    })
                except KeyError:
                    continue
    
    df_results = pd.DataFrame(data_for_df).groupby(['subset', 'cca_type'])
    df_results = df_results.apply(
        lambda df: df.sort_values('sample_size', kind='stable')
    ).reset_index(drop=True)
    
    fig = sns.relplot(
        data=df_results, x='corr_val', y='corr_learn',
        col='cca_type', row='subset', hue='sample_size',
        palette='flare_r',
        kind='scatter', s=100, linewidth=0,
    )

    dpath_out = Path(dpath_cca, DNAME_FIGS)
    fpath_out = Path(dpath_out, f'corrs_CA{i_component+1}_{n_PCs_str}') # without ext
    make_parent_dir(fpath_out)
    save_fig(fig, fpath_out)
    df_results.to_csv(fpath_out.with_suffix('.csv'), header=True, index=False)

if __name__ == '__main__':
    plot_sample_size_results()
