#!/usr/bin/env python
import sys
import pickle
from pathlib import Path

import click
import pandas as pd
import seaborn as sns

from paths import DPATHS

@click.command()
@click.argument('n-pcs', nargs=-1, required=True)
@click.option('--CA', 'i_axis', default=1)
def plot_sample_size_results(n_pcs, i_axis):
    n_PCs_str = '_'.join(n_pcs)
    i_axis = i_axis - 1
    dpath_PCs = Path(DPATHS['cca_sample_size'], f'PCs_{n_PCs_str}')

    if not dpath_PCs.exists():
        print(f'[ERROR] Directory not found: {dpath_PCs}')
        sys.exit(1)

    fpaths_results = [p for p in dpath_PCs.glob('**/*') if (p.is_file() and p.suffix == '.pkl')]
    print(f'Found {len(fpaths_results)} result files')

    data_for_df = []
    for fpath_results in fpaths_results:
        with fpath_results.open('rb') as file_results:
            results = pickle.load(file_results)
            for cca_type in ['cca_without_cv', 'cca_repeated_cv', 'cca_repeated_cv_no_rotate']:
                try:
                    data_for_df.append({
                        'sample_size': results['sample_size'],
                        'i_bootstrap_repetition': results['i_bootstrap_repetition'],
                        'cca_type': cca_type,
                        'corr_learn': abs(results[cca_type]['corrs']['learn'][i_axis]),
                        'corr_val': abs(results[cca_type]['corrs']['val'][i_axis]),
                    })
                except KeyError:
                    continue
    
    df_results = pd.DataFrame(data_for_df)
    
    fig = sns.relplot(
        data=df_results, x='corr_val', y='corr_learn',
        col='cca_type', hue='sample_size',
        kind='scatter', s=100,
    )

    dpath_out = Path(DPATHS['sample_size_figs'])
    dpath_out.mkdir(parents=True, exist_ok=True)
    fpath_out = Path(dpath_out, f'corrs_CA{i_axis}_{n_PCs_str}.png')
    fig.savefig(fpath_out, bbox_inches='tight')

if __name__ == '__main__':
    plot_sample_size_results()
