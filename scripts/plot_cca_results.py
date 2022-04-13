
import sys, os, pickle
import itertools
import numpy as np
from src.helpers_regression import get_lr_score_components
from src.plotting import plot_corrs, plot_lr_results_comparison, plot_scatter
from src.cca_utils import cca_score
from paths import DPATHS

np.set_printoptions(
    precision=4, 
    linewidth=100,
    suppress=True, 
    sign=' ', 
)

color_learn = '#009193'
color_test = '#ED7D31'
label_holdout = f'$\mathregular{{r_{{age}}}}$'

dpath_cv = DPATHS['cv']
extraction_method = 'mean'
n_to_print = 10

n_components_for_scatter = np.arange(3)

if __name__ == '__main__':

    if len(sys.argv) != 2:
        print(f'Usage: {sys.argv[0]} results_prefix')
        sys.exit(1)
    fname_prefix = sys.argv[1]
    fname_results = f'{fname_prefix}_central_tendencies.pkl'
    fpath_results = os.path.join(dpath_cv, fname_results)

    print('----- Parameters -----')
    print(f'fpath_results:\t{fpath_results}')
    print(f'extraction_method:\t{extraction_method}')
    print('----------------------')

    with open(fpath_results, 'rb') as file_results:
        results = pickle.load(file_results)

        n_datasets = results['n_datasets']
        dataset_names = results['dataset_names']
        n_CAs = results['n_CAs']
        n_PCs = results['n_PCs']
        n_features = results['n_features']
        feature_names = results['feature_names']
        subjects_learn = results['subjects_learn']
        subjects_test = results['subjects_test']
        CAs_learn = results['CAs_learn'][extraction_method]
        CAs_test = results['CAs_test'][extraction_method]
        PCs_learn = results['PCs_learn'][extraction_method]
        PCs_test = results['PCs_test'][extraction_method]
        deconfs_learn = results['deconfs_learn'][extraction_method]
        deconfs_test = results['deconfs_test'][extraction_method]
        holdout_learn = results['holdout_learn']
        holdout_test = results['holdout_test']

    # compute CCA correlations
    CA_corrs_learn = cca_score(CAs_learn)
    CA_corrs_test = cca_score(CAs_test)

    print('----------')
    print(f'CCA correlation results ({n_to_print} first CAs):')
    print(f'\tLearn:\t{CA_corrs_learn[:n_to_print]}')
    print(f'\tTest: \t{CA_corrs_test[:n_to_print]}')

    # plot correlations
    n_CAs = len(CA_corrs_learn)
    fig_corrs, n_CAs_plotted = plot_corrs([CA_corrs_learn, CA_corrs_test], ['Learning', 'Test'], [color_learn, color_test])
    if n_CAs_plotted != n_CAs:
        fpath_fig_corrs = os.path.join(DPATHS['cca'], f'corrs_{fname_prefix}_first{n_CAs_plotted}.png')
    else:
        fpath_fig_corrs = os.path.join(DPATHS['cca'], f'corrs_{fname_prefix}.png')
    fig_corrs.savefig(fpath_fig_corrs, dpi=300, bbox_inches='tight')
    print(f'Saved correlations figure to {fpath_fig_corrs}')

    # compute regression results
    print('----------')
    print('Linear regression results (correlation with holdout variable):')
    component_type_data_map = {
        'CAs': {'learn': CAs_learn, 'test': CAs_test},
        'PCs': {'learn': PCs_learn, 'test': PCs_test},
        'CAs_PCs': {'learn': CAs_learn + PCs_learn, 'test': CAs_test + PCs_test},
    }
    for i_dataset, dataset_name in enumerate(dataset_names):
        component_type_data_map[f'PCs_{dataset_name}'] = {
            'learn': [PCs_learn[i_dataset]],
            'test': [PCs_test[i_dataset]],
        }
    cumulative_label_map = {
        True: 'cumulative', 
        False: 'individual',
    }

    scores_all = {a: {b: {} for b in cumulative_label_map.values()} for a in component_type_data_map.keys()}
    for component_type, cumulative in itertools.product(component_type_data_map.keys(), [False, True]):
        component_data = component_type_data_map[component_type]

        scores_learn, scores_test = get_lr_score_components(
            component_data['learn'], holdout_learn, component_data['test'], holdout_test, 
            n_to_check=n_CAs, cumulative=cumulative,
        )
        cumulative_label = cumulative_label_map[cumulative]

        scores_all[component_type][cumulative_label]['learn'] = scores_learn
        scores_all[component_type][cumulative_label]['test'] = scores_test
        print(f'\t{component_type}, {cumulative_label}, test: {scores_all[component_type][cumulative_label]["test"][:n_to_print]}', end='')
        print(f'(max: {np.nanmax(scores_all[component_type][cumulative_label]["test"]):.4f})')

    # plot regression results comparison
    fig_lr, n_components_plotted = plot_lr_results_comparison(scores_all, 
        ylabel=label_holdout, fmts=['x', 'o'], colors=[color_learn, color_test])
    if n_components_plotted == n_CAs:
        fpath_fig_lr = os.path.join(DPATHS['cca'], f'lr_comparison_{fname_prefix}.png')
    else:
        fpath_fig_lr = os.path.join(DPATHS['cca'], f'lr_comparison_{fname_prefix}_first{n_components_plotted}.png')
    fig_lr.savefig(fpath_fig_lr, dpi=300, bbox_inches='tight')
    print(f'Saved linear regression figure to {fpath_fig_lr}')

    # plot scatter plot(s)
    print('----------')
    i_dataset_x = 1
    i_dataset_y = 0
    for i_component in n_components_for_scatter:

        fig_scatter = plot_scatter(
            [CAs[i_dataset_x][:, i_component] for CAs in (CAs_learn, CAs_test)], 
            [CAs[i_dataset_y][:, i_component] for CAs in (CAs_learn, CAs_test)], 
            [holdout_learn.to_numpy(), holdout_test.to_numpy()], 
            dataset_names[i_dataset_x], 
            dataset_names[i_dataset_y], 
            ax_titles=['Learning set', 'Test set'], 
            texts_upper_left=[f'{label_holdout} = {scores_all["CAs"][cumulative_label_map[False]][set_name][i_component]:.3f}' for set_name in ('learn', 'test')], 
            texts_bottom_right=[f'$\mathregular{{CA_{{{i_component+1}}} = {CA_corrs[i_component]:.3f}}}$' for CA_corrs in (CA_corrs_learn, CA_corrs_test)],
        )
        fpath_fig_scatter = os.path.join(DPATHS['cca'], f'scatter_{fname_prefix}_CA{i_component+1}.png')
        fig_scatter.savefig(fpath_fig_scatter, dpi=300, bbox_inches='tight')
        print(f'Saved scatter plot to {fpath_fig_scatter}')

