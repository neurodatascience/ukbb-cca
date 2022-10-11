
import sys, os, pickle
import numpy as np
import matplotlib.pyplot as plt
from src.ensemble_model import EnsembleCCA, apply_ensemble_method
from src.cca_utils import cca_score, cca_get_loadings
from src.utils import make_dir
from paths import DPATHS, FPATHS

# import logging # TODO remove
# logging.basicConfig(
#     filename='get_central_tendencies.log', 
#     format='[%(asctime)s] %(message)s',
#     encoding='utf-8', 
#     level=logging.DEBUG,
# )

bootstrap_mode = False # only save correlations and loadings

compute_PCs = True
compute_loadings = True # if True, gets deconfounded Learning/Test data and gets variable loadings
normalize_loadings = True

# override
if bootstrap_mode:
    compute_PCs = False
    compute_loadings = True # should be True 

rotate_CAs = True
rotate_PCs = True
rotate_deconfs = False
ensemble_methods = ['nanmean', 'nanmedian'] # deconfs may have missing data (CAs/PCs do not)

model_transform_cca = None
model_transform_pca = (lambda model: model['preprocessor'])
model_transform_deconfounder = (lambda model: 
    model['preprocessor'].set_params(
        data_pipelines__behavioural__pca='passthrough', 
        data_pipelines__brain__pca='passthrough'
    )
)

plot_distributions = False
n_rows = 5
n_cols = 5
ax_size = 3
bins = None
labels = ['learn', 'test']

dpath_cv = DPATHS['cv']
dpath_out = DPATHS['central_tendencies']
dpath_figs = DPATHS['cv_figs']
fpath_data = FPATHS['data_Xy']

if __name__ == '__main__':

    # logging.info('Main started')

    if len(sys.argv) != 2:
        print(f'Usage: {sys.argv[0]} results_prefix')
        sys.exit(1)
    fname_prefix = sys.argv[1]
    fname_models = f'{fname_prefix}_results_combined.pkl'
    fpath_models = os.path.join(dpath_cv, fname_models)

    print('----- Parameters -----')
    print(f'fpath_models:\t{fpath_models}')
    print(f'fpath_data:\t{fpath_data}')
    print(f'rotate_CCA:\t{rotate_CAs}')
    print(f'rotate_PCs:\t{rotate_PCs}')
    print(f'rotate_deconfs:\t{rotate_deconfs}')
    print(f'ensemble_methods:\t{ensemble_methods}')
    print('----------------------')

    # create output directory if necessary
    make_dir(dpath_out)

    # load results file
    with open(fpath_models, 'rb') as file_results:
        results = pickle.load(file_results)

        models = results['models_combined']
        n_models = len(models)
        i_split = results['i_split']
        n_datasets = results['n_datasets']
        dataset_names = results['dataset_names']
        n_CAs = results['n_latent_dims']
        n_PCs = results['n_components_all']

    # load learning/test data
    with open(fpath_data, 'rb') as file_data:
        data = pickle.load(file_data)

        X = data['X']
        holdout = data['holdout']
        subjects = data['subjects']
        i_learn = data['i_train_all'][i_split]
        i_test = data['i_test_all'][i_split]
        n_features = data['n_features_datasets']
        feature_names = data['udis_datasets']
        subjects_learn = subjects[i_learn]
        subjects_test = subjects[i_test]
        X_learn = X.loc[subjects_learn]
        X_test = X.loc[subjects_test]
        holdout_learn = holdout.loc[subjects_learn]
        holdout_test = holdout.loc[subjects_test]
        print(f'X_learn: {X_learn.shape}')
        print(f'X_test: {X_test.shape}')

    # logging.info('Data loaded')

    # run models for CCA
    ensemble_model = EnsembleCCA(models, rotate=rotate_CAs, model_transform=model_transform_cca)
    CAs_learn_all = ensemble_model.fit_transform(X_learn)
    CAs_test_all = ensemble_model.transform(X_test)

    # logging.info('Ensemble CCA done')

    # run models for PCA only
    if compute_PCs:
        ensemble_model = EnsembleCCA(models, rotate=rotate_PCs, model_transform=model_transform_pca)
        PCs_learn_all = ensemble_model.fit_transform(X_learn)
        PCs_test_all = ensemble_model.transform(X_test)
    else:
        PCs_learn_all = []
        PCs_test_all = []

    # logging.info('Ensemble PCA done (if run)')

    # run models for preprocessor but only up to deconfounder (no PCA/CCA)
    if compute_loadings:
        ensemble_model = EnsembleCCA(models, rotate=rotate_deconfs, model_transform=model_transform_deconfounder)
        deconfs_learn_all = ensemble_model.fit_transform(X_learn)
        deconfs_test_all = ensemble_model.transform(X_test)
    else:
        deconfs_learn_all = []
        deconfs_test_all = []

    # logging.info('Ensemble deconf done (if run)')

    # print shapes
    to_print = {
        'CAs_learn_all': CAs_learn_all, 
        'CAs_test_all': CAs_test_all, 
        'PCs_learn_all': PCs_learn_all, 
        'PCs_test_all': PCs_test_all,
        'deconfs_learn_all': deconfs_learn_all, 
        'deconfs_test_all': deconfs_test_all,
    }
    for data_label, data in to_print.items():
        if len(data) != 0:
            print(f'{data_label}: {[d.shape for d in data]}')

    # extract central tendencies
    CAs_learn = {}
    CAs_test = {}
    PCs_learn = {}
    PCs_test = {}
    deconfs_learn = {}
    deconfs_test = {}
    for component_dict, data_all in zip(
        [CAs_learn, CAs_test, PCs_learn, PCs_test, deconfs_learn, deconfs_test], 
        [CAs_learn_all, CAs_test_all, PCs_learn_all, PCs_test_all, deconfs_learn_all, deconfs_test_all]
    ):
        if len(data_all) != 0:
            for ensemble_method in ensemble_methods:
                component_dict[ensemble_method] = apply_ensemble_method(data_all, ensemble_method=ensemble_method)
            # print([x.shape for x in component_dict[ensemble_method]])

    # logging.info('Central tendencies extracted')

    loadings_learn = {}
    loadings_test = {}
    if compute_loadings:
        for ensemble_method in ensemble_methods:
            loadings_learn[ensemble_method] = cca_get_loadings(deconfs_learn[ensemble_method], CAs_learn[ensemble_method], normalize=normalize_loadings)
            loadings_test[ensemble_method] = cca_get_loadings(deconfs_test[ensemble_method], CAs_test[ensemble_method], normalize=normalize_loadings)

    # logging.info('Loadings done (if run)')

    # optional plotting
    if plot_distributions:
        make_dir(dpath_figs)
        for i_dataset, dataset_name in enumerate(dataset_names):

            data_to_plot = {
                'CAs': [CAs_learn_all, CAs_test_all],
                'PCs': [PCs_learn_all, PCs_test_all],
                'deconfs': [deconfs_learn_all, deconfs_test_all],
            }
            
            # data_all = {
            #     'CAs': [CAs_learn_all[i_dataset], CAs_test_all[i_dataset]],
            #     'PCs': [PCs_learn_all[i_dataset], PCs_test_all[i_dataset]],
            #     'deconfs': [deconfs_learn_all[i_dataset], deconfs_test_all[i_dataset]],
            # }

            for component_type in data_to_plot.keys():
                for i_data, data in enumerate(data_to_plot[component_type]):
                    # skip if data is empty
                    try:
                        data = data[i_dataset]
                    except IndexError:
                        continue
                    fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(n_cols*ax_size, n_rows*ax_size), sharey='all')
                    for i_row in range(n_rows):
                        for i_col in range(n_cols):
                            ax_data = data[:, i_row, i_col]
                            ax_data = ax_data[~np.isnan(ax_data)] # remove NAs, if any
                            ax = axes[i_row][i_col]
                            ax.hist(ax_data, bins=bins, alpha=0.5)
                    fig.tight_layout()

                    # generate figure file name
                    fpath_fig = os.path.join(dpath_figs, f'{fname_prefix}_{component_type}_{dataset_name}_{labels[i_data]}.png')
                    fig.savefig(fpath_fig, dpi=300, bbox_inches='tight')
                    print(f'Saved figure to {fpath_fig}')

    # logging.info('Plotting done (if run)')

    if not bootstrap_mode:
        to_save = {
            'i_split': i_split,
            'n_models': n_models,
            'n_datasets': n_datasets,
            'dataset_names': dataset_names,
            'n_CAs': n_CAs,
            'n_PCs': n_PCs,
            'n_features': n_features,
            'feature_names': feature_names,
            'subjects_learn': subjects_learn,
            'subjects_test': subjects_test,
            'CAs_learn': CAs_learn,
            'CAs_test': CAs_test,
            'PCs_learn': PCs_learn,
            'PCs_test': PCs_test,
            'deconfs_learn': deconfs_learn,
            'deconfs_test': deconfs_test,
            'loadings_learn': loadings_learn,
            'loadings_test': loadings_test,
            # 'CAs_learn_all': CAs_learn_all, 
            # 'CAs_test_all': CAs_test_all,
            # 'PCs_learn_all': PCs_learn_all,
            # 'PCs_test_all': PCs_test_all,
            # 'deconfs_learn_all': deconfs_learn_all,
            # 'deconfs_test_all': deconfs_test_all,
            'holdout_learn': holdout_learn,
            'holdout_test': holdout_test,
        }
    else:
        to_save = {
            'i_split': i_split,
            'n_models': n_models,
            'n_datasets': n_datasets,
            'dataset_names': dataset_names,
            'n_CAs': n_CAs,
            'n_PCs': n_PCs,
            'n_features': n_features,
            'feature_names': feature_names,
            'loadings_learn': loadings_learn,
            'loadings_test': loadings_test,
            'corrs_learn': {method: cca_score(CAs_learn[method]) for method in ensemble_methods},
            'corrs_test': {method: cca_score(CAs_test[method]) for method in ensemble_methods},
        }

    fpath_out = os.path.join(dpath_out, f'{fname_prefix}_central_tendencies.pkl')
    with open(fpath_out, 'wb') as file_out:
        pickle.dump(to_save, file_out)
        print('-----------')
        print(f'Saved data to {fpath_out}')

    # logging.info('All done')
