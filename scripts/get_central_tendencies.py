
import sys, os, pickle
from pathlib import Path
import matplotlib.pyplot as plt
from src.ensemble_model import EnsembleCCA
from src.utils import make_dir
from paths import DPATHS, FPATHS

rotate_CAs = True
rotate_PCs = True
ensemble_methods = ['mean', 'median']

plot_distributions = True
n_rows = 5
n_cols = 5
ax_size = 3
bins = None
labels = ['learn', 'test']

dpath_cv = DPATHS['cv']
dpath_figs = DPATHS['cv_figs']
fpath_data = FPATHS['data_Xy']

if __name__ == '__main__':

    if len(sys.argv) != 2:
        print(f'Usage: {sys.argv[0]} fname_results')
        sys.exit(1)
    fname_models = sys.argv[1]
    fpath_models = os.path.join(dpath_cv, fname_models)

    fname_prefix = []
    for fname_component in Path(fname_models).stem.split('_'):
        if fname_component == 'results':
            break
        fname_prefix.append(fname_component)
    fname_prefix = '_'.join(fname_prefix)

    print('----- Parameters -----')
    print(f'fpath_models:\t{fpath_models}')
    print(f'fpath_data:\t{fpath_data}')
    print(f'rotate_CCA:\t{rotate_CAs}')
    print(f'rotate_PCs:\t{rotate_PCs}')
    print('----------------------')

    # load results file
    with open(fpath_models, 'rb') as file_results:
        results = pickle.load(file_results)

        models = results['models_combined']
        i_split = results['i_split']
        n_datasets = results['n_datasets']
        dataset_names = results['dataset_names']

    # load learning/test data
    with open(fpath_data, 'rb') as file_data:
        data = pickle.load(file_data)

        X = data['X']
        holdout = data['holdout']
        subjects = data['subjects']
        i_learn = data['i_train_all'][i_split]
        i_test = data['i_test_all'][i_split]
        subjects_learn = subjects[i_learn]
        subjects_test = subjects[i_test]
        X_learn = X.loc[subjects_learn]
        X_test = X.loc[subjects_test]
        holdout_learn = holdout.loc[subjects_learn]
        holdout_test = holdout.loc[subjects_test]
        print(f'X_learn: {X_learn.shape}')
        print(f'X_test: {X_test.shape}')

    # run models for CCA
    ensemble_cca = EnsembleCCA(models, rotate=rotate_CAs)
    CAs_learn_all = ensemble_cca.fit_transform(X_learn, apply_ensemble_method=False)
    CAs_test_all = ensemble_cca.transform(X_test, apply_ensemble_method=False)

    # run models for PCA only
    ensemble_pca = EnsembleCCA(models, rotate=rotate_PCs)
    PCs_learn_all = ensemble_pca.fit_transform(X_learn, key='preprocessor', apply_ensemble_method=False)
    PCs_test_all = ensemble_pca.transform(X_test, key='preprocessor', apply_ensemble_method=False)

    # print shapes
    for data_label, data in {'CAs_learn_all':CAs_learn_all, 'CAs_test_all':CAs_test_all, 'PCs_learn_all':PCs_learn_all, 'PCs_test_all':PCs_test_all}.items():
        print(f'{data_label}: {[d.shape for d in data]}')

    # extract central tendencies
    CAs_learn = {}
    CAs_test = {}
    PCs_learn = {}
    PCs_test = {}
    for ensemble_method in ensemble_methods:
        CAs_learn[ensemble_method] = ensemble_cca.apply_ensemble_method(CAs_learn_all, ensemble_method=ensemble_method)
        CAs_test[ensemble_method] = ensemble_cca.apply_ensemble_method(CAs_test_all, ensemble_method=ensemble_method)
        PCs_learn[ensemble_method] = ensemble_pca.apply_ensemble_method(PCs_learn_all, ensemble_method=ensemble_method)
        PCs_learn[ensemble_method] = ensemble_pca.apply_ensemble_method(PCs_test_all, ensemble_method=ensemble_method)

    # optional plotting
    if plot_distributions:
        make_dir(dpath_figs)
        for i_dataset, dataset_name in enumerate(dataset_names):

            data_all = {
                'CAs': [CAs_learn_all[i_dataset], CAs_test_all[i_dataset]],
                'PCs': [PCs_learn_all[i_dataset], PCs_test_all[i_dataset]],
            }

            for component_type in data_all.keys():
                fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(n_cols*ax_size, n_rows*ax_size))

                for i_row in range(n_rows):
                    for i_col in range(n_cols):
                        ax = axes[i_row][i_col]
                        for i_data, data in enumerate(data_all[component_type]):
                            ax_data = data[:, i_row, i_col]
                            ax.hist(ax_data, bins=bins, alpha=0.5, label=labels[i_data])
                        ax.legend()
                fig.tight_layout()

                # generate figure file name
                fpath_fig = os.path.join(dpath_figs, f'{fname_prefix}_{component_type}_{dataset_name}.png')
                fig.savefig(fpath_fig, dpi=300, bbox_inches='tight')
                print(f'Saved figure to {fpath_fig}')

    to_save = {
        'i_split': i_split,
        'n_datasets': n_datasets,
        'subjects_learn': subjects_learn,
        'subjects_test': subjects_test,
        'CAs_learn': CAs_learn,
        'CAs_test': CAs_test,
        'PCs_learn': PCs_learn,
        'PCs_test': PCs_test,
        # 'CAs_learn_all': CAs_learn_all, 
        # 'CAs_test_all': CAs_test_all,
        # 'PCs_learn_all': PCs_learn_all,
        # 'PCs_test_all': PCs_test_all,
        'holdout_learn': holdout_learn,
        'holdout_test': holdout_test,
    }
    fpath_out = os.path.join(dpath_cv, f'{fname_prefix}_central_tendencies.pkl')
    with open(fpath_out, 'wb') as file_out:
        pickle.dump(to_save, file_out)
        print('-----------')
        print(f'Saved data to {fpath_out}')
