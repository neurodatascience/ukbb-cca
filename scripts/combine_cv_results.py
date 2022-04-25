
import sys, os, glob, pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.base import clone, BaseEstimator
from cca_zoo.models._cca_base import _CCA_Base
from src.utils import make_dir

from paths import DPATHS

save_extracted = False # if True, saved only averaged model instead of all individual models
out_suffix = '' # may be updated later

plot_sample_distributions = False #True
n_to_plot = (5, 5)
ax_size = 3

# folder containing CV results (different parameters/runs)
dpath_cv = dpath_cv = os.path.join(DPATHS['scratch'], os.path.basename(DPATHS['cca'])) 
cv_filename_pattern = '*rep*.pkl'

dpath_out = DPATHS['cv'] # folder for combined results

def average_models(models):

    cca_param_names = ['weights', 'view_means', 'view_stds']

    def find_nested_parameters(root_model, start_keys=[], keys_and_names=[]):
        model = root_model
        for key in start_keys:
            model = model[key]
        fitted_parameter_names = [name for name in vars(model) if name.endswith('_') and not name.startswith('__')]
        if len(fitted_parameter_names) > 0:
            for name in fitted_parameter_names:
                keys_and_names.append((start_keys, name))
        elif isinstance(model, _CCA_Base):
            for param_name in cca_param_names:
                if hasattr(model, param_name):
                    keys_and_names.append((start_keys, param_name))
        else:
            if hasattr(model, 'named_steps'): # sklearn Pipeline
                nested_models = model.named_steps
            elif hasattr(model, 'named_pipelines'): # PreprocessingPipeline, PipelineList
                nested_models = model.named_pipelines
            else:
                raise ValueError(f'Unhandled model: {model}')
            for key, nested_model in nested_models.items():
                keys_and_names = find_nested_parameters(root_model, start_keys=start_keys+[key], keys_and_names=keys_and_names)
        return keys_and_names

    def average_nested_parameter(models, keys, parameter_name):
        all_values = []
        for root_model in models:
            model = root_model
            for key in keys:
                model = model[key]
            all_values.append(getattr(model, parameter_name))
        if hasattr(all_values[0], 'dtype'):
            if np.issubdtype(all_values[0].dtype, np.number):
                # print(f'{"__".join(keys)} --> {parameter_name}')
                first = all_values[0]
                for val in all_values:
                    if val.shape != first.shape:
                        print(f'{"__".join(keys)} --> {parameter_name}')
                        break
                # TODO also return std
                return np.mean(all_values, axis=0)
        elif hasattr(all_values[0], '__len__') and hasattr(all_values[0][0], 'dtype'):
            if np.issubdtype(all_values[0][0].dtype, np.number):
                # TODO also return std
                # all_values_reshaped = []
                return [np.mean([all_values[i][j] for i in range(len(all_values))], axis=0) for j in range(len(all_values[0]))]
        return None

    def set_nested_parameter(root_model, keys, name, value):
        model = root_model
        for key in keys:
            model = model[key]
        setattr(model, name, value)

    # clone the model/pipeline
    fitted_model = models[0]
    model_clone = clone(fitted_model)

    # get all fitted values
    for keys, parameter_name in find_nested_parameters(fitted_model):
        # attempt to average them
        avg_value = average_nested_parameter(models, keys, parameter_name)
        # add averaged value to cloned model
        if avg_value is not None:
            set_nested_parameter(model_clone, keys, parameter_name, avg_value)

    return model_clone

if __name__ == '__main__':

    # get path to CV directory from command line
    if len(sys.argv) != 2:
        print(f'Usage: {sys.argv[0]} <dname_cv>')
        sys.exit(1)
    dname_reps = sys.argv[1]
    dpath_reps = os.path.join(dpath_cv, dname_reps)

    # get all valid files in directory containing CV results for each repetition
    fnames = glob.glob(os.path.join(dpath_reps, cv_filename_pattern))
    fnames.sort()

    print(f'Combining {len(fnames)} files')

    # for each rep
    for i_rep, fname in enumerate(fnames):

        # load the file
        fpath_results = os.path.join(dpath_reps, fname)
        with open(fpath_results, 'rb') as file_results:
            results = pickle.load(file_results)

        if i_rep == 0:

            i_split = results['i_split']
            n_folds = results['n_folds']

            subjects = results['subjects']

            dataset_names = results['dataset_names']
            n_datasets = results['n_datasets']
            conf_name = results['conf_name']

            latent_dims_names = results['latent_dims_names']
            n_latent_dims = results['n_latent_dims']

            PC_names = results['PC_names']
            n_components_all = results['n_components_all']

            udis_datasets = results['udis_datasets']
            udis_conf = results['udis_conf']

            n_features_datasets = results['n_features_datasets']
            n_features_conf = results['n_features_conf']

            # initialization
            models_combined = []
            correlations_val_combined = []
            i_train_combined = []
            i_val_combined = []

        models = results['models']
        correlations_val = results['correlations_val']
        i_train = results['i_train']
        i_val = results['i_val']
        
        models_combined.extend(models)
        correlations_val_combined.append(correlations_val)
        i_train_combined.append(i_train)
        i_val_combined.append(i_val)


    extracted_model = average_models(models_combined)

    # common measures
    results_to_save = {
        'model': extracted_model,
        'correlations_val_combined': np.array(correlations_val_combined),
        'i_split': i_split,
        'dataset_names': dataset_names,
        'n_datasets': n_datasets,
        'conf_name': conf_name,
        'subjects': subjects,
        'latent_dims_names': latent_dims_names,
        'n_latent_dims': n_latent_dims,
        'PC_names': PC_names,
        'n_components_all': n_components_all,
        'udis_datasets': udis_datasets,
        'udis_conf': udis_conf,
        'n_features_datasets': n_features_datasets,
        'n_features_conf': n_features_conf,
        'n_folds': n_folds,
        'n_reps': len(fnames),
    }

    if not save_extracted:
        results_to_save.update({
            'models_combined': models_combined,
            'i_train_combined': i_train_combined,
            'i_val_combined': i_val_combined,
        })
        out_suffix = '_combined'

    make_dir(dpath_out)
    fpath_out = os.path.join(dpath_out, f'{dname_reps}_results{out_suffix}.pkl')
    with open(fpath_out, 'wb') as file_out:
        pickle.dump(results_to_save, file_out)
    print(f'Saved to {fpath_out}')

    # optional plotting
    if plot_sample_distributions:

        dpath_figs = os.path.join(dpath_out, 'figs')
        dpath_fig_data = os.path.join(dpath_figs, 'fig_data')
        make_dir(dpath_figs)
        make_dir(dpath_fig_data)
        n_rows, n_cols = n_to_plot
        
        to_plot = {}
        for label, data in {'projections': projections_combined, 'loadings': loadings_combined, 'weights': weights_combined}.items():
            to_plot.update({f'{label}_{dataset_name}': data[i_dataset][:, :n_rows, :n_cols] for i_dataset, dataset_name in enumerate(dataset_names)})
        
        fname_plot_data = f'plot_data_{dname_reps}.pkl'
        with open(os.path.join(dpath_fig_data, fname_plot_data), 'wb') as file_out:
            pickle.dump(to_plot, file_out)

        for label, data in to_plot.items():
            fig, axes = plt.subplots(
                nrows=n_rows, ncols=n_cols, 
                figsize=(n_cols*ax_size, n_rows*ax_size),
                sharey='all',
            )
            if n_rows == 1:
                axes = [axes]
            if n_cols == 1:
                axes = [axes]

            for i_row in range(n_rows):
                for i_col in range(n_cols):
                    ax = axes[i_row][i_col]
                    ax.hist(data[:, i_row, i_col], bins=25)
            
            fig.tight_layout()

            fname_fig = f'{label}_{dname_reps}.png'
            fpath_fig = os.path.join(dpath_figs, fname_fig)
            fig.savefig(fpath_fig, dpi=300, bbox_inches='tight')
            print(f'Saved figure to {fpath_fig}')
