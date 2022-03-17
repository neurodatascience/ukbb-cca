
import sys, os, glob, pickle
import numpy as np
import matplotlib.pyplot as plt
from paths import DPATHS

flip_sign = True

save_extracted = True # if True, saved only summary (e.g., mean/median) measures instead of everything
plot_sample_distributions = True
n_to_plot = (5, 5)
ax_size = 3

# folder containing CV results (different parameters/runs)
dpath_cv = dpath_cv = os.path.join(DPATHS['scratch'], os.path.basename(DPATHS['cca'])) 
cv_filename_pattern = '*rep*.pkl'

dpath_out = DPATHS['cv'] # folder for combined results

extraction_methods = {
    'mean': (lambda x: np.mean(x, axis=0)),
    'median': (lambda x: np.median(x, axis=0)),
    'std': (lambda x: np.std(x, axis=0)),
}

def flip(V_combined):

    # collapse first two dimensions and find largest (absolute) value
    V_combined_reshaped = V_combined.reshape(V_combined.shape[0]*V_combined.shape[1], -1)
    idx_max_abs = np.argmax(np.abs(V_combined_reshaped), axis=0)
    max_abs = V_combined_reshaped[idx_max_abs, range(V_combined_reshaped.shape[1])]

    # find rows in original matrix that correspond to the max values
    _, idx_row, idx_col = np.nonzero(V_combined == max_abs)
    idx_sort = np.argsort(idx_col)
    idx_row = idx_row[idx_sort]
    idx_col = idx_col[idx_sort]

    # determine which repetitions and which columns need to be flipped
    signs = np.where(np.sign(max_abs) == np.sign(V_combined[:, idx_row, idx_col]), 1, -1)
    V_combined_flipped = V_combined * signs[:, np.newaxis, :]

    return V_combined_flipped

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

            subjects = results['subjects']

            dataset_names = results['dataset_names']
            n_datasets = results['n_datasets']

            latent_dims_names = results['latent_dims_names']
            n_latent_dims = results['n_latent_dims']

            PC_names = results['PC_names']
            n_components_all = results['n_components_all']

            udis = results['udis_datasets']
            n_folds = results['n_folds']

            # initialization
            projections_combined = [[] for _ in range(n_datasets)]
            loadings_combined = [[] for _ in range(n_datasets)]
            weights_combined = [[] for _ in range(n_datasets)]
            correlations_val_combined = []
            R2_PC_reg_val_combined = []

        correlations_val = results['correlations_val']
        R2_PC_reg_val = results['R2_PC_reg_val']

        projections_val = results['projections_val']
        loadings_val = results['loadings_val']
        weights_train = results['weights_train']
        
        for i_dataset in range(n_datasets):
            projections_combined[i_dataset].append(projections_val[i_dataset])
            loadings_combined[i_dataset].append(loadings_val[i_dataset])
            weights_combined[i_dataset].append(weights_train[i_dataset])

        correlations_val_combined.append(correlations_val)
        R2_PC_reg_val_combined.append(R2_PC_reg_val)

    projections_extracted = {key: [] for key in extraction_methods.keys()}
    loadings_extracted = {key: [] for key in extraction_methods.keys()}
    weights_extracted = {key: [] for key in extraction_methods.keys()}
    for i_dataset in range(n_datasets):

        # convert to numpy array
        projections_combined[i_dataset] = np.array(projections_combined[i_dataset])
        loadings_combined[i_dataset] = np.array(loadings_combined[i_dataset])
        weights_combined[i_dataset] = np.array(weights_combined[i_dataset]).reshape(
            (-1, n_components_all[i_dataset], n_latent_dims),
        )

        if flip_sign:
            weights_combined[i_dataset] = flip(weights_combined[i_dataset])

            # doesn't work for projections (still get bimodal distributions)
            # projections because it is the combination of 5 folds, and weights within folds don't have consistent signs
            projections_combined[i_dataset] = flip(projections_combined[i_dataset])

            # loadings don't have bimodal distributions, but the distributions have multiple peaks
            # so flipping probably doesn't make sense
            loadings_combined[i_dataset] = flip(loadings_combined[i_dataset])

        for method, fc_extraction in extraction_methods.items():
            projections_extracted[method].append(fc_extraction(projections_combined[i_dataset]))
            loadings_extracted[method].append(fc_extraction(loadings_combined[i_dataset]))
            weights_extracted[method].append(fc_extraction(weights_combined[i_dataset]))

    # common measures
    results_to_save = {
        'correlations_val_combined': correlations_val_combined,
        'R2_PC_reg_val_combined': R2_PC_reg_val_combined,
        'dataset_names': dataset_names,
        'n_datasets': n_datasets,
        'subjects': subjects,
        'latent_dims_names': latent_dims_names,
        'n_latent_dims': n_latent_dims,
        'PC_names': PC_names,
        'n_components_all': n_components_all,
        'udis': udis,
        'n_folds': n_folds,
        'n_reps': len(fnames),
    }

    if save_extracted:
        results_to_save.update({
            'projections_extracted': projections_extracted,
            'loadings_extracted': loadings_extracted,
            'weights_extracted': weights_extracted,
        })
        out_suffix = 'extracted'

    else:
        results_to_save.update({
            'projections_combined': projections_combined,
            'loadings_combined': loadings_combined,
            'weights_combined': weights_combined,
        })
        out_suffix = 'combined'

    if flip_sign:
        out_suffix = f'{out_suffix}_flipped'

    fpath_out = os.path.join(dpath_out, f'{dname_reps}_results_{out_suffix}.pkl')
    with open(fpath_out, 'wb') as file_out:
        pickle.dump(results_to_save, file_out)
    print(f'Saved to {fpath_out}')

    # optional plotting
    if plot_sample_distributions:

        dpath_figs = os.path.join(dpath_out, 'figs')
        n_rows, n_cols = n_to_plot
        
        to_plot = {}
        for label, data in {'projections': projections_combined, 'loadings': loadings_combined, 'weights': weights_combined}.items():
            to_plot.update({f'{label}_{dataset_name}': data[i_dataset][:, :n_rows, :n_cols] for i_dataset, dataset_name in enumerate(dataset_names)})
        
        if not flip_sign:
            fname_plot_data = f'plot_data_{dname_reps}.pkl'
        else:
            fname_plot_data = f'plot_data_{dname_reps}_flipped.pkl'
        with open(os.path.join(dpath_figs, fname_plot_data), 'wb') as file_out:
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

            fpath_fig = os.path.join(dpath_figs, f'{label}_{dname_reps}.png')
            fig.savefig(fpath_fig, dpi=300, bbox_inches='tight')
            print(f'Saved figure to {fpath_fig}')
