
import sys, os, glob, pickle
import numpy as np
from paths import DPATHS

dpath_cv = DPATHS['cv'] # folder containing all CV results (different parameters/runs)
cv_filename_pattern = '*rep*.pkl'

if __name__ == '__main__':

    # get path to CV directory from command line
    if len(sys.argv) != 2:
        raise ValueError(f'Usage: {sys.argv[0]} <dname_cv>')
    dname_reps = sys.argv[1]
    dpath_reps = os.path.join(dpath_cv, dname_reps)

    # get all valid files in directory containing CV results for each repetition
    fnames = glob.glob(os.path.join(dpath_reps, cv_filename_pattern))
    fnames.sort()

    # for each rep
    fold_results_all = []
    for i_rep, fname in enumerate(fnames):

        # load the file
        fpath_results = os.path.join(dpath_reps, fname)
        with open(fpath_results, 'rb') as file_results:
            results = pickle.load(file_results)

        dfs_projections = results['dfs_projections']
        dfs_loadings = results['dfs_loadings']
        fold_results = results['cv_results']

        fold_results_all.append(fold_results)

        if i_rep == 0:
            dataset_names = results['dataset_names']
            n_datasets = results['n_datasets']
            subjects = results['subjects']
            latent_dims_names = results['latent_dims_names']
            udis = results['udis_datasets']

            # initialization
            projections_combined = [[] for _ in range(n_datasets)]
            loadings_combined = [[] for _ in range(n_datasets)]

        for i_dataset in range(n_datasets):
            projections_combined[i_dataset].append(dfs_projections[i_dataset].loc[subjects, latent_dims_names].values)
            loadings_combined[i_dataset].append(dfs_loadings[i_dataset].loc[udis[i_dataset], latent_dims_names].values)

    projections_median = []
    loadings_median = []
    for i_dataset in range(n_datasets):

        # convert to numpy array
        projections_combined[i_dataset] = np.stack(projections_combined[i_dataset], axis=0)
        loadings_combined[i_dataset] = np.stack(loadings_combined[i_dataset], axis=0)

        projections_median.append(np.median(projections_combined[i_dataset], axis=0))
        loadings_median.append(np.median(loadings_combined[i_dataset], axis=0))

    results_combined = {
        'projections_combined': projections_combined,
        'loadings_combined': loadings_combined,
        'projections_median': projections_median,
        'loadings_median': loadings_median,
        'dataset_names': dataset_names,
        'n_datasets': n_datasets,
        'subjects': subjects,
        'latent_dims_names': latent_dims_names,
        'udis': udis,
    }

    fpath_out = os.path.join(dpath_cv, f'{dname_reps}_all_results.pkl')
    with open(fpath_out, 'wb') as file_out:
        pickle.dump(results_combined, file_out)
