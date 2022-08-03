import pickle
import os
import numpy as np
from paths import FPATHS, DPATHS

n_bootstrap_repetitions = 100
n_sample_sizes = 20
val_sample_fraction = 0.5
n_folds = 5 # for CV
seed = 3791

fpath_data = FPATHS['data_Xy']

fname_out = f'bootstrap_samples_{n_sample_sizes}steps_{n_bootstrap_repetitions}times.pkl'
fpath_out = os.path.join(DPATHS['clean'], fname_out)

if __name__ == '__main__':

    np.set_printoptions(precision=4, suppress=True, linewidth=100, sign=' ')

    print('----- Parameters -----')
    print(f'fpath_data:\t{fpath_data}')
    print(f'n_bootstrap_repetitions:\t{n_bootstrap_repetitions}')
    print(f'n_sample_sizes:\t{n_sample_sizes}')
    print(f'val_sample_fraction:\t{val_sample_fraction}')
    print(f'seed:\t{seed}')
    print('----------------------')

    # random number generator
    rng = np.random.default_rng(seed)

    with open(fpath_data, 'rb') as file_data:
        data = pickle.load(file_data)
        subjects = data['subjects']
        n_features_datasets = data['n_features_datasets']

    # bounds for sample size
    n_subjects = len(subjects)
    sample_size_min = np.ceil(np.min(n_features_datasets) / (1 - (1/n_folds)))
    sample_size_max = int(val_sample_fraction * n_subjects)

    # log-spaced sample sizes
    sample_sizes = np.geomspace(sample_size_min, sample_size_max, n_sample_sizes, dtype=np.int64)
    print(f'Sample sizes: {sample_sizes}')

    i_samples = np.arange(n_subjects) # to be sampled from
    i_samples_learn_all = [] # list of dicts
    i_samples_val_all = [] # list only (val samples are same across sample sizes for each in/out split)

    for i_bootstrap_repetition in range(n_bootstrap_repetitions):

        # split dataset into in/out-sample sets
        i_samples_in = rng.choice(i_samples, size=sample_size_max, replace=False)
        i_samples_val = np.array(list(set(i_samples) - set(i_samples_in)))

        i_samples_val_all.append(i_samples_val)
        i_samples_learn_all.append({})

        for sample_size in sample_sizes:

            # sample without replacement
            i_samples_learn = rng.choice(i_samples_in, size=sample_size, replace=False)

            # append
            i_samples_learn_all[i_bootstrap_repetition][sample_size] = i_samples_learn

    results = {
        'sample_sizes': sample_sizes,
        'i_samples_learn_all': i_samples_learn_all,
        'i_samples_val_all': i_samples_val_all,
        'n_bootstrap_repetitions': n_bootstrap_repetitions,
        'n_sample_sizes': n_sample_sizes,
        'val_sample_fraction': val_sample_fraction,
        'seed': seed,
    }

    with open(fpath_out, 'wb') as file_out:
        pickle.dump(results, file_out)

    print(f'Saved bootstrapped sample indices to {fpath_out}')
