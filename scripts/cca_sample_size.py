#!/usr/bin/env python
import warnings
import click
from src.pipeline_definitions import build_cca_pipeline
from src.cca import CcaAnalysis, CcaResultsPipelines
from src.data_processing import XyData
from src.sampling import BootstrapSamples
from src.utils import print_params

SUPPRESS_WARNINGS = True
DEBUG = False

# CCA settings
NORMALIZE_LOADINGS = True

# for repeated CV
CV_N_REPETITIONS = 1
CV_N_FOLDS = 5
CV_SEED = 3791
CV_SHUFFLE = True

@click.command()
@click.argument('n_sample_sizes', type=int)
@click.argument('n_bootstrap_repetitions', type=int)
@click.argument('i_sample_size', type=int)
@click.argument('i_bootstrap_repetition', type=int)
@click.argument('n_PCs_all', type=int, nargs=-1) # if 0, will be set to number of features
@click.option('--dpath-data', required=True, envvar='DPATH_PROCESSED')
@click.option('--dpath-cca', default='.', envvar='DPATH_CCA_SAMPLE_SIZE') # TODO
@click.option('--normalize-loadings', default=NORMALIZE_LOADINGS)
@click.option('--cv-n-repetitions', default=CV_N_REPETITIONS)
@click.option('--cv-n-folds', default=CV_N_FOLDS)
@click.option('--cv-seed', default=CV_SEED)
@click.option('--cv-shuffle', default=CV_SHUFFLE)
@click.option('--suppress-warnings/--with-warnings', default=SUPPRESS_WARNINGS)
@click.option('--debug/--no-debug', default=DEBUG)
@click.option('--verbose/--quiet', default=True)
def cca_sample_size(n_sample_sizes, n_bootstrap_repetitions,
    i_sample_size, i_bootstrap_repetition, n_pcs_all, dpath_data,
    dpath_cca, normalize_loadings, cv_n_repetitions, cv_n_folds,
    cv_seed, cv_shuffle, suppress_warnings, debug, verbose,
):

    print_params(locals())

    # load data
    data = XyData(dpath_data).load()
    n_datasets = len(data.dataset_names)

    # load bootstrap sample
    bootstrap_samples = BootstrapSamples(
        dpath=dpath_data,
        n_bootstrap_repetitions=n_bootstrap_repetitions,
        n_sample_sizes=n_sample_sizes,
    ).load()

    # get learn/val indices
    sample_size = bootstrap_samples.sample_sizes[i_sample_size-1] # zero-indexing
    i_learn = bootstrap_samples.i_samples_learn_all[i_bootstrap_repetition-1][sample_size]
    i_val = bootstrap_samples.i_samples_val_all[i_bootstrap_repetition-1]
    print(f'Sample size: {sample_size}')

    # check/process number of PCs/CAs
    if len(n_pcs_all) != n_datasets:
        raise ValueError(f'Mismatch between n_PCs_all (size {len(n_pcs_all)}) and data ({n_datasets} datasets)')
    for i_dataset in range(len(data.dataset_names)):
        if n_pcs_all[i_dataset] == 0:
            n_pcs_all[i_dataset] = data.n_features_datasets[i_dataset]
    n_CAs = min(n_pcs_all)
    print(f'Using {n_CAs} latent dimensions')

    # build pipeline/model
    cca_pipeline = build_cca_pipeline(
        dataset_names=data.dataset_names,
        conf_name=data.conf_name,
        n_PCs_all=n_pcs_all,
        n_CAs=n_CAs,
        verbosity=1,
    )
    print('------------------------------------------------------------------')
    print(cca_pipeline)
    print('------------------------------------------------------------------')

    if suppress_warnings:
        # this warnings occurs when n_samples < 1000 (default n_quantiles in sklearn QuantileTransformer)
        warnings.filterwarnings('ignore', '.*n_quantiles is set to n_samples')
    
    # initialize results
    cca_results = CcaResultsPipelines(
        data=data,
        verbose=verbose,
    ).set_fpath_sample_size(dpath_cca, n_pcs_all, sample_size, i_bootstrap_repetition)

    # run CCA analysis pipelines
    cca_methods = CcaAnalysis(
        data=data,
        normalize_loadings=normalize_loadings,
        seed=cv_seed,
        shuffle=cv_shuffle,
        debug=debug,
    )
    cca_results['without_cv'] = cca_methods.without_cv(
        data, i_learn, i_val, cca_pipeline, preprocess=True,
    )
    cca_results['repeated_cv'] = cca_methods.repeated_cv(
        data, i_learn, i_val, cca_pipeline,
        n_repetitions=cv_n_repetitions,
        n_folds=cv_n_folds,
        preprocess_before_cv=False,
        rotate_CAs=True,
        rotate_deconfs=False,
        ensemble_method='nanmean',
    )
    cca_results['repeated_cv_no_rotate'] = cca_methods.repeated_cv(
        data, i_learn, i_val, cca_pipeline,
        n_repetitions=cv_n_repetitions,
        n_folds=cv_n_folds,
        preprocess_before_cv=False,
        rotate_CAs=False,
        rotate_deconfs=False,
        ensemble_method='nanmean',
    )

    # print top correlations
    for method_name in cca_results.method_names:
        method_results = cca_results[method_name]
        for set_name in method_results.set_names:
            set_results = method_results[set_name]
            print(f'Corrs for {method_name} ({set_name}):\t{set_results.corrs[:10]}')
    
    # save results
    cca_results.save()
    