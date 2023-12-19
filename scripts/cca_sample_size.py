#!/usr/bin/env python
from copy import deepcopy
import warnings
import click
import numpy as np
import pandas as pd
from src.database_helpers import DatabaseHelper
from src.pipeline_definitions import build_cca_pipeline, build_feature_selector
from src.cca import CcaAnalysis, CcaResultsSampleSize
from src.data_processing import XyData
from src.sampling import BootstrapSamples
from src.utils import print_params

SUPPRESS_WARNINGS = True
DEBUG = False

# CCA settings
NORMALIZE_LOADINGS = True

# for repeated CV
CV_N_REPETITIONS = 10
CV_N_FOLDS = 5
CV_SEED = 3791
CV_SHUFFLE = True

# for feature selection
THRESHOLD_MISSING = 0.5
# THRESHOLD_HIGH_FREQ_CONF = 0.95#1

def select_features(data: XyData, i_learn, high_freq_threshold=0.8):
    data = deepcopy(data)

    X_selected_data = {}
    # feature selection (datasets)
    for i_dataset, dataset_name in enumerate(data.dataset_names):

        print(f'========== {dataset_name} ==========')
        dropped_features_data = {}
        feature_selector_data = build_feature_selector(
            dropped_features_dict=dropped_features_data,
            remove_missing__threshold=THRESHOLD_MISSING,
            remove_high_freq__threshold=high_freq_threshold,
        )
        print(feature_selector_data)

        feature_selector_data.fit(data.X[dataset_name].iloc[i_learn])
        X_selected_data[dataset_name] = feature_selector_data.transform(data.X[dataset_name])
        data.udis_datasets[i_dataset] = pd.MultiIndex.from_product([
            [dataset_name], 
            X_selected_data[dataset_name].columns
        ])

        for reason, dropped in dropped_features_data.items():
            print(f'Dropped {len(dropped)} features ({reason})')
            print(f'{dropped}')

    # feature selection (conf)
    print(f'========== {data.conf_name} ==========')
    dropped_features_conf = {}
    feature_selector_conf = build_feature_selector(
        dropped_features_dict=dropped_features_conf,
        remove_missing__threshold=THRESHOLD_MISSING,
        # remove_high_freq__threshold=THRESHOLD_HIGH_FREQ_CONF,
        remove_high_freq__threshold=high_freq_threshold,
    )
    print(feature_selector_conf)

    feature_selector_conf.fit(data.X[data.conf_name].iloc[i_learn])
    X_selected_data[data.conf_name] = feature_selector_conf.transform(data.X[data.conf_name])
    data.udis_conf = pd.MultiIndex.from_product([
        [data.conf_name], 
        # data.X[data.conf_name].columns,
        X_selected_data[data.conf_name].columns,
    ])

    for reason, dropped in dropped_features_conf.items():
        print(f'Dropped {len(dropped)} confs ({reason})')
        print(f'{dropped}')

    # update X
    data.X = pd.concat(X_selected_data, axis='columns')
    return data


@click.command()
@click.argument('n_sample_sizes', type=int)
@click.argument('n_bootstrap_repetitions', type=int)
@click.argument('i_sample_size', type=int)
@click.argument('i_bootstrap_repetition', type=int)
@click.argument('n_PCs_all', type=int, nargs=-1) # if 0, will be set to number of features
@click.option('--min', 'sample_size_min', type=int)
@click.option('--max', 'sample_size_max', type=int)
@click.option('--match-val/--no-match-val', 'match_val_set_size', default=False)
@click.option('--dpath-data', required=True, envvar='DPATH_PROCESSED')
@click.option('--dpath-cca', default='.', envvar='DPATH_CCA_SAMPLE_SIZE')
@click.option('--tag')
@click.option('--normalize-loadings', default=NORMALIZE_LOADINGS)
@click.option('--stratify/--no-stratify', default=False)
@click.option('--scipy-procrustes/--old-procrustes', 'use_scipy_procrustes', default=False)
@click.option('--cv-n-repetitions', default=CV_N_REPETITIONS)
@click.option('--cv-n-folds', default=CV_N_FOLDS)
@click.option('--cv-seed', default=CV_SEED)
@click.option('--cv-shuffle', default=CV_SHUFFLE)
@click.option('--null-model/--no-null-model', default=False)
@click.option('--dpath-schema', required=True, envvar='DPATH_SCHEMA')
@click.option('--fpath-udis', required=True, envvar='FPATH_UDIS')
@click.option('--suppress-warnings/--with-warnings', default=SUPPRESS_WARNINGS)
@click.option('--debug/--no-debug', default=DEBUG)
@click.option('--verbose/--quiet', default=True)
def cca_sample_size(n_sample_sizes, n_bootstrap_repetitions, i_sample_size, 
    i_bootstrap_repetition, n_pcs_all, sample_size_min, sample_size_max, match_val_set_size,
    dpath_data, dpath_cca, stratify, use_scipy_procrustes, tag, normalize_loadings, cv_n_repetitions, cv_n_folds,
    cv_seed, cv_shuffle, null_model, dpath_schema, fpath_udis, suppress_warnings, debug, verbose,
):

    print_params(locals())

    # load data
    data = XyData(dpath_data).load()
    print(data)
    
    n_datasets = len(data.dataset_names)

    # check/process number of PCs/CAs
    if len(n_pcs_all) != n_datasets:
        raise ValueError(f'Mismatch between n_PCs_all (size {len(n_pcs_all)}) and data ({n_datasets} datasets)')
    for i_dataset in range(len(data.dataset_names)):
        if n_pcs_all[i_dataset] == 0:
            n_pcs_all[i_dataset] = data.n_features_datasets[i_dataset]
    
    # load bootstrap sample
    bootstrap_samples = BootstrapSamples(
        dpath=dpath_data,
        n_bootstrap_repetitions=n_bootstrap_repetitions,
        n_sample_sizes=n_sample_sizes,
        max_n_PCs=max(n_pcs_all),
        stratify=stratify,
        sample_size_min=sample_size_min,
        sample_size_max=sample_size_max,
        match_val_set_size=match_val_set_size,
        tag=tag,
        generate=False, # don't generate samples, load from existing file
    ).load()
    print(bootstrap_samples)

    # get learn/val indices
    sample_size = bootstrap_samples.sample_sizes[i_sample_size-1] # zero-indexing
    i_learn = bootstrap_samples.i_samples_learn_all[i_bootstrap_repetition-1][sample_size]
    if bootstrap_samples.match_val_set_size:
        i_val = bootstrap_samples.i_samples_val_all[i_bootstrap_repetition-1][sample_size]
    else:
        i_val = bootstrap_samples.i_samples_val_all[i_bootstrap_repetition-1]
    print(f'Sample size: {sample_size}')

    n_CAs = min(n_pcs_all)
    print(f'Using {n_CAs} latent dimensions')

    if suppress_warnings:
        # this warnings occurs when n_samples < 1000 (default n_quantiles in sklearn QuantileTransformer)
        warnings.filterwarnings('ignore', '.*n_quantiles is set to n_samples')
    
    data = select_features(data, i_learn=i_learn, high_freq_threshold=(1 - (1/cv_n_folds)))
    
    # build pipeline/model
    db_helper = DatabaseHelper(dpath_schema, fpath_udis)
    expected_udis_conf = data.X[data.conf_name].columns # updated
    cca_pipeline = build_cca_pipeline(
        dataset_names=data.dataset_names,
        conf_name=data.conf_name,
        n_PCs_all=n_pcs_all,
        n_CAs=n_CAs,
        verbosity=1,
        kwargs_conf={
            'squarer__db_helper': db_helper,
            'squarer__expected_udis': expected_udis_conf,
            'inv_norm__db_helper': db_helper,
            'inv_norm__expected_udis': expected_udis_conf,
        }
    )
    for dataset_name in data.dataset_names:
        cca_pipeline['preprocessor'].data_pipelines.set_params(**{
            f'{dataset_name}__inv_norm__db_helper': db_helper,
            f'{dataset_name}__inv_norm__expected_udis': data.X[dataset_name].columns,
        })
    print('------------------------------------------------------------------')
    print(cca_pipeline)
    print('------------------------------------------------------------------')

    # initialize results
    cca_results = CcaResultsSampleSize(
        sample_size=sample_size,
        i_bootstrap_repetition=i_bootstrap_repetition,
        data=data,  # with updated UDIs
        verbose=verbose,
    ).set_fpath_sample_size(
        dpath_cca, n_pcs_all, tag, sample_size, i_bootstrap_repetition, 
        null_model=null_model,
    )

    # run CCA analysis pipelines
    cca_methods = CcaAnalysis(
        data=data,
        normalize_loadings=normalize_loadings,
        seed=cv_seed,
        shuffle=cv_shuffle,
        debug=debug,
        null_model=null_model,
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
        use_scipy_procrustes=use_scipy_procrustes,
        procrustes_reference=cca_results['without_cv'],
    )
    # cca_results['repeated_cv_no_rotate'] = cca_methods.repeated_cv(
    #     data, i_learn, i_val, cca_pipeline,
    #     n_repetitions=cv_n_repetitions,
    #     n_folds=cv_n_folds,
    #     preprocess_before_cv=False,
    #     rotate_CAs=False,
    #     rotate_deconfs=False,
    #     ensemble_method='nanmean',
    # )

    # print top correlations
    for method_name in cca_results.method_names:
        method_results = cca_results[method_name]
        for set_name in method_results.set_names:
            set_results = method_results[set_name]
            print(f'Corrs for {method_name} ({set_name}):\t{set_results.corrs[:10]}')
    
    # save results
    cca_results.save()

if __name__ == '__main__':
    np.set_printoptions(precision=4, suppress=True, linewidth=100, sign=' ')
    cca_sample_size()
    