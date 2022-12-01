from __future__ import annotations
import re
import warnings
from pathlib import Path
from copy import deepcopy

import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold
from sklearn.base import clone

from .base import _Base, _BaseData
from .data_processing import XyData
from .ensemble_model import EnsembleCCA
from .cca_utils import cca_score, cca_get_loadings
from .utils import add_suffix, load_pickle, select_rows

LEARN_SET = 'learn'
VAL_SET = 'val'

class CcaResults():
    def __init__(self, CAs, deconfs, normalize_loadings=True) -> None:
        self.CAs = CAs
        self.corrs = cca_score(CAs)
        self.loadings = cca_get_loadings(deconfs, CAs, normalize=normalize_loadings)

class CcaResultsSets(_BaseData):

    def __init__(self, data: XyData, **kwargs) -> None:
        super().__init__(
            dataset_names=data.dataset_names, 
            conf_name=data.conf_name, 
            udis_datasets=data.udis_datasets, 
            udis_conf=data.udis_conf, 
            n_features_datasets=data.n_features_datasets, 
            n_features_conf=data.n_features_conf, 
            subjects=data.subjects, 
            **kwargs,
        )

        self.set_names = []

        # for debugging
        self.model = None
        self.i_train_all = None
        self.i_test_all = None

    def __getitem__(self, set_name) -> CcaResults:
        return self.__getattribute__(set_name)

    def __setitem__(self, set_name: str, results: CcaResults):
        if set_name in self.set_names:
            raise RuntimeError(f'Results for {set_name} set already exist')
        self.set_names.append(set_name)
        self.__setattr__(set_name, results)

class CcaResultsPipelines(_BaseData):
    def __init__(self, dpath=None, data: XyData = None, **kwargs) -> None:
        if data is not None:
            super().__init__(
                dataset_names=data.dataset_names, 
                conf_name=data.conf_name, 
                udis_datasets=data.udis_datasets, 
                udis_conf=data.udis_conf, 
                n_features_datasets=data.n_features_datasets, 
                n_features_conf=data.n_features_conf, 
                subjects=data.subjects, 
                **kwargs,
            )
        
        if dpath is not None:
            dpath = Path(dpath)

        self.dpath = dpath
        self.method_names = []

    def __setitem__(self, method_name: str, results: CcaResultsSets):
        if method_name in self.method_names:
            raise RuntimeError(f'Results for {method_name} already exist')
        self.method_names.append(method_name)
        self.__setattr__(method_name, results)

    def __getitem__(self, pipeline_name) -> CcaResultsSets:
        return self.__getattribute__(pipeline_name)

    def set_fpath_sample_size(self, dpath_cca, n_PCs_all, tag, 
        sample_size, i_bootstrap_repetition) -> CcaResultsPipelines:

        sample_size_str = self.get_dname_sample_size(sample_size)
        self.dpath = Path(dpath_cca) / self.get_dname_PCs(n_PCs_all) / tag / sample_size_str
        self.fname = add_suffix(sample_size_str, f'rep{i_bootstrap_repetition}')

        return self

    @staticmethod
    def get_dname_PCs(n_PCs_all):
        n_PCs_str = '_'.join([str(n) for n in n_PCs_all])
        return f'PCs_{n_PCs_str}'

    @staticmethod
    def get_dname_sample_size(sample_size):
        return f'sample_size_{sample_size}'

class CcaResultsSampleSize(CcaResultsPipelines):

    re_fname = re.compile(f'sample[_-]size[_-](\d+)[_-]rep(\d+)')

    def __init__(self, sample_size, i_bootstrap_repetition, dpath=None, data: XyData = None, **kwargs) -> None:
        super().__init__(dpath, data, **kwargs)
        self.sample_size = sample_size
        self.i_bootstrap_repetition = i_bootstrap_repetition

    @classmethod
    def load_and_cast(cls, fpath) -> CcaResultsSampleSize:
        fpath = Path(fpath)
        try:
            results = cls.load_fpath(fpath)
            if type(results.sample_size) == str:
                results.sample_size = int(results.sample_size)
            if type(results.i_bootstrap_repetition) == str:
                results.i_bootstrap_repetition = int(results.i_bootstrap_repetition)
            return results
        except RuntimeError:
            results = CcaResultsPipelines.load_fpath(fpath)
        results.__class__ = cls
        
        # get info from filename
        fname = fpath.stem
        sample_size, i_bootstrap_repetition = cls.re_fname.match(fname).groups()
        results.sample_size = int(sample_size)
        results.i_bootstrap_repetition = int(i_bootstrap_repetition)
        return results

class CcaAnalysis(_Base):

    def __init__(
        self, 
        data: XyData,
        normalize_loadings=True,
        seed = None,
        shuffle = False,
        debug=False,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)

        self.dataset_names = data.dataset_names
        self.normalize_loadings = normalize_loadings
        self.seed = seed
        self.shuffle = shuffle
        self.debug = debug

        if not shuffle:
            self.random_state = None
        else:
            self.random_state = np.random.RandomState(seed)

        self.cache_repeated_cv_models = None
        self.cache_repeated_cv_deconfs_learn = None
        self.cache_repeated_cv_deconfs_val = None
        self.cache_repeated_cv_subjects = None

    def without_cv(self, data: XyData, i_train, i_test, model: Pipeline, preprocess=True, deconfs=None, return_fitted_model=False):

        model = clone(model)

        X = data.X
        X_train = select_rows(X, i_train)
        X_test = select_rows(X, i_test)

        if preprocess:

            # # TODO remove
            # nan_cols = X.columns[X.isna().all()]
            # if len(nan_cols) > 0:
            #     warnings.warn(f'nan columns: {list(nan_cols)}')
            # constant_cols = X.columns[(X == X.iloc[0]).all()]
            # if len(constant_cols) > 0:
            #     warnings.warn(f'constant columns: {list(constant_cols)}')

            preprocessor = model['preprocessor']
            X_train_preprocessed = preprocessor.fit_transform(X_train)
            X_test_preprocessed = preprocessor.transform(X_test)

            deconfounder = self.model_transform_deconfounder(model)
            deconfs_train = deconfounder.transform(X_train)
            deconfs_test = deconfounder.transform(X_test)

        else:
            if deconfs is None:
                raise RuntimeError(f'deconfs must be given if preprocess=False')
            X_train_preprocessed = X_train
            X_test_preprocessed = X_test
            deconfs_train = select_rows(deconfs, i_train)
            deconfs_test = select_rows(deconfs, i_test)

        # print([np.sum(np.logical_not(np.isfinite(X))) for X in X_train_preprocessed])

        cca = model['cca']
        cca.fit(X_train_preprocessed)

        if return_fitted_model:
            return model
        else:
            results = CcaResultsSets(data)
            for set_name, X_preprocessed, deconfs in zip(
                [LEARN_SET, VAL_SET], 
                [X_train_preprocessed, X_test_preprocessed],
                [deconfs_train, deconfs_test],
            ):
                results[set_name] = CcaResults(
                    CAs=cca.transform(X_preprocessed),
                    deconfs=deconfs,
                    normalize_loadings=self.normalize_loadings,
                )
            # # results = {'corrs': {}, 'CAs': {}, 'loadings': {}}
            # for set_name, X_preprocessed in {LEARN_SET: X_train_preprocessed, VAL_SET: X_test_preprocessed}.items():
            #     CAs = cca.transform(X_preprocessed)
            #     results[set_name] = CcaResults(
            #         CAs=CAs,
            #         corrs=cca_score(CAs),
            #         loadings=cca_get_loadings(X_preprocessed, CAs, normalize=self.normalize_loadings),
            #     )
            #     # results['CAs'][set_name] = CAs
            #     # results['corrs'][set_name] = cca_score(CAs)
            #     # results['loadings'][set_name] = cca_get_loadings(X_preprocessed, CAs, normalize=normalize_loadings)

            #     # np.set_printoptions(precision=4, linewidth=100, suppress=True, sign=' ')
            #     # print(f'cca.score: {cca.score(X_preprocessed)[:10]}')
            #     # print(f'cca_score: {results["corrs"][set_name][:10]}')
            if self.debug:
                results.model = model
            return results
        
    def cv(self, data: XyData, model: Pipeline, n_folds, preprocess=True, deconfs=None):

        model = clone(model)

        n_subjects = len(data.subjects)
        cv_splitter = KFold(n_splits=n_folds, shuffle=self.shuffle, random_state=self.random_state)
        
        fitted_models = []
        i_train_all = []
        i_test_all = []
        for i_train, i_test in cv_splitter.split(np.arange(n_subjects)):

            fitted_model = self.without_cv(
                data, i_train, i_test, model,
                preprocess=preprocess, deconfs=deconfs, return_fitted_model=True)
            fitted_models.append(fitted_model)

            if self.debug:
                i_train_all.append(i_train)
                i_test_all.append(i_test)

        return fitted_models, i_train_all, i_test_all

    def repeated_cv(self, data: XyData, i_learn, i_val, model: Pipeline, n_repetitions, n_folds, 
            preprocess_before_cv=False, rotate_CAs=True, rotate_deconfs=False, 
            ensemble_method='nanmean'):

        def apply_ensemble_CCA(data_learn: XyData, data_val: XyData, rotate, model_transform):
            ensemble_model = EnsembleCCA(fitted_models, rotate=rotate, model_transform=model_transform)
            result_learn = ensemble_model.fit_transform(data_learn.X, apply_ensemble_method=True, ensemble_method=ensemble_method)
            result_val = ensemble_model.transform(data_val.X, apply_ensemble_method=True, ensemble_method=ensemble_method)
            return result_learn, result_val, ensemble_model

        model = clone(model)

        # TODO untested
        if preprocess_before_cv:
            preprocessor = model['preprocessor']
            data.X = preprocessor.fit_transform(data.X) # preprocessed data
            deconfounder = self.model_transform_deconfounder(model)
            deconfs = deconfounder.transform(data.X) # keep deconfs for computing loadings later
            model = model.set_params(preprocessor='passthrough')

        data_learn = data.subset(i_learn)
        data_val = data.subset(i_val)

        # for debugging
        i_train_all = []
        i_test_all = []

        # try to use cached results if possible
        subjects = data_learn.subjects
        if (self.cache_repeated_cv_models is not None) and (set(subjects) == set(self.cache_repeated_cv_subjects)):
            fitted_models = self.cache_repeated_cv_models
            deconfs_learn = self.cache_repeated_cv_deconfs_learn
            deconfs_val = self.cache_repeated_cv_deconfs_val
            # print('Using cached results')
        else:
            fitted_models = []
            deconfs_learn = None
            deconfs_val = None
            while (len(fitted_models) != (n_repetitions * n_folds)):

                try:
                    new_models, new_i_train, new_i_test = self.cv(
                        data_learn, model, n_folds, 
                        preprocess=(not preprocess_before_cv))
                    fitted_models.extend(new_models)
                    i_train_all.extend(new_i_train)
                    i_test_all.extend(new_i_test)

                # try again if non-convergence error
                except np.linalg.LinAlgError as error:
                    print(f'LinAlgError: {error}')
                    continue

        # ensemble model
        CAs_learn, CAs_val, ensemble_model = apply_ensemble_CCA(
            data_learn, data_val, rotate=rotate_CAs, model_transform=self.model_transform_cca,
        )
        if (deconfs_learn is None) and (deconfs_val is None):
            if preprocess_before_cv:
                deconfs_learn = select_rows(deconfs, i_learn)
                deconfs_val = select_rows(deconfs, i_val)
            else:
                deconfs_learn, deconfs_val, _ = apply_ensemble_CCA(
                    data_learn, data_val, rotate=rotate_deconfs, model_transform=self.model_transform_deconfounder,
                )

        # cache results
        self.cache_repeated_cv_models = fitted_models
        self.cache_repeated_cv_deconfs_learn = deconfs_learn
        self.cache_repeated_cv_deconfs_val = deconfs_val
        self.cache_repeated_cv_subjects = subjects

        results = CcaResultsSets(data)
        results[LEARN_SET] = CcaResults(
            CAs=CAs_learn,
            deconfs=deconfs_learn,
            normalize_loadings=self.normalize_loadings,
        )
        results[VAL_SET] = CcaResults(
            CAs=CAs_val,
            deconfs=deconfs_val,
            normalize_loadings=self.normalize_loadings,
        )

        if self.debug:
            results.model = ensemble_model
            results.i_train_all = i_train_all
            results.i_test_all = i_test_all

        return results

    def model_transform_cca(self, model: Pipeline):
        return model # no change

    def model_transform_deconfounder(self, model: Pipeline) -> Pipeline:
        params = {
            f'data_pipelines__{dataset_name}__pca': 'passthrough'
            for dataset_name in self.dataset_names
        }
        deconfounder = deepcopy(model)['preprocessor'].set_params(**params)
        return deconfounder
