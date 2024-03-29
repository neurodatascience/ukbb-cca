from __future__ import annotations
import re
import warnings
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Union

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold
from sklearn.base import clone

from .base import _Base, _BaseData, NestedItems
from .data_processing import XyData
from .ensemble_model import EnsembleCCA
from .cca_utils import cca_score, cca_get_loadings
from .utils import add_suffix, load_pickle, select_rows

LEARN_SET = 'learn'
VAL_SET = 'val'

class CcaResults(_Base):
    def __init__(self, CAs, deconfs, normalize_loadings=True) -> None:
        self.CAs = CAs
        self.corrs = cca_score(CAs)
        self.loadings = cca_get_loadings(deconfs, CAs, normalize=normalize_loadings)

    def loadings_to_df(self, udis):
        self.loadings = self._apply_to_datasets(
            self.loadings,
            lambda view, i: pd.DataFrame(view, index=udis[i].droplevel()),
            with_index=True,
        )

    @classmethod
    def _apply_to_datasets(cls, views, func, with_index=False):
        if with_index:
            return [func(view, i) for i, view in enumerate(views)]
        else:
            return [func(view) for view in views]

    @classmethod
    def _multiply_datasets(cls, views, values):
        return cls._apply_to_datasets(views, lambda x: x*values)

    def multiply_corrs(self, values):
        if len(values) != len(self.corrs):
            raise ValueError(
                f'Invalid values. Expected length {len(self.corrs)}'
                f', got {len(values)}'
            )
        self.corrs *= values

    def multiply_CAs(self, values):
        self.CAs = self._multiply_datasets(self.CAs, values)
        
    def multiply_loadings(self, values):
        self.loadings = self._multiply_datasets(self.loadings, values)

    def __str__(self) -> str:
        components = [
            f'corrs={self.corrs.shape if self.corrs is not None else None}'
        ]
        for name in ['CAs', 'loadings']:
            attr = getattr(self, name)
            components.append(
                f'{name}={[x.shape for x in attr] if attr is not None else None}'
            )
        return self._str_helper(components=components)

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

    def __str__(self) -> str:
        results_dict = {
            set_name: self[set_name]
            for set_name in self.set_names
        }
        return self._str_helper(components=[results_dict])

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
        sample_size, i_bootstrap_repetition, null_model=False) -> CcaResultsPipelines:

        sample_size_str = self.get_dname_sample_size(sample_size)
        if null_model:
            tag = f'{tag}-null_model'
        self.dpath = Path(dpath_cca) / self.get_dname_PCs(n_PCs_all) / tag / sample_size_str
        self.fname = add_suffix(sample_size_str, f'rep{i_bootstrap_repetition}')

        return self

    @staticmethod
    def get_dname_PCs(n_PCs_all):
        n_PCs_str = '_'.join([str(n) for n in n_PCs_all])
        return f'PCs_{n_PCs_str}'

    @staticmethod
    def get_dname_tag(tag):
        return f'{tag}'

    @staticmethod
    def get_dname_sample_size(sample_size):
        return f'sample_size_{sample_size}'

    def __str__(self) -> str:
        results_dict = {
            method_name: self[method_name]
            for method_name in self.method_names
        }
        return self._str_helper(components=[results_dict])

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

    def __str__(self) -> str:
        results_dict = {
            method_name: self[method_name]
            for method_name in self.method_names
        }
        return self._str_helper(
            components=[results_dict], 
            names=['sample_size', 'i_bootstrap_repetition'],
        )

class CcaResultsCombined(NestedItems[list[CcaResults]], _BaseData):

    @staticmethod
    def agg_func_helper(data, func_pd_str: str, func_np, kwargs_pd=None, kwargs_np=None):
        
        if kwargs_pd is None:
            kwargs_pd = {}
        if kwargs_np is None:
            kwargs_np = {}

        # if pandas dataframe
        try:
            df_concat = pd.concat(data)
            df_grouped = df_concat.groupby(df_concat.index)
            func_pd = getattr(df_grouped, func_pd_str)
            return func_pd(**kwargs_pd)
        # else numpy array
        except TypeError:
            if not 'axis' in kwargs_np:
                kwargs_np['axis'] = 0
            return func_np(data, **kwargs_np)

    agg_funcs_map = {
        'mean': (lambda x: CcaResultsCombined.agg_func_helper(x, 'mean', np.nanmean)),
        'std': (lambda x: CcaResultsCombined.agg_func_helper(x, 'std', np.nanstd)),
        # 'mean': (lambda x: np.nanmean(x, axis=0)),
        # 'std': (lambda x: np.nanstd(x, axis=0)),
    }

    @dataclass
    class Summary():
        CAs: list[np.array]
        corrs: np.array
        loadings: list[np.array]

    def __init__(self, levels: list[str], data: _BaseData = None):
        super().__init__(levels, factory=list)
        if data is not None:
            self.set_dataset_params(data)

    @classmethod
    def from_items(cls, items: Mapping, levels: Union[Iterable, Mapping[str, list], None] = None, data: _BaseData = None):
        output = super().from_items(items, levels)
        if data is not None:
            output.set_dataset_params(data)
        return output

    def __getitem__(self, keys: tuple):
        item = super().__getitem__(keys)
        if isinstance(item, self.__class__):
            item.set_dataset_params(self)
        return item
    
    def set_dataset_params(self, data: _BaseData):
        return super().set_dataset_params(
            dataset_names=data.dataset_names,
            conf_name=data.conf_name,
            udis_datasets=data.udis_datasets,
            udis_conf=data.udis_conf,
            n_features_datasets=data.n_features_datasets,
            n_features_conf=data.n_features_conf,
            subjects=data.subjects,
        )
        
    def append(self, keys: tuple, item):
        self.levels.update_levels(keys)
        self[keys].append(item)

    def aggregate(self, agg_funcs: Union[Mapping[str, Callable[[list]], Any], None] = None):
        
        def cca_results_wrapper(func: Callable[[list[CcaResults]], Any]):
            def _cca_results_wrapper(results: list[CcaResults]) -> CcaResultsCombined.Summary:
                if len(results) < 1:
                    raise RuntimeError('CcaResults list cannot be empty')

                n_datasets = len(results[0].CAs)
                corrs = [result.corrs for result in results]

                CAs = [
                    [result.CAs[i_dataset] for result in results]
                    for i_dataset in range(n_datasets)
                ]

                loadings = [
                    [result.loadings[i_dataset] for result in results]
                    for i_dataset in range(n_datasets)
                ]
                
                return CcaResultsCombined.Summary(
                    CAs=[func(CA) for CA in CAs],
                    corrs=func(corrs),
                    loadings=[func(loading) for loading in loadings],
                )
                
            return _cca_results_wrapper
        
        if agg_funcs is None:
            agg_funcs = deepcopy(self.agg_funcs_map)

        agg_funcs_wrapped: dict[str, Callable[[list[CcaResults]], CcaResultsCombined.Summary]] = {}
        for func_name, func in agg_funcs.items():
            agg_funcs_wrapped[func_name] = cca_results_wrapper(func)

        return self.apply_funcs(agg_funcs_wrapped, data=self)

class CcaAnalysis(_Base):

    def __init__(
        self, 
        data: XyData,
        normalize_loadings=True,
        seed = None,
        shuffle = False,
        debug=False,
        null_model=False,
        # shuffle_if_null=False,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)

        self.dataset_names = data.dataset_names
        self.normalize_loadings = normalize_loadings
        self.seed = seed
        self.shuffle = shuffle
        self.debug = debug
        self.null_model = null_model
        # self.shuffle_if_null = shuffle_if_null

        if not shuffle:
            self.random_state = None
        else:
            self.random_state = np.random.RandomState(seed)

        self.cache_repeated_cv_models = None
        self.cache_repeated_cv_deconfs_learn = None
        self.cache_repeated_cv_deconfs_val = None
        self.cache_repeated_cv_subjects = None

    def get_i_shuffled_all(self, X: pd.DataFrame, n_datasets: int):
        i_shuffled_all = [np.arange(len(X))]
        for _ in range(n_datasets - 1):
            i_shuffled_all.append(
                self.random_state.choice(np.arange(len(X)), size=len(X), replace=False),
            )
        return i_shuffled_all

    def without_cv(self, data: XyData, i_train, i_test, model: Pipeline, preprocess=True, deconfs=None, return_fitted_model=False, i_shuffled_all=None):

        model = clone(model)

        X = data.X
        X_train = select_rows(X, i_train)
        X_test = select_rows(X, i_test)

        if self.null_model:
            if (not preprocess) and (i_shuffled_all is None):
                raise RuntimeError(f'i_shuffled_all must be given for the null model if preprocess=False')
            i_shuffled_all = self.get_i_shuffled_all(X_train, n_datasets=len(self.dataset_names))
        else:
            i_shuffled_all = None

        if preprocess:

            # # TODO remove
            # nan_cols = X.columns[X.isna().all()]
            # if len(nan_cols) > 0:
            #     warnings.warn(f'nan columns: {list(nan_cols)}')
            # constant_cols = X.columns[(X == X.iloc[0]).all()]
            # if len(constant_cols) > 0:
            #     warnings.warn(f'constant columns: {list(constant_cols)}')

            preprocessor = model['preprocessor']
            X_train_preprocessed = preprocessor.fit_transform(X_train, i_shuffled_all=i_shuffled_all)
            X_test_preprocessed = preprocessor.transform(X_test)

            deconfounder = self.model_transform_deconfounder(model)
            deconfs_train = deconfounder.transform(X_train, i_shuffled_all=i_shuffled_all)
            deconfs_test = deconfounder.transform(X_test)

        else:
            if deconfs is None:
                raise RuntimeError(f'deconfs must be given if preprocess=False')
            if self.null_model:
                raise NotImplementedError('null models not supported if preprocess=False')
            X_train_preprocessed = X_train
            X_test_preprocessed = X_test
            deconfs_train = select_rows(deconfs, i_train)
            deconfs_test = select_rows(deconfs, i_test)

        # if self.null_model and self.shuffle_if_null:

        #     if not isinstance(X_train_preprocessed, list):
        #         raise RuntimeError(f'Expected X_train_preprocessed to be a list, got: {type(X_train_preprocessed)}')
        #     else:
        #         print(f'X_train_preprocessed (CcaAnalysis.without_cv): {[tmp.shape for tmp in X_train_preprocessed]}')

        #     # # randomly shuffle all views
        #     # need to shuffle the deconfounded data as well
        #     # for view in X_train_preprocessed:
        #     #     self.random_state.shuffle(view)

        #     # randomly shuffle all views except the first one
        #     # need to shuffle the deconfounded data as well
        #     for view in X_train_preprocessed[1:]:
        #         self.random_state.shuffle(view)
            
        # print([np.sum(np.logical_not(np.isfinite(X))) for X in X_train_preprocessed])
        # else:
        cca = model['cca']

        try:
            cca.fit(X_train_preprocessed)
        except Exception as exception:
            if self.debug:
                import dill as pickle
                with open('debug.pkl', 'wb') as file_debug:
                    pickle.dump({
                        'data': data,
                        'i_train': i_train,
                        'i_test': i_test,
                        'model': model,
                        'preprocess': preprocess,
                        'deconfs': deconfs,
                        'return_fitted_model': return_fitted_model,
                        'X_train_preprocessed': X_train_preprocessed,
                        'X_test_preprocessed': X_test_preprocessed,
                        'deconfs_train': deconfs_train,
                        'deconfs_test': deconfs_test,
                    }, file_debug)
            raise exception

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

    def cv(self, data: XyData, model: Pipeline, n_folds, preprocess=True, deconfs=None, i_shuffled_all=None):

        model = clone(model)

        n_subjects = len(data.subjects)
        cv_splitter = KFold(n_splits=n_folds, shuffle=self.shuffle, random_state=self.random_state)
        
        fitted_models = []
        i_train_all = []
        i_test_all = []
        for i_train, i_test in cv_splitter.split(np.arange(n_subjects)):

            fitted_model = self.without_cv(
                data, i_train, i_test, model,
                preprocess=preprocess, deconfs=deconfs,
                return_fitted_model=True,
                i_shuffled_all=i_shuffled_all,
            )
            fitted_models.append(fitted_model)

            if self.debug:
                i_train_all.append(i_train)
                i_test_all.append(i_test)

        return fitted_models, i_train_all, i_test_all

    def repeated_cv(self, data: XyData, i_learn, i_val, model: Pipeline, n_repetitions, n_folds, 
            preprocess_before_cv=False, rotate_CAs=True, rotate_deconfs=False, 
            ensemble_method='nanmean', use_scipy_procrustes=False, procrustes_reference: CcaResultsSets = None):

        def apply_ensemble_CCA(data_learn: XyData, data_val: XyData, rotate, model_transform, procrustes_reference=None):
            ensemble_model = EnsembleCCA(fitted_models, rotate=rotate, model_transform=model_transform, use_scipy_procrustes=use_scipy_procrustes, procrustes_reference=procrustes_reference)
            result_learn = ensemble_model.fit_transform(data_learn.X, apply_ensemble_method=True, ensemble_method=ensemble_method)
            result_val = ensemble_model.transform(data_val.X, apply_ensemble_method=True, ensemble_method=ensemble_method)
            return result_learn, result_val, ensemble_model

        model = clone(model)

        # TODO untested
        if preprocess_before_cv:
            if self.null_model:
                raise NotImplementedError('preprocess_before_cv not implemented for null model')
                
            preprocessor = model['preprocessor']
            data.X = preprocessor.fit_transform(data.X, i_shuffled_all=i_shuffled_all) # preprocessed data
            deconfounder = self.model_transform_deconfounder(model)
            deconfs = deconfounder.transform(data.X) # keep deconfs for computing loadings later
            model = model.set_params(preprocessor='passthrough')
        else:
            i_shuffled_all = None

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
                        preprocess=(not preprocess_before_cv),
                        i_shuffled_all=i_shuffled_all,
                    )
                    fitted_models.extend(new_models)
                    i_train_all.extend(new_i_train)
                    i_test_all.extend(new_i_test)

                # try again if non-convergence error
                except np.linalg.LinAlgError as exception:
                    print(f'LinAlgError: {exception}')
                    continue

                except Exception as exception:

                    # this error happens in the CCA zoo model (eigh function)
                    # the input data does not seem to contain infs or NaNs
                    # so ignore and try more models
                    if isinstance(exception, ValueError) and str(exception) == 'array must not contain infs or NaNs':
                        continue

                    # otherwise throw the error
                    else:
                        raise exception

        # ensemble model
        CAs_learn, CAs_val, ensemble_model = apply_ensemble_CCA(
            data_learn, data_val, rotate=rotate_CAs, model_transform=self.model_transform_cca,
            procrustes_reference=procrustes_reference[LEARN_SET].CAs if procrustes_reference is not None else None,
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
