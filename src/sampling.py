from __future__ import annotations
from abc import ABC, abstractmethod
from pathlib import Path
import warnings

import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.utils import resample

from .base import _Base
from .data_processing import XyData

class _Samples(_Base, ABC):

    def __init__(self, seed=None, **kwargs) -> None:
        super().__init__(**kwargs)

        self.seed = seed
        self.rng = np.random.default_rng(self.seed)

    @abstractmethod
    def _generate_samples(self):
        raise NotImplementedError

class BootstrapSamples(_Samples):

    def __init__(
        self,
        dpath,
        tag,
        n_bootstrap_repetitions,
        n_sample_sizes,
        upper_bound_fraction=0.5,
        match_val_set_size=False,
        sample_size_min=None,
        sample_size_max=None,
        max_n_PCs=None,
        n_folds=5,
        subset_fn=None,
        stratify=False,
        seed=None,
        verbose=False,
        generate=True,
    ) -> None:

        super().__init__(seed=seed, verbose=verbose)

        data = XyData(dpath=dpath).load()

        if max_n_PCs is None:
            max_n_PCs = np.min(data.n_features_datasets)
        
        self.dpath = Path(dpath)
        self.tag = tag
        self.n_bootstrap_repetitions = n_bootstrap_repetitions
        self.n_sample_sizes = n_sample_sizes
        self.upper_bound_fraction = upper_bound_fraction
        self.match_val_set_size = match_val_set_size
        self.max_n_PCs = max_n_PCs
        self.n_folds = n_folds
        self.subset_fn = subset_fn
        self.stratify = stratify

        self.sample_sizes = None
        self.i_samples_learn_all = None
        self.i_samples_val_all = None

        self.n_subjects = len(data.subjects)
        self.random_state = np.random.RandomState(self.seed)

        if sample_size_min is None:
            sample_size_min = int(np.ceil(max_n_PCs / (1 - (1/n_folds))))

        if sample_size_max is None:
            sample_size_max =  self._get_upper_bound(self.n_subjects)
        elif sample_size_max > self.n_subjects:
            raise ValueError(
                f'sample_size_max ({sample_size_max}) is larger than '
                f'the total number of subjects ({self.n_subjects})!'
            )

        self.sample_size_min = sample_size_min
        self.sample_size_max = sample_size_max

        self.fname = self.generate_fname()

        if generate:
            self._generate_samples(data)

    def generate_fname(self):
        str_tag = f'-{self.tag}' if self.tag is not None else ''
        str_matchval = '_matchval' if self.match_val_set_size else ''
        str_stratify = '_stratify' if self.stratify else ''
        return (
            f'bootstrap_samples{str_tag}-{self.sample_size_min}_{self.sample_size_max}'
            f'_{self.n_sample_sizes}steps_{self.n_bootstrap_repetitions}times{str_matchval}{str_stratify}'
        )

    def _get_upper_bound(self, n_samples):
        return int(self.upper_bound_fraction * n_samples)

    def _generate_samples(self, data: XyData):            

        # bounds for sample size
        # n_subjects = len(data.subjects)
        # if self.max_n_PCs is None:
        #     self.max_n_PCs = np.min(data.n_features_datasets)
        # sample_size_min = np.ceil(self.max_n_PCs / (1 - (1/self.n_folds)))
        # sample_size_max = int(self.val_sample_fraction * n_subjects)

        if self.sample_size_min > self.sample_size_max:
            raise ValueError(f'Invalid bounds: min={self.sample_size_min}, max={self.sample_size_max}')

        # log-spaced sample sizes
        sample_sizes = np.geomspace(self.sample_size_min, self.sample_size_max, self.n_sample_sizes, dtype=np.int64)

        if len(set(sample_sizes)) != len(sample_sizes):
            warnings.warn('Duplicate sample sizes')

        if self.subset_fn is None:
            i_samples = np.arange(self.n_subjects) # to be sampled from
        else:
            i_samples = self.subset_fn(data)
            self.sample_size_max = self._get_upper_bound(len(i_samples))
            sample_sizes = [s for s in sample_sizes if s <= self.sample_size_max]

            print(f'Selected subset: {len(i_samples)} subjects')
            print(f'Keeping {len(sample_sizes)} sample sizes')

        # handle groups
        if self.stratify:
            if data.group is None:
                raise ValueError('self.stratify=True but data.group is None')
            else:
                groups = data.group
        else:
            groups = None

        i_samples_learn_all = [] # list of dicts
        i_samples_val_all = [] # list of dicts OR list only (val samples are same across sample sizes for each in/out split)

        for i_bootstrap_repetition in range(self.n_bootstrap_repetitions):

            # split dataset into in/out-sample sets
            i_samples_in = resample(
                i_samples, 
                n_samples=self.sample_size_max, 
                replace=False, 
                stratify=groups.iloc[i_samples] if groups is not None else None)
            i_samples_out = np.array(list(set(i_samples) - set(i_samples_in)))
            # if groups is not None:
            #     i_samples_in = self.rng.choice(i_samples, size=self.sample_size_max, replace=False)
            #     i_samples_out = np.array(list(set(i_samples) - set(i_samples_in)))
            # else:
            #     cv = StratifiedKFold(n_splits=2, shuffle=True, random_state=self.random_state)
            #     i_samples_in, i_samples_out = list(cv.split(X=i_samples, y=groups))[0]

            i_samples_learn_all.append({})
            if self.match_val_set_size:
                i_samples_val_all.append({})
            else:
                i_samples_val_all.append(i_samples_out)

            for sample_size in sample_sizes:

                # sample with replacement
                groups_in = data.group.iloc[i_samples_in] if groups is not None else None
                i_samples_learn = resample(i_samples_in, n_samples=sample_size, replace=True, stratify=groups_in)
                # i_samples_learn = self.rng.choice(i_samples_in, size=sample_size, replace=True)
                i_samples_learn_all[i_bootstrap_repetition][sample_size] = i_samples_learn

                if self.match_val_set_size:
                    groups_out = data.group.iloc[i_samples_out] if groups is not None else None
                    i_samples_val = resample(i_samples_out, n_samples=sample_size, replace=True, stratify=groups_out)
                    # i_samples_val = self.rng.choice(i_samples_out, size=sample_size, replace=True)
                    i_samples_val_all[i_bootstrap_repetition][sample_size] = i_samples_val

        self.sample_sizes = sample_sizes
        self.i_samples_learn_all = i_samples_learn_all
        self.i_samples_val_all = i_samples_val_all

    def load(self) -> BootstrapSamples:
        return super().load()

    @classmethod
    def rename(cls, fpath_old):
        fpath_old = Path(fpath_old)
        samples: BootstrapSamples = cls.load_fpath(fpath_old)
        if (samples.sample_sizes is None) or (len(samples.sample_sizes) == 0):
            raise RuntimeError('sample_sizes is None')
        samples.sample_size_min = min(samples.sample_sizes)
        samples.sample_size_max = max(samples.sample_sizes)
        samples.fname = samples.generate_fname()
        if samples.verbose:
            print(f'Copying {fpath_old} to {samples.generate_fpath()}')
            print('(Old file not removed)')
        samples.save()

    def __str__(self) -> str:
        names = [
            'n_bootstrap_repetitions',
            'n_sample_sizes',
            'sample_sizes',
        ]
        return self._str_helper(names=names)
