from __future__ import annotations
import warnings
from pathlib import Path
from abc import ABC, abstractmethod
import numpy as np
from .base import _Base
from .data_processing import XyData

class _Samples(_Base, ABC):

    def __init__(self, seed=None, **kwargs) -> None:
        super().__init__(**kwargs)

        self.seed = seed
        self.rng = np.random.default_rng(self.seed)

    @abstractmethod
    def generate(self, data: XyData):
        raise NotImplementedError

class BootstrapSamples(_Samples):

    def __init__(
        self,
        dpath,
        n_bootstrap_repetitions,
        n_sample_sizes,
        val_sample_fraction=0.5,
        max_n_PCs=100,
        n_folds=5,
        seed=None,
        verbose=False,
    ) -> None:

        super().__init__(seed=seed, verbose=verbose)
        
        self.dpath = Path(dpath)
        self.n_bootstrap_repetitions = n_bootstrap_repetitions
        self.n_sample_sizes = n_sample_sizes
        self.val_sample_fraction = val_sample_fraction
        self.max_n_PCs = max_n_PCs
        self.n_folds = n_folds

        self.sample_sizes = None
        self.i_samples_learn_all = None
        self.i_samples_val_all = None

        self.fname = (
            f'bootstrap_samples_{n_sample_sizes}steps'
            f'_{n_bootstrap_repetitions}times'
        )

    def generate(self, data: XyData):

        # bounds for sample size
        n_subjects = len(data.subjects)
        if self.max_n_PCs is None:
            self.max_n_PCs = np.min(data.n_features_datasets)
        sample_size_min = np.ceil(self.max_n_PCs / (1 - (1/self.n_folds)))
        sample_size_max = int(self.val_sample_fraction * n_subjects)

        if sample_size_min > sample_size_max:
            raise ValueError(f'Invalid bounds: min={sample_size_min}, max={sample_size_max}')

        # log-spaced sample sizes
        sample_sizes = np.geomspace(sample_size_min, sample_size_max, self.n_sample_sizes, dtype=np.int64)

        if len(set(sample_sizes)) != len(sample_sizes):
            warnings.warn('Duplicate sample sizes')

        i_samples = np.arange(n_subjects) # to be sampled from
        i_samples_learn_all = [] # list of dicts
        i_samples_val_all = [] # list only (val samples are same across sample sizes for each in/out split)

        for i_bootstrap_repetition in range(self.n_bootstrap_repetitions):

            # split dataset into in/out-sample sets
            i_samples_in = self.rng.choice(i_samples, size=sample_size_max, replace=False)
            i_samples_val = np.array(list(set(i_samples) - set(i_samples_in)))

            i_samples_val_all.append(i_samples_val)
            i_samples_learn_all.append({})

            for sample_size in sample_sizes:

                # sample with replacement
                i_samples_learn = self.rng.choice(i_samples_in, size=sample_size, replace=True)
                i_samples_learn_all[i_bootstrap_repetition][sample_size] = i_samples_learn

        self.sample_sizes = sample_sizes
        self.i_samples_learn_all = i_samples_learn_all
        self.i_samples_val_all = i_samples_val_all

    def load(self) -> BootstrapSamples:
        return super().load()

    def __str__(self) -> str:
        names = [
            'n_bootstrap_repetitions',
            'n_sample_sizes',
            'sample_sizes',
        ]
        return self._str_helper(names=names)