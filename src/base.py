from pathlib import Path
from abc import ABC
from .utils import save_pickle, load_pickle

class _Base(ABC):

    fname = None

    def __init__(self, dpath=None, verbose=False) -> None:
        super().__init__()
        self.dpath = dpath
        self.verbose = verbose

    def generate_fpath(self, dpath=None, fname=None):
        if dpath is None:
            dpath = self.dpath
        if fname is None:
            fname = self.fname

        if dpath is None or fname is None:
            raise ValueError(f'dpath ({dpath}) and fpath ({fname}) cannot be None')
        
        return Path(dpath, fname)

    def save(self, dpath=None):
        fpath = self.generate_fpath(dpath=dpath)
        save_pickle(self, fpath, verbose=self.verbose)

    def load(self):
        fpath = self.generate_fpath()
        return load_pickle(fpath)

    def _str_helper(self, components=None, names=None, sep=', '):
        if components is None:
            components = []

        if names is not None:
            for name in names:
                components.append(f'{name}={getattr(self, name)}')
        return f'{type(self).__name__}({sep.join(components)})'

    def __str__(self) -> str:
        return self._str_helper()

class _BaseData(_Base, ABC):
    def __init__(
        self, 
        dataset_names=None,
        conf_name=None,
        udis_datasets=None,
        udis_conf=None,
        n_features_datasets=None,
        n_features_conf=None,
        subjects=None,
        **kwargs,
    ) -> None:

        super().__init__(**kwargs)

        if dataset_names is None:
            dataset_names = []
        if udis_datasets is None:
            udis_datasets = []
        if n_features_datasets is None:
            n_features_datasets = []

        self.dataset_names = dataset_names
        self.conf_name = conf_name
        self.udis_datasets = udis_datasets
        self.udis_conf = udis_conf
        self.n_features_datasets = n_features_datasets
        self.n_features_conf = n_features_conf
        self.subjects = subjects
