from pathlib import Path
from abc import ABC
from .utils import save_pickle, load_pickle

class _Base(ABC):

    def generate_fpath(self, dpath=None, fname=None):
        if dpath is None:
            dpath = getattr(self, 'dpath', None)
        if fname is None:
            fname = getattr(self, 'fname', None)

        if dpath is None or fname is None:
            raise ValueError(f'Either dpath ({dpath}) or fpath ({fname}) is undefined')
        
        return Path(dpath, fname)

    def save(self, dpath=None):
        fpath = self.generate_fpath(dpath=dpath)
        verbose = getattr(self, 'verbose', False)
        save_pickle(self, fpath, verbose=verbose)

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
    def __init__(self) -> None:
        self.dataset_names = []
        self.conf_name = None
        self.udis_datasets = []
        self.udis_conf = None
        self.n_features_datasets = []
        self.n_features_conf = None
        self.subjects = None
