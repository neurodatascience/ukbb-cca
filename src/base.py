from __future__ import annotations
from pathlib import Path
from abc import ABC
from collections import defaultdict, OrderedDict
from copy import deepcopy
from functools import wraps
from itertools import islice, product
from typing import Any, Callable, Generic, Iterable, Mapping, Sequence, TypeVar, Union
from .utils import save_pickle, load_pickle

T = TypeVar('T')
U = TypeVar('U')

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

    def save(self, dpath=None, verbose=None):
        if verbose is None and hasattr(self, 'verbose'):
            verbose = self.verbose
        fpath = self.generate_fpath(dpath=dpath)
        save_pickle(self, fpath, verbose=verbose)

    def load(self):
        fpath = self.generate_fpath()
        return self.load_fpath(fpath)

    @classmethod
    def load_fpath(cls, fpath):
        obj = load_pickle(fpath)
        if not isinstance(obj, cls):
            raise RuntimeError(f'Object loaded from {fpath} is not an instance of {cls}')
        return obj

    def _str_helper(self, components=None, names=None, sep=', '):
        if components is None:
            components = []

        if names is not None:
            for name in names:
                components.append(f'{name}={getattr(self, name)}')
        return f'{type(self).__name__}({sep.join([str(c) for c in components])})'

    def __str__(self) -> str:
        return self._str_helper()

    def __repr__(self) -> str:
        return self.__str__()

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
    ):

        super().__init__(**kwargs)

        self.set_dataset_params(
            dataset_names=dataset_names,
            conf_name=conf_name,
            udis_datasets=udis_datasets,
            udis_conf=udis_conf,
            n_features_datasets=n_features_datasets,
            n_features_conf=n_features_conf,
            subjects=subjects,
        )

    def set_dataset_params(
        self, 
        dataset_names=None,
        conf_name=None,
        udis_datasets=None,
        udis_conf=None,
        n_features_datasets=None,
        n_features_conf=None,
        subjects=None,
    ):
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

class NestedItems(Generic[T], _Base):

    class NestedLevels(OrderedDict[str, list]):

        def __init__(self, levels: Union[Sequence[str], Mapping[str, list], None] = None) -> None:

            if levels is None:
                levels = {}
            elif isinstance(levels, Sequence):
                levels = {level: self.initialize_level() for level in levels}

            super().__init__(deepcopy(levels))

        def initialize_level(self):
            return []

        @property
        def names(self):
            return list(self.keys())

        def add_level(self, name):
            self[name] = self.initialize_level()

        def update_level(self, level, key):
            if key not in self[level]:
                self[level].append(key)

        def update_levels(self, keys):
            keys = self.check_keys(keys)
            for level, key in zip(self.keys(), keys):
                self.update_level(level, key)

        def generate_all_keys(self, depth=None):
            if depth is None:
                depth = len(self)
            labels_ordered = [self[level] for level in islice(self, depth)]
            return product(*labels_ordered)

        def check_keys(self, keys, strict=True):
            if isinstance(keys, str) or not isinstance(keys, Iterable):
                keys = (keys,)
            if strict and len(keys) != len(self):
                raise KeyError(
                    f'Expected {len(self)} keys (got {len(keys)})'
                )
            elif len(keys) > len(self):
                raise KeyError(
                    f'Expected at most {len(self)} keys (got {len(keys)})'
                )
            return keys

    def __init__(self, levels: Union[Sequence[str], Mapping[str, list]], factory=None) -> None:
        
        def get_factory(factory):
            @wraps(factory)
            def _get_factory():
                return deepcopy(factory)
            return _get_factory

        if len(levels) < 1:
            raise ValueError(f'Must have at least 1 level (got {len(self.levels)})')
        
        self.levels = self.NestedLevels(levels)

        factory = defaultdict(factory)
        for _ in range(len(self.levels) - 1):
            factory = defaultdict(get_factory(factory))
        
        self.items = factory

    @classmethod
    def from_items(cls, items: Mapping, levels: Union[Iterable, Mapping[str, list], None] = None):

        def get_depth(d, mode: str = 'min') -> int:

            def _get_depth(d, mode: str = 'min', current_depth=0):
                if not isinstance(d, Mapping):
                    return current_depth
                depths = [
                    _get_depth(d[key], current_depth=current_depth+1, mode=mode)
                    for key in d.keys()
                ]
                if len(depths) == 0:
                    return current_depth
                if mode == 'min':
                    return min(depths)
                elif mode == 'max':
                    return max(depths)
                else:
                    raise KeyError(f'Invalid mode: {mode}. Valid modes are: min, max')

            return _get_depth(d, mode=mode.lower())

        def get_labels(d, depth=None):

            def _get_labels(d, found_labels: dict[int, set], depth=1) -> list[list]:
                if depth < 1 or not isinstance(d, Mapping):
                    return
                labels = set(d.keys())
                try:
                    # found_labels[depth].update(labels)
                    if found_labels[depth] != labels:
                        raise RuntimeError(
                            f'Incompatible labels: {found_labels[depth]}, {labels}'
                        )
                except KeyError:
                    found_labels[depth] = labels
                for label in labels:
                    _get_labels(d[label], found_labels=found_labels, depth=depth-1)

            max_depth = get_depth(d)
            if depth is None:
                depth = max_depth
            elif depth > max_depth:
                raise ValueError(f'Invalid depth: {depth}. Maximum depth is {max_depth}')
            labels_dict = {}
            _get_labels(d, found_labels=labels_dict, depth=depth)
            return [list(labels_dict[i+1]) for i in reversed(range(depth))]

        n_levels = get_depth(items)
        if levels is None:
            levels = [f'level{i+1}' for i in range(n_levels)]

        # initialize empty
        output = cls(levels=levels)

        # iterate over key combinations and add items
        for keys in product(*get_labels(items, depth=len(output.levels))):
            item = items
            for key in keys:
                item = item[key]
            output[keys] = item

        return output

    def __str__(self) -> str:
        return self._str_helper(components=[list(self.levels.values())])

    def __setitem__(self, keys: tuple, item: T):
        self.levels.update_levels(keys)
        items = self.items
        for key in keys[:-1]:
            items = items[key]
        items[keys[-1]] = item

    def __getitem__(self, keys: tuple) -> Union[NestedItems, T]:
        keys = self.levels.check_keys(keys, strict=False)
        items = self.items
        for key in keys:
            items = items[key]

        if len(keys) < len(self.levels):
            return self.from_items(items, levels=list(self.levels.keys())[len(keys):])
        else:
            return items

    def apply_func(self, func: Callable[[T], U], **init_kwargs):
        result: NestedItems[U] = type(self)(levels=self.levels, **init_kwargs)
        for keys in self.levels.generate_all_keys():
            result[keys] = func(self[keys])
        return result

    def apply_funcs(self, funcs: Mapping[str, Callable[[T], U]], level_name='func', **init_kwargs):
        levels = deepcopy(self.levels)
        levels.add_level(level_name)
        result: NestedItems[U] = type(self)(levels=levels, **init_kwargs)
        for func_name, func in funcs.items():
            for keys in self.levels.generate_all_keys():
                new_keys = keys + (func_name,)
                result[new_keys] = func(self[keys])
        return result
