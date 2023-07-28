from __future__ import annotations
from copy import deepcopy
import re
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import pandas as pd
from scipy import stats

from .database_helpers import DatabaseHelper
from .base import _BaseData
from .plotting import plot_na_histograms, plot_group_histograms, save_fig
from .utils import add_suffix, make_parent_dir, zscore_df, load_data_df, select_rows

FILE_EXT = '.csv'
SUFFIX_CLEAN = 'clean'
MULTIINDEX_NAMES = ['udi', 'udi_encoded']

PREFIX_HOLDOUT = 'holdouts'
PREFIX_AGE_GROUP = 'age_group'
PREFIX_DROPPED_SUBJECTS = 'dropped_subjects'
PREFIX_DROPPED_UDIS = 'dropped_udis'

UDI_AGE = '21003-2.0'

# plotting
PREFIX_FIG_NA = 'hist_na'
PREFIX_FIG_AGE_GROUP = f'hist_{PREFIX_AGE_GROUP}'

def parse_udis(fpath_in) -> pd.DataFrame:

    def parse_udi(udi):
        match = re_udi.match(udi)
        if not match:
            raise ValueError(f'Could not parse UDI {udi}')
        else:
            return match.groups()

    re_udi = re.compile('(\d+)-(\d+)\\.(\d+)')

    # read UDIs (column names) from data
    # faster to read 1 row then drop it than reading 0 rows
    df_tabular = load_data_df(fpath_in, nrows=1)
    df_udis = df_tabular.iloc[:0].transpose()
    
    df_udis.index.name = 'udi'
    df_udis.columns.name = None

    df_udis = df_udis.reset_index()
    df_udis['field_id'], df_udis['instance'], df_udis['array_index'] =  zip(*df_udis['udi'].map(parse_udi))

    return df_udis

def write_subset(fpath_data, fpath_out, colnames=None, fn_to_apply=None, header=0, index_col='eid', chunksize=10000):
    
    if (colnames is not None) and (index_col not in colnames):
        colnames.insert(0, index_col) # passing index_col to pd.read_csv() directly causes KeyError when writing

    df_chunks = pd.read_csv(
        fpath_data, usecols=colnames, header=header, chunksize=chunksize, low_memory=False)

    mode = 'w' # first chunk
    write_header=True

    n_rows = 0
    n_cols = 0

    make_parent_dir(fpath_out)

    for df_chunk in df_chunks:

        df_chunk = df_chunk.set_index(index_col)

        if not (fn_to_apply is None):
            df_chunk = fn_to_apply(df_chunk)

        n_rows += df_chunk.shape[0]
        n_cols = df_chunk.shape[1]

        df_chunk.to_csv(fpath_out, mode=mode, header=write_header, index=True)
        mode = 'a' # append subsequent chunks
        write_header=False

    return n_rows, n_cols

def find_cols_with_missing(df: pd.DataFrame, threshold=0.5) -> list:
    na_freqs = df.isna().mean(axis='index')
    return df.columns[na_freqs >= threshold].tolist()

def find_cols_with_high_freq(df: pd.DataFrame, threshold=0.95) -> list:
    n_rows_nonan = (~df.isna()).sum(axis='index').values
    _, mode_counts = np.squeeze(stats.mode(df, axis=0))
    return df.columns[(mode_counts / n_rows_nonan) >= threshold].tolist() 

def find_cols_with_outliers(df: pd.DataFrame, threshold=100) -> list:

    def has_outliers(s):
        # outlier if max((X - median(X))**2) > 100*mean((X - median(X))**2)
        squared_distances_from_median = (s - s.median()) ** 2
        ratio = squared_distances_from_median.max() / squared_distances_from_median.mean()
        return ratio >= threshold

    df = inv_norm_df(df)
    df = zscore_df(df)

    return df.columns[df.apply(has_outliers, axis='index')].tolist()

def find_cols_with_large(df: pd.DataFrame, threshold=None) -> list:
    is_large = (df > threshold).any(axis='index')
    return df.columns[is_large].tolist()

def average_fields(df: pd.DataFrame, fields: list[str], multiindex_names: list[str], db_helper: DatabaseHelper) -> pd.DataFrame:
    
    dfs_averaged = {}
    udis_averaged = set()
    for field in fields:
        udis_to_average = db_helper.filter_udis_by_field(df.columns, [field])

        if len(udis_to_average) == 0:
            continue

        instances = {udi.split('-')[-1].split('.')[0] for udi in udis_to_average}
        array_indices = sorted([udi.split('.')[-1] for udi in udis_to_average])

        if len(instances) == 1:
            instance = instances.pop()
        else:
            raise NotImplementedError(f'Multiple instances found for field {field}: {instances}')
        new_udi = f'{field}-{instance}.{array_indices[0]}' # use smallest available

        s_mean = df[udis_to_average].mean(axis='columns')
        s_std = df[udis_to_average].std(axis='columns')
        dfs_averaged[new_udi] = pd.concat(
            [s_mean, s_std],
            axis='columns',
            keys=[f'{new_udi}_mean{len(udis_to_average)}', f'{new_udi}_std{len(udis_to_average)}'],
        )

        udis_averaged.update(udis_to_average)

    if len(dfs_averaged) == 0:
        df_averaged = None
    else:
        df_averaged = pd.concat(dfs_averaged, axis='columns', names=multiindex_names)
    return df_averaged, udis_averaged

def one_hot_encode(df: pd.DataFrame, multiindex_names: list[str], keep=None) -> pd.DataFrame:

    def fn_rename(colname):
        components = colname.split('.')
        if len(components) != 2:
            return '.'.join(components[:-1]) # remove trailing floating point
        else:
            return colname

    dfs_encoded = {} # to be concatenated

    for colname_original in df.columns:

        df_encoded = pd.get_dummies(df[colname_original], prefix=colname_original, prefix_sep='_')
        df_encoded = df_encoded.rename(columns=fn_rename)

        if (keep is not None) and (colname_original in keep):
            df_encoded = df_encoded[[keep[colname_original]]]
        
        dfs_encoded[colname_original] = df_encoded

    return pd.concat(dfs_encoded, axis='columns', names=multiindex_names)

def square_df(df: pd.DataFrame, multiindex_names: list[str]) -> pd.DataFrame:
    df = np.square(df)
    # change suffix to "squared"
    df.columns = df.columns.remove_unused_levels().set_levels(
        df.columns.map(lambda x: f'{x[1].split("_")[0]}_squared'),
        level=multiindex_names[1],
    )
    return df

def remove_bad_cols(df: pd.DataFrame, threshold_na=0.5, threshold_high_freq=0.95, threshold_outliers=100, threshold_large=None, return_colnames=False):

    # identify columns with too much missing data
    cols_with_missing = find_cols_with_missing(df, threshold=threshold_na)
    df = df.drop(columns=cols_with_missing)
    print(f'\t\tRemoved {len(cols_with_missing)} column(s) with too much missing data')

    # identify columns with not enough variability
    cols_without_variability = find_cols_with_high_freq(df, threshold=threshold_high_freq)
    df = df.drop(columns=cols_without_variability)
    print(f'\t\tRemoved {len(cols_without_variability)} column(s) with not enough variability')

    # identify columns with outliers
    cols_with_outliers = find_cols_with_outliers(df, threshold=threshold_outliers)
    print(f'\t\tRemoved {len(cols_with_outliers)} column(s) with outlier(s)')
    df = df.drop(columns=cols_with_outliers)

    cols_with_large = find_cols_with_large(df, threshold=threshold_large)
    print(f'\t\tRemoved {len(cols_with_large)} column(s) with large value(s)')
    df = df.drop(columns=cols_with_large)

    if return_colnames:
        bad_colnames = (
            set(cols_with_missing) | 
            set(cols_without_variability) |
            set(cols_with_outliers) |
            set(cols_with_large)
        )
        return df, list(bad_colnames)

    return df

def generate_fname_data(dataset_name, clean=False):
    fname = Path(dataset_name).with_suffix(FILE_EXT)
    if clean:
        fname = add_suffix(fname, SUFFIX_CLEAN)
    return fname

# def find_subjects_with_outliers(df: pd.DataFrame, k=3):
#     df_z_score = (df - df.mean(axis='index')) / df.std(axis='index')
#     return (df_z_score.abs() > k).any(axis='columns').index

def clean_datasets(
    db_helper: DatabaseHelper,
    dpath_data,
    dpath_figs,
    domains: list[str],
    domains_extra: list[str],
    holdout_fields: list[int] = None,
    square_conf: bool = True,
    domains_to_square: list[str] = None,
    fields_to_average: list[str] = None,
    one_hot_encoded_udis_keep: dict[str, ] = None, # TODO
    threshold_na=0.5,
    threshold_high_freq=0.95,
    threshold_outliers=100,
    threshold_large=10000000,
):

    if holdout_fields is None:
        holdout_fields = []
    if domains_to_square is None:
        domains_to_square = []

    fpath_holdout = Path(dpath_data, generate_fname_data(PREFIX_HOLDOUT, clean=True))
    fpath_dropped_udis = Path(dpath_data, generate_fname_data(PREFIX_DROPPED_UDIS))
    fpath_dropped_subjects = Path(dpath_data, generate_fname_data(PREFIX_DROPPED_SUBJECTS))

    fpaths_map = {}
    dfs_data = {}
    dfs_holdout = [] # to be concatenated
    subjects_to_drop = set()

    print('----- Loading unprocessed data -----')
    for domain in domains:

        fpath_data = Path(dpath_data, generate_fname_data(domain))
        fpaths_map[domain] = fpath_data
        print(f'{domain}: {fpath_data}')

        # load data
        df_data = load_data_df(fpath_data)
        print(f'\tDataframe shape: {df_data.shape}')

        # if data contains holdout UDIs, extract them and drop them from dataframe
        holdout_udis = db_helper.filter_udis_by_field(df_data.columns, holdout_fields)
        if len(holdout_udis) > 0:

            print(f'\tExtracting {len(holdout_udis)} holdout variables')
            df_holdout = df_data.loc[:, holdout_udis]

            # remove any subject with missing holdout data
            n_missing = df_holdout.isna().sum(axis='columns')
            subjects_with_missing = n_missing.loc[n_missing > 0].index
            print(f'\t\tRemoving {len(subjects_with_missing)} subjects with no holdout data')
            subjects_to_drop.update(subjects_with_missing)
            dfs_holdout.append(df_holdout) # to save later
            df_data = df_data.drop(columns=holdout_udis)

        # replace some columns by their mean/standard deviation value
        if fields_to_average is not None and len(fields_to_average) > 0:
            df_averaged, udis_averaged = average_fields(df_data, fields_to_average, MULTIINDEX_NAMES, db_helper)

            # drop columns that were averaged
            print(f'\t\tDropping {len(udis_averaged)} columns that were averaged')
            df_data = df_data.drop(columns=udis_averaged)
        else:
            df_averaged = None

        # one-hot encode categorical variables
        udis = df_data.columns
        categorical_udis = db_helper.filter_udis_by_value_type(udis, 'categorical')
        if len(categorical_udis) > 0:

            print(f'\tOne-hot encoding {len(categorical_udis)} categorical UDIs')
            df_encoded = one_hot_encode(df_data.loc[:, categorical_udis], MULTIINDEX_NAMES, keep=one_hot_encoded_udis_keep) # returns multiindexed df

            # drop original categorical columns
            df_data = df_data.drop(columns=categorical_udis)
        else:
            df_encoded = None

        # add column level for compatibility with df_encoded
        df_data.columns = pd.MultiIndex.from_arrays(
            [df_data.columns, df_data.columns.map(lambda colname: f'{colname}_orig')],
            names=MULTIINDEX_NAMES,
        )
        
        df_data = pd.concat([df_data, df_encoded, df_averaged], axis='columns')
        print(f'\t\tShape after one-hot encoding: {df_data.shape}')

        # square confounders
        if square_conf and domain in domains_to_square:
            # only square non-categorical (i.e., integer/continuous) columns
            non_categorical_udis = db_helper.filter_udis_by_value_type(udis, [11, 31])
            print(f'\tSquaring {len(non_categorical_udis)} numerical, non-categorical columns')
            df_data = pd.concat([df_data, square_df(df_data.loc[:, non_categorical_udis], MULTIINDEX_NAMES)], axis='columns')
            print(f'\t\tShape after squaring: {df_data.shape}')

        # plot histograms of row-/column-wise NaN frequencies
        fig, freqs_na_row, freqs_na_col = plot_na_histograms(df_data, return_freqs=True)
        fig.subplots_adjust(top=0.85)
        fig.suptitle(f'{domain.capitalize()} dataset {df_data.shape}')

        # save figure
        fpath_fig = Path(dpath_figs, f'{PREFIX_FIG_NA}_{domain}_before.png')
        make_parent_dir(fpath_fig)
        save_fig(fig, fpath_fig)

        # add rows to drop later
        subjects_to_drop.update(freqs_na_row.loc[np.isclose(freqs_na_row, 1)].index)

        dfs_data[domain] = df_data

    if len(dfs_holdout) == 0:
        df_holdout = None
    else:
        df_holdout = pd.concat(dfs_holdout, axis='columns')

    # load extra dataframes
    # don't do anything else
    print('----- Loading extra data (will keep all columns) -----')
    for domain in domains_extra:
        fpath_extra = Path(dpath_data, generate_fname_data(domain))
        fpaths_map[domain] = fpath_extra
        df_extra = load_data_df(fpath_extra)
        print(f'{domain}: {df_extra.shape}')
        dfs_data[domain] = df_extra

    print('----- Cleaning data -----')
    mode = 'w'
    dfs_dropped_cols = []
    while len(subjects_to_drop) != 0:

        # write dropped subject IDs
        pd.Series(list(subjects_to_drop), name='eid').to_csv(fpath_dropped_subjects, 
            header=True, index=False, mode=mode)
        mode = 'a' # append for subsequent iterations

        # drop subjects from holdouts dataframe
        if df_holdout is not None:
            df_holdout = df_holdout.drop(index=subjects_to_drop)

        # drop subjects from extra dataframes
        for domain in domains_extra:
            dfs_data[domain] = dfs_data[domain].drop(index=subjects_to_drop)

        subjects_to_drop_new = set()
        for domain in domains:

            print(f'{domain}')

            # remove bad rows
            print(f'\tRemoving {len(subjects_to_drop)} rows')
            df_clean = dfs_data[domain].drop(index=subjects_to_drop)

            # # NOTE comment this
            # # remove bad columns
            # print('\tLooking for bad columns...')
            # df_clean, bad_cols = remove_bad_cols(df_clean, return_colnames=True,
            #     threshold_na=threshold_na, threshold_high_freq=threshold_high_freq, 
            #     threshold_outliers=threshold_outliers, threshold_large=threshold_large,
            # )

            # # save in df, to write later
            # bad_cols = pd.MultiIndex.from_tuples(bad_cols, names=df_clean.columns.names)
            # df_dropped_cols = pd.DataFrame({name: bad_cols.get_level_values(name) for name in bad_cols.names})
            # df_dropped_cols['domain'] = domain
            # dfs_dropped_cols.append(df_dropped_cols)
            # # NOTE end comment

            print(f'\tDataframe shape after removing bad rows/columns: {df_clean.shape}')

            # report NaN count/percentage
            na_count = df_clean.isna().values.sum()
            print(f'\tNaN count: {na_count} ({100*na_count/df_clean.size:.0f}%)')

            # plot histograms of row-/column-wise NaN frequencies
            fig, freqs_na_row, freqs_na_col = plot_na_histograms(df_clean, return_freqs=True)
            fig.subplots_adjust(top=0.85)
            fig.suptitle(f'{domain.capitalize()} dataset {df_clean.shape}')

            # add rows to drop later, if any
            subjects_to_drop_new.update(freqs_na_row.loc[np.isclose(freqs_na_row, 1)].index)

            # save figure
            fpath_fig = Path(dpath_figs, f'{PREFIX_FIG_NA}_{domain}_after.png')
            save_fig(fig, fpath_fig)

            dfs_data[domain] = df_clean # to be saved in csv file

        subjects_to_drop = subjects_to_drop_new
        print('-------------------------')

    if df_holdout is not None:
        print(f'Holdout dataframe shape: {df_holdout.shape}')

    # save cleaned data
    for domain in dfs_data.keys():
        print(f'{domain} dataframe shape: {dfs_data[domain].shape}')
        fpath_out = Path(dpath_data, generate_fname_data(domain, clean=True))
        dfs_data[domain].to_csv(fpath_out, header=True, index=True)
        print(f'Saved {domain} data to {fpath_out}')

    # save cleaned holdouts dataframe
    if df_holdout is not None:
        df_holdout.to_csv(fpath_holdout, header=True, index=True)

    # log dropped columns
    if len(dfs_dropped_cols) > 0:
        df_dropped_cols = pd.concat(dfs_dropped_cols)
    else:
        df_dropped_cols = pd.DataFrame()
    df_dropped_cols.to_csv(fpath_dropped_udis, header=True, index=False)

def get_age_groups_from_holdouts(dpath, udi_age=UDI_AGE, year_step=5, plot=False, dpath_figs='.', alternative_dataset=None):
    dpath = Path(dpath)
    dpath_figs = Path(dpath_figs)
    fpath_holdout = dpath / generate_fname_data(PREFIX_HOLDOUT, clean=True)
    encoded = False

    if not fpath_holdout.exists():
        if alternative_dataset is None:
            raise RuntimeError(f'Holdout file ({fpath_holdout}) not found and no alternative file was provided')
        else:
            fpath_holdout = dpath / generate_fname_data(alternative_dataset, clean=True)
            encoded = True
            if not fpath_holdout.exists():
                raise RuntimeError(f'Alternative file ({fpath_holdout}) not found')

    df_holdout = load_data_df(fpath_holdout, encoded=encoded)

    print(f'{udi_age} statistics:')
    print(f'\tMin: {df_holdout[udi_age].min()}')
    print(f'\tMax: {df_holdout[udi_age].max()}')
    print(f'\tMean: {df_holdout[udi_age].mean()}')
    print(f'\tStd: {df_holdout[udi_age].std()}')
    print(f'\tMedian: {df_holdout[udi_age].median()}')
    print('----------------------')

    age_groups = df_holdout[udi_age].squeeze().map(lambda age: age // year_step)
    age_groups.name = PREFIX_AGE_GROUP
    group_ids = set(age_groups) # unique

    print(f'Dataset has {len(group_ids)} age groups: {group_ids}')

    # plot and save age histogram
    if plot:
        bins = np.arange(year_step*min(group_ids), year_step*(max(group_ids)+2), year_step)
        fig = plot_group_histograms(df_holdout[udi_age], bins)
        fpath_fig = dpath_figs / PREFIX_FIG_AGE_GROUP
        save_fig(fig, fpath_fig)

    # save
    fpath_out = dpath / generate_fname_data(PREFIX_AGE_GROUP, clean=True)
    age_groups.to_csv(fpath_out, header=True, index=True)
    print(f'Dataframe saved to {fpath_out}')

class XyData(_BaseData):

    fname = 'Xy'
    df_names = ['X', 'group', 'holdout']

    def __init__(
        self,
        dpath,
        dataset_names=None,
        extra_dataset_names=None,
        conf_name=None,
        udi_holdout=None,
        group_name=None,
        column_level_to_drop=None,
        **kwargs,
    ) -> None:

        super().__init__(**kwargs)

        dpath = Path(dpath)
        if column_level_to_drop is None:
            column_level_to_drop = MULTIINDEX_NAMES[0]
        if dataset_names is None:
            dataset_names = []
        if extra_dataset_names is None:
            extra_dataset_names = []

        self.dpath = dpath
        self.dataset_names = dataset_names
        self.extra_dataset_names = extra_dataset_names
        self.conf_name = conf_name
        self.udi_holdout = udi_holdout
        self.group_name = group_name
        self.column_level_to_drop = column_level_to_drop

        self.X: pd.DataFrame = None
        self.extra: dict[str, pd.DataFrame] = {}
        self.group = None
        self.holdout = None

        for dataset_name in dataset_names:
            self.add_dataset(dataset_name)

        for extra_dataset_name in extra_dataset_names:
            self.add_extra_dataset(extra_dataset_name)

        if conf_name is not None:
            self.set_conf(conf_name)

        if udi_holdout is not None:
            self.set_holdout(udi_holdout)

        if group_name is not None:
            self.set_group(group_name)

    def _add_data(self, name, fpath) -> pd.DataFrame:

        print(f'Adding {name} data')
        print(f'\tPath: {fpath}')

        # load data
        df = load_data_df(fpath, encoded=True)

        # make sure all datasets contain the same subjects
        subjects = set(df.index)
        if self.subjects is None:
            self.subjects = np.sort(list(subjects))
        else:
            if subjects != set(self.subjects):
                raise ValueError(f'Datasets do not have exactly the same subjects')

        # drop multiindex level
        df = df.droplevel(self.column_level_to_drop, axis='columns')
        # add level for dataset/conf name
        df.columns = pd.MultiIndex.from_product([[name], df.columns])

        # add to X
        df = df.loc[self.subjects] # make sure subject order is the same
        if self.X is not None:
            self.X = self.X.loc[self.subjects]
        self.X = pd.concat([self.X, df], axis='columns')

        print(f'\tX shape: {self.X.shape}')

        return df

    def check_index_order(self):
        for name, df_extra in self.extra.items():
            if not self.X.index.equals(df_extra.index):
                raise RuntimeError(
                    f'Index of extra dataset {name} does not match index of X'
                )

    def add_dataset(self, name):
        fpath = self.dpath / generate_fname_data(name, clean=True)
        df = self._add_data(name, fpath)
        self.udis_datasets.append(df.columns)
        self.n_features_datasets.append(df.shape[1])

    def add_extra_dataset(self, name):
        fpath = self.dpath / generate_fname_data(name, clean=True)
        print('Adding extra dataset')
        print(f'\tPath: {fpath}')
        df = load_data_df(fpath)
        df = df.loc[self.subjects]
        self.extra[name] = df

    def set_conf(self, name):
        if self.udis_conf is not None:
            raise RuntimeError(f'conf data have already been added')
        fpath = self.dpath / generate_fname_data(name, clean=True)
        df = self._add_data(name, fpath)
        self.udis_conf = df.columns
        self.n_features_conf = df.shape[1]

    def set_group(self, name):
        if self.group is not None:
            raise RuntimeError(f'groups have already been added')
        print(f'Adding group data ({name})')
        fpath = self.dpath / generate_fname_data(name, clean=True)
        print(f'\tPath: {fpath}')
        self.group = load_data_df(fpath, encoded=False).squeeze('columns')[self.subjects]

    def set_holdout(self, udi):
        if self.holdout is not None:
            raise RuntimeError(f'self.holdout already exists')
        fpath = self.dpath / generate_fname_data(PREFIX_HOLDOUT, clean=True)
        
        print(f'Adding holdout data')
        print(f'\tPath: {fpath}')
        self.holdout = load_data_df(fpath).loc[self.subjects, udi]
        self.udi_holdout = udi

    def subset(self, indices):
        data_subset = deepcopy(self)
        for df_name in self.df_names:
            df = getattr(data_subset, df_name)
            if df is None:
                continue
            df = select_rows(df, indices)
            setattr(data_subset, df_name, df)
        data_subset.subjects = data_subset.X.index.to_numpy()
        return data_subset

    def filter_samples(self, fn: Callable[[XyData], Iterable]):
        return self.subset(fn(self))

    def load(self) -> XyData:
        return super().load()

    def __str__(self) -> str:
        components = [
            f'n_features_datasets={self.n_features_datasets}', 
            f'n_features_conf={self.n_features_conf}'
        ]
        for df_name in self.df_names:
            df: pd.DataFrame = getattr(self, df_name)
            if df is not None:
                value = df.shape
            else:
                value = None
            components.append(f'{df_name}={value}')
        
        components.append(f'extra={ {k: v.shape for k, v in self.extra.items()} }')

        return self._str_helper(components)

    # @classmethod
    # def load(cls, dpath):
    #     dpath = Path(dpath)
    #     fpath = dpath / cls.fname
    #     return load_pickle(fpath)

def deconfound_df(df_data, df_conf):

    # TODO add reference for deconfounding method

    # df_conf cannot have missing data because pinv() would fail
    if df_conf.isna().values.sum() != 0:
        raise ValueError('confound data cannot contain NaNs')

    # remember columns/index of original dataframe
    data_cols = df_data.columns
    data_index = df_data.index

    conf = df_conf.values

    # mask NaNs in df_data
    data_masked = np.ma.masked_invalid(df_data.values)

    # deconfound
    conf_weights = np.ma.dot(np.linalg.pinv(conf), data_masked)
    conf_effects = np.ma.dot(conf, conf_weights)
    data_deconfounded = data_masked - conf_effects

    # put back NaNs and rebuild dataframe
    return pd.DataFrame(data=data_deconfounded.filled(np.nan), index=data_index, columns=data_cols)

def inv_norm(X, method='blom', rank_method='average'):
    '''Rank-based inverse normal transformation.'''

    constants = {
        'blom': 3/8, 
        'tukey': 1/3, 
        'bliss': 1/2,
        'waerden': 0,
    }

    try:
        c = constants[method]
    except KeyError:
        raise KeyError(f'Invalid method "{method}". Valid methods are: {constants.keys()}')

    if len(X.shape) > 2:
        raise ValueError(f'Matrix must have at most 2 dimensions (got shape {X.shape})')

    n_rows = X.shape[0]
    try:
        n_cols = X.shape[1]
    except IndexError:
        n_cols = 1

    if np.isnan(X).sum() == 0:

        # get within-column ranks
        ranks = stats.rankdata(X, axis=0, method=rank_method)

        # inverse normal transform
        quantiles = (ranks - c) / (n_rows - (2*c) + 1)
        X_transformed = stats.norm.ppf(quantiles)

    else:

        # initialize matrix of NaNs
        # to be filled with transformed data in the correct positions
        # so that elements that were NaN are still NaN
        X_transformed = np.empty((n_rows, n_cols))
        X_transformed.fill(np.nan)

        for i_col in range(n_cols):

            # find where non-NaN elements
            not_na = ~(np.isnan(X[:, i_col]))

            # call inv_norm() on clean data and fill spots in NaN matrix
            X_transformed[not_na, i_col] = inv_norm(X[not_na, i_col], method=method, rank_method=rank_method)

    return X_transformed

def inv_norm_df(df, method='blom', rank_method='average'):

    # deconstruct dataframe
    cols = df.columns
    index = df.index
    data = df.values

    # reconstruct dataframe
    data_transformed = inv_norm(data, method=method, rank_method=rank_method)
    return pd.DataFrame(data=data_transformed, index=index, columns=cols)
