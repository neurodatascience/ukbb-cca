
import os
import pandas as pd

class FieldHelper():

    def __init__(self, dpath_schema, fname_field='field.txt', fname_recommended='recommended.txt', fname_catbrowse='catbrowse.txt'):
        
        self.dpath_schema = dpath_schema

        self.fpath_field = os.path.join(dpath_schema, fname_field)
        self.fpath_recommended = os.path.join(dpath_schema, fname_recommended)
        self.fpath_catbrowse = os.path.join(dpath_schema, fname_catbrowse)

        self.df_field = pd.read_csv(self.fpath_field, sep='\t')
        self.df_recommended = pd.read_csv(self.fpath_recommended, sep='\t')
        self.df_catbrowse = pd.read_csv(self.fpath_catbrowse, sep='\t')

    def get_info(self, fields, colnames='all'):

        if colnames == 'all':
            return self.df_field.set_index('field_id').loc[fields]
        else:
            return self.df_field.set_index('field_id').loc[fields, colnames]

    def __get_fields_from_category(self, category_id, value_types='all'):

        # 11: integer
        # 21: categorical (single)
        # 22: categorical (multiple)
        # 31: continuous
        # 41: text
        # 51: date
        # 61: time
        # 101: compound
        value_types_all = {11, 21, 22, 31, 41, 51, 61, 101}
        value_types_numeric = {11, 21, 22, 31}

        if value_types == 'all':
            value_types = value_types_all
        elif value_types == 'numeric':
            value_types = value_types_numeric
        else:
            diff = set(value_types) - value_types_all

            if len(diff) != 0:
                raise ValueError(f'Invalid value types {diff}. Valid calue types are {value_types_all}')

        fields = set(self.df_field.loc[self.df_field['main_category'] == category_id, 'field_id'])
        fields.update(self.df_recommended.loc[self.df_recommended['category_id'] == category_id, 'field_id'])

        fields = self.filter_by_list(fields, value_types, 'value_type')

        return list(fields)

    def get_fields_from_categories(self, category_ids, value_types='all'):

        category_ids = set(category_ids)
        fields = []

        while len(category_ids) != 0:
            category_id = category_ids.pop()
            category_ids.update(self.df_catbrowse.loc[self.df_catbrowse['parent_id'] == category_id, 'child_id'])
            fields.extend(self.__get_fields_from_category(category_id, value_types=value_types))

        return list(set(fields)) # removes duplicates

    def filter_by_condition(self, fields, fn_condition, colname):

        df_field_subset = self.df_field.set_index('field_id').loc[fields]
        df_field_filtered = df_field_subset.loc[df_field_subset[colname].map(fn_condition)]

        return df_field_filtered.index.tolist()

    def filter_by_value(self, fields, value, colname='title', check_inclusion=False, keep=True):

        if check_inclusion:
            fn_condition = (lambda x: value in x)
        else:
            fn_condition = (lambda x: value == x) # check equality

        if not keep:
            fn_condition = (lambda x: not fn_condition(x))

        return self.filter_by_condition(fields, fn_condition, colname)

    def filter_by_list(self, fields, value_list, colname, keep=True):

        if keep:
            fn_condition = lambda x: x in value_list
        else:
            fn_condition = lambda x : x not in value_list

        return self.filter_by_condition(fields, fn_condition, colname)

class UDIHelper():

    def __init__(self, fpath_udis):

        self.fpath_udis = fpath_udis
        self.df_udis = pd.read_csv(self.fpath_udis)

    def get_info(self, udis, colnames='all'):

        if colnames == 'all':
            return self.df_udis.set_index('udi').loc[udis]
        else:
            return self.df_udis.set_index('udi').loc[udis, colnames]

    def get_udis_from_fields(self, fields, instances='all', keep_instance='all'):

        udis = self.df_udis.loc[self.df_udis['field_id'].isin(fields), 'udi'].tolist()

        if instances != 'all':
            udis = self.filter_by_instance(udis, instances, keep_instance=keep_instance)

        return udis

    def filter_by_instance(self, udis, instances, keep_instance='all'):

        df_to_keep = self.df_udis.loc[self.df_udis['udi'].isin(udis) & self.df_udis['instance'].isin(instances)]

        if keep_instance == 'all':
            return df_to_keep.loc[:, 'udi'].tolist()
        elif keep_instance == 'max':
            idx = df_to_keep.groupby(['field_id', 'array_index'])['instance'].transform(max) == df_to_keep['instance']
            return df_to_keep.loc[idx, 'udi'].tolist()
        else:
            raise ValueError(f'keep_instance="{keep_instance}" is invalid')

    def filter_by_field(self, udis, fields, keep=True):

        df_fields_subset = self.get_info(udis, colnames='field_id')
        to_keep = df_fields_subset.isin(fields)

        if not keep:
            to_keep = not to_keep

        return df_fields_subset.loc[to_keep].index.tolist()
            