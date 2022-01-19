
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

    def __get_fields_from_category(self, category_id):

        fields = set(self.df_field.loc[self.df_field['main_category'] == category_id, 'field_id'])
        fields.update(self.df_recommended.loc[self.df_recommended['category_id'] == category_id, 'field_id'])

        return list(fields)

    def get_fields_from_categories(self, category_ids):

        category_ids = set(category_ids)
        fields = []

        while len(category_ids) != 0:
            category_id = category_ids.pop()
            category_ids.update(self.df_catbrowse.loc[self.df_catbrowse['parent_id'] == category_id, 'child_id'])
            fields.extend(self.__get_fields_from_category(category_id))

        return list(set(fields)) # removes duplicates

    def filter_by_value(self, fields, value, colname='title', keep=True):

        if keep:
            fn_condition = (lambda x: value in x)
        else:
            fn_condition = (lambda x: not (value in x))

        df_field_subset = self.df_field.set_index('field_id').loc[fields]
        df_field_filtered = df_field_subset.loc[df_field_subset[colname].map(fn_condition)]

        return df_field_filtered.index.tolist()

class UDIHelper():

    def __init__(self, fpath_udis):

        self.fpath_udis = fpath_udis
        self.df_udis = pd.read_csv(self.fpath_udis)

    def get_udis_from_fields(self, fields, instances='all'):

        udis = self.df_udis.loc[self.df_udis['field_id'].isin(fields), 'udi'].tolist()

        if instances != 'all':
            udis = self.filter_by_instance(udis, instances)

        return udis

    def filter_by_instance(self, udis, instances):

        return self.df_udis.loc[self.df_udis['udi'].isin(udis) & self.df_udis['instance'].isin(instances), 'udis'].tolist()
