
import os
import pandas as pd

class FieldSelector():

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

        return fields

    def get_fields_from_categories(self, category_ids):

        category_ids = set(category_ids)
        fields = set()

        while len(category_ids) != 0:
            category_id = category_ids.pop()
            category_ids.update(self.df_catbrowse.loc[self.df_catbrowse['parent_id'] == category_id, 'child_id'])
            fields.update(self.__get_fields_from_category(category_id))

        return fields

    def filter_by_value(self, fields, value, keep=True):

        # TODO
        pass

class UDISelector():

    def __init__(self, fpath_UDIs):

        self.fpath_UDIs = fpath_UDIs
        self.df_UDIs = pd.read_csv(self.fpath_UDIs)

    def get_UDIs_from_fields(self, fields, instances='all'):

        udis = self.df_UDIs.loc[self.df_UDIs['field_id'].isin(fields), 'udi'].tolist()

        if instances != 'all':
            udis = self.filter_by_instance(udis, instances)

        return udis

    def filter_by_instance(self, udis, instances):

        return self.df_UDIs.loc[self.df_UDIs['udi'].isin(udis) & self.df_UDIs['instance_id'].isin(instances), 'udis'].tolist()
