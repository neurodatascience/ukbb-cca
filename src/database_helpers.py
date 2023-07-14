
import os
import pandas as pd

class DatabaseHelper():

    def __init__(self, dpath_schema, fpath_udis, 
        fname_field='field.txt', fname_recommended='recommended.txt', 
        fname_catbrowse='catbrowse.txt', fname_category='category.txt'):

        self.field_helper = FieldHelper(dpath_schema, fname_field=fname_field, 
            fname_recommended=fname_recommended, fname_catbrowse=fname_catbrowse)
        self.udi_helper = UDIHelper(fpath_udis)
        self.category_helper = CategoryHelper(dpath_schema, fname_category=fname_category)

        # merge UDI/field/category dataframes
        self.df_merged = self.udi_helper.df_udis.reset_index().merge(
            self.field_helper.df_field.reset_index(), 
            how='inner', on='field_id')
        self.df_merged = self.df_merged.merge(
            self.category_helper.df_category.reset_index(),
            how='inner', left_on='main_category', right_on='category_id', suffixes=('_field', '_category'),
        )
        self.df_merged = self.df_merged.set_index('udi')

    def parse_udis(self, udis):
        return [udi.split('_')[0] for udi in udis]

    def udis_to_text(self, udis, encoded=True):
        output = self.get_info(self.parse_udis(udis), colnames='title_field')
        if encoded:
            for i_udi, udi in enumerate(udis):
                suffix = udi.split('_')[-1]
                if suffix not in ['orig', 'squared']:
                    output[i_udi] = f'{output[i_udi]} ({suffix})'
        return output
        
    def get_info(self, udis, colnames='all'):

        if colnames == 'all':
            return self.df_merged.loc[udis]
        else:
            return self.df_merged.loc[udis, colnames]

    def get_udis_from_categories_and_fields(self, category_ids, fields=None,
        value_types='all', keep_value_types=True,
        title_substring=None, keep_title_substring=True,
        title_substrings_reject=[],
        instances='all', keep_instance='all'):

        if fields is None:
            fields = []

        # get all fields from list of categories
        fields.extend(self.field_helper.get_fields_from_categories(category_ids))

        # keep unique only
        fields = list(set(fields))

        # filter fields by value type
        if value_types != 'all':
            fields = self.field_helper.filter_by_value_type(fields, value_types, keep=keep_value_types)

        # filter fields by title
        if title_substring is not None:
            fields = self.field_helper.filter_by_title(fields, title_substring, keep=True)

        for title_substring_reject in title_substrings_reject:
            fields = self.field_helper.filter_by_title(fields, title_substring_reject, keep=False)

        # get all UDIs from fields
        udis = self.udi_helper.get_udis_from_fields(fields)

        # filter UDIs by instance
        if not (instances == 'all' and keep_instance == 'all'):
            udis = self.udi_helper.filter_by_instance(udis, instances, keep_instance=keep_instance)

        return udis

    def get_categories_from_udis(self, udis, drop_duplicates=True):
        categories = self.df_merged.loc[udis, 'main_category']
        if drop_duplicates:
            categories = categories.drop_duplicates()
        return categories.tolist()

    def filter_udis_by_category(self, udis, category_ids, keep=True):

        df_merged_subset = self.df_merged.loc[udis]
        selected = df_merged_subset['main_category'].isin(category_ids)

        if not keep:
            selected = ~selected

        return df_merged_subset.loc[selected].index.tolist()

    def filter_udis_by_value_type(self, udis, value_types, keep=True):
        value_types = self.field_helper.parse_value_types(value_types)
        df_merged_subset = self.df_merged.loc[udis]
        selected = df_merged_subset['value_type'].isin(value_types)

        if not keep:
            selected = ~selected

        return df_merged_subset.loc[selected].index.tolist()

    def filter_udis_by_field(self, udis, fields, keep=True):
        return self.udi_helper.filter_by_field(udis, fields, keep=keep)

    def get_category_title(self, category_id):
        return self.category_helper.get_info([category_id], colnames='title').tolist()[0]

class FieldHelper():

    def __init__(self, dpath_schema, fname_field='field.txt', fname_recommended='recommended.txt', fname_catbrowse='catbrowse.txt'):
        
        self.dpath_schema = dpath_schema

        self.fpath_field = os.path.join(dpath_schema, fname_field)
        self.fpath_recommended = os.path.join(dpath_schema, fname_recommended)
        self.fpath_catbrowse = os.path.join(dpath_schema, fname_catbrowse)

        self.df_field = pd.read_csv(self.fpath_field, sep='\t', index_col='field_id')
        self.df_recommended = pd.read_csv(self.fpath_recommended, sep='\t')
        self.df_catbrowse = pd.read_csv(self.fpath_catbrowse, sep='\t')

    def get_info(self, fields, colnames='all'):

        if colnames == 'all':
            return self.df_field.loc[fields]
        else:
            return self.df_field.loc[fields, colnames]

    def get_fields_from_categories(self, category_ids):

        # helper function that assumes category_id does not have any subcategories
        def get_fields_from_category(category_id):
            fields = set(self.df_field.loc[self.df_field['main_category'] == category_id].index)
            fields.update(self.df_recommended.loc[self.df_recommended['category_id'] == category_id, 'field_id'])
            return list(fields)

        category_ids = set(category_ids)
        fields = []

        while len(category_ids) != 0:
            category_id = category_ids.pop()
            category_ids.update(self.df_catbrowse.loc[self.df_catbrowse['parent_id'] == category_id, 'child_id'])
            fields.extend(get_fields_from_category(category_id))

        return list(set(fields)) # removes duplicates

    def filter_by_title(self, fields, substring, keep=True):

        df_field_subset = self.df_field.loc[fields]
        selected = df_field_subset['title'].str.contains(substring)

        if not keep:
            selected = ~selected

        return df_field_subset.loc[selected].index.tolist()

    def parse_value_types(self, value_types):

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
        value_types_categorical = {21, 22}

        if value_types == 'all':
            value_types = value_types_all
        elif value_types == 'numeric':
            value_types = value_types_numeric
        elif value_types == 'categorical':
            value_types = value_types_categorical
        else:
            diff = set(value_types) - value_types_all

            if len(diff) != 0:
                raise ValueError(f'Invalid value types {diff}. Valid calue types are {value_types_all}')

        return value_types

    def filter_by_value_type(self, fields, value_types, keep=True):

        value_types = self.parse_value_types(value_types)

        df_field_subset = self.df_field.loc[fields]
        selected = df_field_subset['value_type'].isin(value_types)

        if not keep:
            selected = ~selected

        return df_field_subset.loc[selected].index.tolist()

class UDIHelper():

    def __init__(self, fpath_udis):

        self.fpath_udis = fpath_udis
        self.df_udis = pd.read_csv(self.fpath_udis, index_col='udi')

    def get_info(self, udis, colnames='all'):

        if colnames == 'all':
            return self.df_udis.loc[udis]
        else:
            return self.df_udis.loc[udis, colnames]

    def get_udis_from_fields(self, fields):
        return self.df_udis.loc[self.df_udis['field_id'].isin(fields)].index.tolist()

    def filter_by_instance(self, udis, instances, keep_instance='all'):

        instances_all = {0, 1, 2, 3}

        if instances == 'all':
            instances = instances_all
        else:
            diff = set(instances) - instances_all
            if len(diff) != 0:
                raise ValueError(f'Invalid value types {diff}. Valid calue types are {instances_all}')

        df_udis_subset = self.df_udis.loc[udis]

        df_to_keep = df_udis_subset.loc[df_udis_subset['instance'].isin(instances)]

        if keep_instance == 'all':
            return df_to_keep.index.tolist()
        elif keep_instance == 'max':
            selected = df_to_keep.groupby(['field_id', 'array_index'])['instance'].transform(max) == df_to_keep['instance']
            return df_to_keep.loc[selected].index.tolist()
        else:
            raise ValueError(f'keep_instance="{keep_instance}" is invalid')

    def filter_by_field(self, udis, fields, keep=True):

        df_udis_subset = self.df_udis.loc[udis]
        selected = df_udis_subset['field_id'].isin(fields)

        if not keep:
            selected = ~selected

        return df_udis_subset.loc[selected].index.tolist()

class CategoryHelper():

    def __init__(self, dpath_schema, fname_category='category.txt'):

        self.dpath_schema = dpath_schema
        self.fpath_category = os.path.join(dpath_schema, fname_category)
        self.df_category = pd.read_csv(self.fpath_category, sep='\t', index_col='category_id')

    def get_info(self, categories, colnames='all'):

        if colnames == 'all':
            return self.df_category.loc[categories]
        else:
            return self.df_category.loc[categories, colnames]
            