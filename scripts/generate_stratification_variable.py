
import os

import numpy as np
import matplotlib.pyplot as plt

from src.utils import load_data_df
from paths import FPATHS, DPATHS

age_udi = '21003-2.0'
year_group = 5

fpath_holdout = FPATHS['data_holdout_clean']
fpath_out = FPATHS['data_stratification_clean']
fpath_fig = os.path.join(DPATHS['preprocessing'], 'hist_age.png')

if __name__ == '__main__':

    print('----- Parameters -----')
    print(f'year_group:\t{year_group}')
    print(f'fpath_holdout:\t{fpath_holdout}')
    print(f'fpath_out:\t{fpath_out}')
    print('----------------------')

    # load holdouts
    df_holdout = load_data_df(fpath_holdout, encoded=False)

    print(f'{age_udi} statistics:')
    print(f'\tMin: {df_holdout[age_udi].min()}')
    print(f'\tMax: {df_holdout[age_udi].max()}')
    print(f'\tMean: {df_holdout[age_udi].mean()}')
    print(f'\tStd: {df_holdout[age_udi].std()}')
    print(f'\tMedian: {df_holdout[age_udi].median()}')
    print('----------------------')

    # get age groups (for stratification) (pandas Series)
    age_groups = df_holdout[age_udi].map(lambda age: age // year_group)
    age_groups.name = 'age_group'
    group_ids = set(age_groups) # unique

    print(f'Data has {len(group_ids)} age groups: {group_ids}')

    # plot and save age histogram
    fig, ax = plt.subplots(figsize=(4, 4))
    bins = np.arange(5*min(group_ids), 5*(max(group_ids)+2), 5)
    ax.hist(df_holdout[age_udi], bins=bins)
    ax.set_title('Age distribution')
    ax.set_xlabel('Age')
    ax.set_ylabel('Count')
    fig.savefig(fpath_fig, dpi=300, bbox_inches='tight')
    print(f'Figure saved: {fpath_fig}')

    # save
    age_groups.to_csv(fpath_out, header=True, index=True)
    print(f'Dataframe saved: {fpath_out}')
