
import os

import numpy as np
import matplotlib.pyplot as plt

from src.utils import load_data_df
from paths import FPATHS, DPATHS

year_group = 5

rename_map = {
    '34-0.0': 'birth year',
    '21003-2.0': 'age'
}

fpath_holdout = FPATHS['data_holdout_clean']
fpath_out = FPATHS['data_age_clean']
fpath_fig = os.path.join(DPATHS['preprocessing'], 'hist_age.png')

if __name__ == '__main__':

    print('----- Parameters -----')
    print(f'year_group:\t{year_group}')
    print(f'rename_map:\t{rename_map}')
    print(f'fpath_holdout:\t{fpath_holdout}')
    print(f'fpath_out:\t{fpath_out}')
    print('----------------------')

    # load holdouts
    df_age = load_data_df(fpath_holdout, encoded=False).rename(columns=rename_map)

    print('Age statistics:')
    print(f'\tMin: {df_age["age"].min()}')
    print(f'\tMax: {df_age["age"].max()}')
    print(f'\tMean: {df_age["age"].mean()}')
    print(f'\tStd: {df_age["age"].std()}')
    print(f'\tMedian: {df_age["age"].median()}')
    print('----------------------')

    # create new column for 5-year period (for stratification)
    df_age['age_group'] = df_age['age'].map(lambda age: age // year_group)
    group_ids = set(df_age['age_group'])

    print(f'Data has {len(group_ids)} age groups: {group_ids}')

    # plot and save age histogram
    fig, ax = plt.subplots(figsize=(4, 4))
    bins = np.arange(5*min(group_ids), 5*(max(group_ids)+2), 5)
    ax.hist(df_age['age'], bins=bins)
    ax.set_title('Age distribution')
    ax.set_xlabel('Age')
    ax.set_ylabel('Count')
    fig.savefig(fpath_fig, dpi=300, bbox_inches='tight')
    print(f'Figure saved: {fpath_fig}')

    # save
    df_age.to_csv(fpath_out, header=True, index=True)
    print(f'Dataframe saved: {fpath_out}')
