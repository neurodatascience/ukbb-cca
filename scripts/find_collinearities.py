
import pickle
import sys
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.linalg import qr

from src.data_selection import FieldHelper, UDIHelper, CategoryHelper
from paths import FPATHS, DPATHS

def get_coefs(V, idx_small):

    # QR factorization on columns V that are 
    # associated with small singular values
    Q, R, e = qr(V[:, idx_small].T, pivoting=True)
    
    R1 = R[:, :n_small]
    R2 = R[:, n_small:]
    coefs = - R2.T @ np.linalg.inv(R1.T)

    return coefs, e # e is permutation array

if __name__ == '__main__':

    within_only = True # if True, ignores 'all' category
    save_figures = True

    fpath_udis = FPATHS['udis_tabular_raw']
    dpath_schema = DPATHS['schema']
    dpath_figs = DPATHS['collinearity']

    if len(sys.argv) != 2:
        raise ValueError(f'Usage: {sys.argv[0]} <domain>')

    domain = sys.argv[1]

    valid_domains = ['behavioural', 'brain', 'demographic']
    if domain not in valid_domains:
        raise ValueError(f'domain "{domain}" not in {valid_domains}')

    fpath_svd = FPATHS[f'res_{domain}_svd']
    fpath_out_csv = FPATHS[f'res_{domain}_collinear']
    fpath_out_pkl = FPATHS[f'res_{domain}_collinear_coefs']

    print('----- Parameters -----')
    print(f'domain:\t{domain}')
    print(f'within_only:\t{within_only}')
    print(f'save_figures:\t{save_figures}')
    print(f'fpath_svd:\t{fpath_svd}')
    print(f'fpath_out_csv:\t{fpath_out_csv}')
    print(f'fpath_out_pkl:\t{fpath_out_pkl}')
    print(f'fpath_udis:\t{fpath_udis}')
    print(f'dpath_schema:\t{dpath_schema}')
    print(f'dpath_figs:\t{dpath_figs}')
    print('----------------------')

    field_helper = FieldHelper(dpath_schema)
    udi_helper = UDIHelper(fpath_udis)
    category_helper = CategoryHelper(dpath_schema)

    # load saved SVD output
    with open(fpath_svd, 'rb') as file_in:
        results_svd = pickle.load(file_in)

    # initialize results
    dfs_collinear = [] # UDIs of columns to remove
    results_coefs = {} # linear combination coefficients for collinear columns

    for category in results_svd:

        if within_only and category == 'all':
            continue

        try:
            category_desc = category_helper.get_info([category], colnames='title').tolist()[0]
        except KeyError:
            category_desc = 'All available categories'

        udis = results_svd[category]['udis']
        s = results_svd[category]['s']
        V = results_svd[category]['VT'].T

        idx_small = np.where(np.isclose(s, 0))[0]
        n_small = len(idx_small)

        print(f'{category}: {category_desc} ({len(udis)} columns)')
        print(f'Found {n_small} near-zero singular value(s)')

        # plot singular values
        if save_figures:
            fig, ax = plt.subplots(figsize=(4, 3))
            ax.semilogy(s, marker='o', markersize=4)
            ax.set_title(f'{category}: {category_desc}')
            ax.set_xlabel('Component')
            ax.set_ylabel('Singular value')
            fpath_fig = os.path.join(dpath_figs, f'plot_svd_{domain}_{category}.png')
            fig.savefig(fpath_fig, dpi=300, bbox_inches='tight')
            print(f'Figure saved: {fpath_fig}')

        if n_small == 0:
            print('--------------------')
            continue

        df_udi_info = udi_helper.get_info(udis, colnames=['field_id']).reset_index().rename(columns={'index': 'udi'})
        df_field_info = field_helper.get_info(df_udi_info['field_id'], colnames=['title']).reset_index()
        df_info = df_udi_info.merge(df_field_info, on='field_id').drop_duplicates()

        # get linear combination coefficients and permutation order
        coefs, e = get_coefs(V, idx_small)

        # find the n_small columns that can be written as a linear combination of the other columns
        udis_collinear = [udis[i] for i in e[:n_small]]
        df_tmp = df_info.set_index('udi').loc[udis_collinear]
        df_tmp['category_id'] = category
        dfs_collinear.append(df_tmp)

        # find and save exact linear combinations
        original_data = results_svd[category]['U'] @ np.diag(s) @ V.T
        for i_coefs, i_udi_collinear in enumerate(e[:n_small]):
            udi_collinear = udis[i_udi_collinear]
            labelled_coefs = pd.Series(data=coefs[:, i_coefs], index=[udis[i] for i in e[n_small:]])
            # print(df_info.set_index('udi').loc[udi_collinear])
            # print(np.around(df_coefs.loc[ ~np.isclose(df_coefs['coef'], 0)], decimals=5))

            # make sure that the linear combination is correct
            if not np.allclose(original_data[:, i_udi_collinear], original_data[:, e[n_small:]] @ labelled_coefs):
                raise Exception(f'Linear combination does not work for {udi_collinear}!')

            results_coefs[udi_collinear] = labelled_coefs

        print('--------------------')

    # save csv file
    if len(dfs_collinear) != 0:
        df_collinear = pd.concat(dfs_collinear).reset_index().drop_duplicates()
    else:
        df_collinear = pd.DataFrame(columns=['udi', 'field_id', 'title', 'category_id'])
    df_collinear.to_csv(fpath_out_csv, header=True, index=False)

    # save pickle file
    with open(fpath_out_pkl, 'wb') as file_out:
        pickle.dump(results_coefs, file_out)
