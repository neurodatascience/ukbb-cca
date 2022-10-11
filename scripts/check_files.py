#!/usr/bin/env python

import os

if __name__ == '__main__':

    dpath = '/home/mwang8/ukbb-cca/scratch/cca_gridsearch'

    filenames = os.listdir(dpath)

    counts = {}

    for filename in filenames:
        bad = False
        for n_components1 in range(140, 390, 10):
            if f'cv_{n_components1}' in filename:
                bad = True
                break
        if bad:
            continue
        filename_trimmed = filename.split('.')[0][:-1]
        if filename_trimmed not in counts.keys():
            counts[filename_trimmed] = 0

        counts[filename_trimmed] += 1

    print(max(counts.values()))
    print(len([x for x in counts.values() if x == max(counts.values())]))
    print('---')
    print(min(counts.values()))
    print(len([x for x in counts.values() if x == min(counts.values())]))

    bad = set()
    for filename_trimmed in counts.keys():
        if counts[filename_trimmed] != 2:
            bad.add(filename_trimmed)
    for b in bad:
        print(b)