#!/usr/bin/env python3
import subprocess
from pathlib import Path

import click

@click.command()
@click.argument('dname', type=str)
@click.argument('dpath_results', type=click.Path(exists=True, path_type=Path), envvar='DPATH_RESULTS')
@click.argument('dpath_logs', type=click.Path(exists=True, path_type=Path), envvar='DPATH_LOGS')
def tar_cca_results(dname: str, dpath_results: Path, dpath_logs: Path):

    def run(args):
        args = [str(arg) for arg in args]
        print(f'\t{args}')
        return subprocess.run(args, check=True)
    
    fpath_results = dpath_results / dname    
    fpath_logs = dpath_logs / dname

    paths_to_tar = [fpath_results]
    # paths_to_tar = [fpath_results, fpath_logs] # do not tar logs since they are symlinked to home directory

    print('Validating paths...')
    for path_to_tar in paths_to_tar:
        if path_to_tar.exists():
            print(f'\t{path_to_tar}')
        else:
            raise FileNotFoundError(f'{path_to_tar} does not exist')
    
    for path_to_tar in paths_to_tar:
        print(f'Tarring {path_to_tar}')
        path_tarred = path_to_tar.with_suffix('.tar.gz')
        run(['tar', '-czvf', path_tarred, '-C', path_to_tar.parent, path_to_tar.name])
        run(['rm', '-rf', path_to_tar])

if __name__ == '__main__':
    tar_cca_results()
