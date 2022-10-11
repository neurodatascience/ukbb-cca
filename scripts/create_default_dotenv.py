#!/usr/bin/env python
from pathlib import Path
import click

@click.command()
@click.argument('dpath-project')
@click.option('-f', '--fname-dotenv', default='.env')
@click.option('-v', '--verbose', default=True)
def create_default_dotenv(dpath_project, fname_dotenv, verbose):

    dpath_project = Path(dpath_project).resolve()
    
    if verbose:
        click.echo(f'Generating default dotenv file with root directory: {dpath_project}')

    # project root directory
    constants = {'DPATH_PROJECT': dpath_project}
    
    # project subdirectories
    constants['DPATH_DATA'] = constants['DPATH_PROJECT'] / 'data'
    constants['DPATH_RESULTS'] = constants['DPATH_PROJECT'] / 'results'
    constants['DPATH_SCRIPTS'] = constants['DPATH_PROJECT'] / 'scripts'
    constants['DPATH_SCRATCH'] = constants['DPATH_PROJECT'] / 'scratch'

    # data subdirectories
    constants['DPATH_SCHEMA'] = constants['DPATH_DATA'] / 'schema'
    constants['DPATH_RAW'] = constants['DPATH_DATA'] / 'raw'
    constants['DPATH_CLEAN'] = constants['DPATH_DATA'] / 'clean'

    # results subdirectories
    constants['DPATH_CCA_SAMPLE_SIZE'] = constants['DPATH_RESULTS'] / 'cca_sample_size'

    # UK Biobank tabular data
    constants['FPATH_TABULAR_RAW'] = constants['DPATH_RAW'] / 'ukbb_tabular.csv'
    constants['FPATH_TABULAR_MRI'] = constants['DPATH_RAW'] / 'ukbb_tabular_mri.csv'

    # write dotenv file
    fpath_out = Path(constants['DPATH_PROJECT'], fname_dotenv)
    with fpath_out.open('w') as file_dotenv:
        for key, value in constants.items():
            line = f'{key}={value}\n'
            file_dotenv.write(line)
        
        if verbose:
            click.echo(f'Variables written to {fpath_out}')

if __name__ == '__main__':
    create_default_dotenv()
