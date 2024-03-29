#!/usr/bin/env python
from pathlib import Path
import click
from src.utils import add_suffix

@click.command()
@click.argument('dpath-project', default='.')
@click.option('-f', '--fname-dotenv', default='.env')
@click.option('--verbose/--quiet', default=True)
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
    constants['DPATH_TOOLBOX'] = constants['DPATH_PROJECT'] / 'toolbox'

    # data subdirectories
    constants['DPATH_SCHEMA'] = constants['DPATH_DATA'] / 'schema'
    constants['DPATH_RAW'] = constants['DPATH_DATA'] / 'raw'
    constants['DPATH_PROCESSED'] = constants['DPATH_DATA'] / 'processed'

    # results subdirectories
    constants['DPATH_CCA_SAMPLE_SIZE'] = constants['DPATH_RESULTS'] / 'cca_sample_size'
    constants['DPATH_CCA_COMPARISONS'] = constants['DPATH_RESULTS'] / 'cca_comparisons'
    constants['DPATH_PREPROCESSING'] = constants['DPATH_RESULTS'] / 'preprocessing'

    # scripts subdirectories
    constants['DPATH_LOGS'] = constants['DPATH_SCRIPTS'] / 'logs'

    # requirements.txt
    constants['FPATH_REQUIREMENTS'] = constants['DPATH_PROJECT'] / 'requirements.txt'

    # job template file
    constants['FPATH_TEMPLATE'] = constants['DPATH_SCRIPTS'] / 'python_job_template.sh'

    # UK Biobank tabular data
    constants['FPATH_UDIS'] = constants['DPATH_RAW'] / 'UDIs.csv'
    constants['FPATH_SUBJECTS_WITHDRAWN'] = constants['DPATH_RAW'] / 'subjects_to_remove.csv'
    constants['FPATH_TABULAR_RAW'] = constants['DPATH_RAW'] / 'ukbb_tabular.csv'
    constants['FPATH_TABULAR_MRI'] = add_suffix(constants['FPATH_TABULAR_RAW'], 'mri')
    constants['FPATH_TABULAR_MRI_FILTERED'] = add_suffix(constants['FPATH_TABULAR_MRI'], 'filtered')

    # write dotenv file
    fpath_out = Path(constants['DPATH_SCRIPTS'], fname_dotenv)
    with fpath_out.open('w') as file_dotenv:
        for key, value in constants.items():
            line = f'{key}={value}\n'
            file_dotenv.write(line)
        
    if verbose:
        click.echo(f'Variables written to {fpath_out}')

if __name__ == '__main__':
    create_default_dotenv()
