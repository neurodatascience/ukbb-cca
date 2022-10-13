#!/usr/bin/env python
import click
from src.data_processing import parse_udis

@click.command()
@click.option('--fpath-raw', required=True, envvar='FPATH_TABULAR_RAW')
@click.option('--fpath-udis', required=True, envvar='FPATH_UDIS')
def process_udis(fpath_raw, fpath_udis):
    df_udis = parse_udis(fpath_raw)
    df_udis.to_csv(fpath_udis, header=True, index=False)
    print(f'Saved parsed UDIs to {fpath_udis}')

if __name__ == '__main__':
    process_udis()
