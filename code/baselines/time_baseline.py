import click

@click.command()
@click.option('--time',
    'time_data',
    type=click.File('r'),
    prompt='Time file',
    help='Time data file.')
@click.option('--save',
    'save_file',
    prompt='Save file',
    type=click.File('wb'),
    help='File to save the pickled results in.')
@click.option('--T',
    'period',
    default=-1,
    type=float,
    help='The periodicity of data (in the same units as the data provided)')
@click.option('--kernel',
    'kernel',
    default=1.0,
    type=float,
    help='The value of the kernel.')
def run(time_data, save_file, period, kernel):
    pass
