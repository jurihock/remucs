import pathlib
import traceback

import click

# pylint: disable=wildcard-import,unused-wildcard-import
from remucs.common import *
from remucs.remucs import remucs

@click.command(                context_settings={'help_option_names': ['-h', '--help']},
                               no_args_is_help=True)
@click.argument('files',
                               nargs=-1,
                               required=True,
                               type=click.Path(exists=True, file_okay=True, dir_okay=False, path_type=pathlib.Path))
@click.option('-f', '--fine',
                               default=False,
                               is_flag=True,
                               help=f'Use fine-tuned "{MODELS[1]}" model.')
@click.option('-n', '--norm',
                               default=False,
                               is_flag=True,
                               help='Normalize output amplitude.')
@click.option('-m', '--mono',
                               default=False,
                               is_flag=True,
                               help='Convert stereo input to mono.')
@click.option('-b', '--bala',
                               default=','.join(["0"]*len(STEMS)),
                               show_default=True,
                               help=f'Balance of individual stems [{",".join(sorted(STEMS))}].')
@click.option('-g', '--gain',
                               default=','.join(["1"]*len(STEMS)),
                               show_default=True,
                               help=f'Gain of individual stems [{",".join(sorted(STEMS))}].')
@click.option('-d', '--data',
                               default=pathlib.Path().home(),
                               show_default=True,
                               type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=pathlib.Path),
                               help='Directory where to store intermediate files.')
@click.option('-q', '--quiet',
                               default=False,
                               is_flag=True,
                               help='Don\'t trash stdout.')
@click.version_option(
                               VERSION,
                               '-V', '--version',
                               message='%(version)s')
def main(files, fine, norm, mono, bala, gain, data, quiet):

    try:

        balance = [float(_) for _ in bala.split(',')]
        gain    = [float(_) for _ in gain.split(',')]

        for file in list(set(files)):
            remucs(file, fine=fine, norm=norm, mono=mono, balance=balance, gain=gain, data=data, quiet=quiet)

    except Exception as error:

        click.echo(traceback.format_exc(), err=True)

        raise click.ClickException(str(error))

if __name__ == '__main__':

    main() # pylint: disable=no-value-for-parameter
