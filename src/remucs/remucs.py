import pathlib

import click

# pylint: disable=wildcard-import,unused-wildcard-import
from remucs.common import *
from remucs.analysis import analyze
from remucs.synthesis import synthesize

def remucs(file, *, fine=False, norm=False, mono=False, balance=None, gain=None, pitch=1.0, data='~', quiet=True):

    file = pathlib.Path(file)

    if not file.is_file():
        raise FileNotFoundError(
            f'The specified file "{file}" does not exist!')

    data = pathlib.Path(data).expanduser()

    if not data.is_dir():
        raise FileNotFoundError(
            f'The specified data path "{data}" does not exist!')

    if not quiet:
        click.echo(f'Processing {file.resolve()}')

    data = data / REMUCS / file.stem
    data.mkdir(parents=True, exist_ok=True)

    src = file
    dst = file.with_suffix(REMUCS + file.suffix)

    model = MODELS[fine]

    analyze(src, data, model=model, quiet=quiet)
    synthesize(dst, data, model=model, norm=norm, mono=mono, balance=balance, gain=gain, pitch=pitch, quiet=quiet)
