from pathlib import PurePath
from typing import Union

import pathlib

import click

# pylint: disable=wildcard-import,unused-wildcard-import
from remucs.common import *
from remucs.options import RemucsOptions
from remucs.analysis import analyze
from remucs.synthesis import synthesize

def remucs(file: Union[str, PurePath], data: Union[str, PurePath] = '~', opts: Union[RemucsOptions, None] = None):

    file = pathlib.Path(file)

    if not file.is_file():
        raise FileNotFoundError(
            f'The specified file "{file}" does not exist!')

    data = pathlib.Path(data).expanduser()

    if not data.is_dir():
        raise FileNotFoundError(
            f'The specified data path "{data}" does not exist!')

    opts = opts or RemucsOptions()

    if not opts.quiet:
        click.echo(f'Processing {file.resolve()}')

    data = data / REMUCS / file.stem
    data.mkdir(parents=True, exist_ok=True)

    src = file
    dst = file.with_suffix(REMUCS + file.suffix)

    analyze(src, data, opts)
    synthesize(dst, data, opts)
