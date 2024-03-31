from os import PathLike
from typing import Union

import pathlib

import click

from remucs.options import RemucsOptions
from remucs.analysis import analyze
from remucs.synthesis import synthesize
from remucs.tuning import howto_shift_pitch


def remucs(file: Union[str, PathLike], data: Union[str, PathLike] = '~', opts: Union[RemucsOptions, None] = None):

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

    data = data / opts.remucs / file.stem
    data.mkdir(parents=True, exist_ok=True)

    src = file
    dst = file.with_suffix(opts.remucs + file.suffix)

    analyze(src, data, opts)

    if opts.a4:

        file       = data / opts.model / ('other' + src.suffix)
        opts.pitch = howto_shift_pitch(file, opts)

    synthesize(dst, data, opts)
