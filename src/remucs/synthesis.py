import click
import numpy
import soundfile

# pylint: disable=wildcard-import,unused-wildcard-import
from remucs.common import *

def parse_balance_weights(balance):

    if balance is None:
        balance = numpy.zeros(len(STEMS))

    x = numpy.atleast_1d(balance).ravel()
    y = numpy.zeros(len(STEMS))
    n = min(len(x), len(y))

    y[:n] = x[:n]

    return numpy.clip(y[..., None, None] * [-1, +1] + 1, 0, 1)

def parse_gain_weights(gain):

    if gain is None:
        gain = numpy.ones(len(STEMS))

    x = numpy.atleast_1d(gain).ravel()
    y = numpy.ones(len(STEMS))
    n = min(len(x), len(y))

    y[:n] = x[:n]

    return numpy.clip(y[..., None, None], -10, +10)

def synthesize(file, data, *, model='htdemucs', norm=False, mono=False, balance=None, gain=None, quiet=True):

    suffix = file.suffix

    src = [data / model / (stem + suffix) for stem in sorted(STEMS)]
    dst = file

    if not quiet:
        click.echo(f'Synthesizing {dst.resolve()}')

    balance = parse_balance_weights(balance)
    gain    = parse_gain_weights(gain)

    x, sr = zip(*[soundfile.read(stem) for stem in src])

    assert len(list(set(sr))) == 1
    sr = sr[0]
    x  = numpy.array(x)
    assert len(x.shape) == 3 and x.shape[-1] == 2

    if not quiet:
        if mono:
            click.echo('Converting input to mono')
        if not numpy.all(numpy.equal(numpy.unique(balance), 1)):
            click.echo(f'Applying balance weights {balance.tolist()}')
        if not numpy.all(numpy.equal(numpy.unique(gain), 1)):
            click.echo(f'Applying gain weights {gain.tolist()}')
        if norm:
            click.echo('Normalizing output')

    if mono:
        x = numpy.mean(x, axis=-1)
        x = numpy.repeat(x[..., None], 2, axis=-1)

    y = numpy.sum(x * balance * gain, axis=0)

    if norm:
        y /= numpy.max(numpy.abs(y)) or 1

    y = numpy.clip(y, -1, +1)

    soundfile.write(dst, y, sr)
