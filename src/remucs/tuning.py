from pathlib import Path
from typing import Tuple
from numpy.typing import ArrayLike, NDArray

import click
import numpy
import soundfile

from qdft import QDFT
from qdft.fafe import QFAFE

from remucs.options import RemucsOptions


def findpeaks(x: ArrayLike, n: int) -> NDArray:

    x = numpy.atleast_2d(x)
    assert len(x.shape) == 2
    assert x.shape[0] > 0
    assert x.shape[1] > 3

    a = x[..., 0:-3]
    y = x[..., 1:-2]
    b = x[..., 2:-1]

    i = (y > a) & (y > b)
    j = numpy.argpartition(numpy.negative(y * i), n)[..., :n]

    return j + 1


def analyze(src: Path, opts: RemucsOptions) -> Tuple[NDArray, NDArray]:

    if not opts.quiet:
        click.echo(f'Analyzing {src.resolve()}')

    x, samplerate = soundfile.read(src)
    x = numpy.atleast_2d(x).mean(axis=-1)

    reference  = 440
    bandwidth  = (100, 3000)
    resolution = int(1200 / 25)
    batchsize  = int(1 * samplerate)
    numpeaks   = 3

    qdft = QDFT(samplerate=samplerate, bandwidth=bandwidth, resolution=resolution)
    fafe = QFAFE(qdft)

    # use qdft.latencies in the next qdft release
    latency = int(numpy.max(qdft.periods[0] - qdft.offsets))

    oldsize = len(x)
    newsize = int(numpy.ceil((latency + oldsize) / batchsize) * batchsize)

    if oldsize < latency:

        s0 = int(numpy.round(oldsize / samplerate))
        s1 = int(numpy.ceil(latency / samplerate))

        raise ValueError(
            f'The audio file \"{src}\" length of {s0} seconds is too short, ' +
            f'and needs to be at least {s1} seconds! ' +
            'Otherwise reduce the analysis resolution.')

    x.resize(newsize)

    batches   = numpy.arange(len(x)).reshape((-1, batchsize))
    estimates = numpy.zeros(len(x), float)
    weights   = numpy.zeros(len(x), float)

    for batch in batches:

        dfts  = qdft.qdft(x[batch])
        magns = numpy.abs(dfts)
        freqs = fafe.hz(dfts)

        i = numpy.arange(len(batch))[..., None]
        j = findpeaks(magns, numpeaks)

        magns = magns[i, j]
        freqs = freqs[i, j]

        a = numpy.round(12 * numpy.log2(freqs / reference))
        b = numpy.power(2, a / 12)
        c = numpy.power(2, a / 6)

        estimates[batch] = numpy.sum(freqs * b, axis=-1) / numpy.sum(c, axis=-1)
        weights[batch]   = numpy.prod(magns, axis=-1)

    estimates = estimates[latency:latency+oldsize]
    weights   = weights[latency:latency+oldsize]

    return estimates, weights
