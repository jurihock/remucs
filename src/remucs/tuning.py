from pathlib import Path
from typing import Tuple
from numpy.typing import ArrayLike, NDArray

import click
import numpy
import soundfile

from qdft import QDFT
from qdft.fafe import QFAFE
from qdft.scale import Scale

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

    x, sr = soundfile.read(src)
    x     = numpy.atleast_2d(x).mean(axis=-1)

    oldsize = len(x)
    newsize = int(numpy.ceil(oldsize / sr) * sr)
    x.resize(newsize)

    scale      = Scale(440)
    bandwidth  = (scale.frequency('A0'), scale.frequency('C#8'))
    resolution = 12*4

    qdft = QDFT(samplerate=sr, bandwidth=bandwidth, resolution=resolution)
    fafe = QFAFE(qdft)

    batches   = numpy.arange(len(x)).reshape((-1, sr))
    estimates = numpy.full(len(x), 440, float)
    weights   = numpy.zeros(len(x), float)

    numpeaks = 3
    kernel   = 1  # int(100e-3 * sr)

    # use qdft.latencies in the next qdft release
    roi = [int(numpy.max(qdft.periods[0] - qdft.offsets)), oldsize - 1]

    for batch in batches:

        dfts  = qdft.qdft(x[batch])
        magns = numpy.abs(dfts)
        freqs = fafe.hz(dfts)

        i = numpy.arange(len(batch))
        j = findpeaks(magns, numpeaks)

        for n, m in zip(i, j):

            if (batch[n] < roi[0]) or (roi[1] < batch[n]):
                continue

            estimate = numpy.median(numpy.roll(estimates, batch[n] + kernel)[:kernel]) \
                       if kernel > 1 else estimates[batch[n] - 1]

            a = numpy.round(12 * numpy.log2(freqs[n, m] / estimate))
            b = numpy.power(2, a / 12)
            c = numpy.power(2, a / 6)

            estimate = numpy.sum(freqs[n, m] * b) / numpy.sum(c)

            if not numpy.isfinite(estimate):
                estimates[batch[n]] = estimates[batch[n] - 1]
                continue

            estimates[batch[n]] = estimate
            weights[batch[n]]   = numpy.prod(magns[n, m])

    return estimates[:oldsize], weights[:oldsize]
