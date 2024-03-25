import click
import numpy
import soundfile
import stftpitchshift

# pylint: disable=wildcard-import,unused-wildcard-import
from remucs.common import *

def stereo_balance_weights(balance):

    if balance is None:
        balance = numpy.zeros(len(STEMS))

    x = numpy.atleast_1d(balance).ravel()
    y = numpy.zeros(len(STEMS))
    n = min(len(x), len(y))

    y[:n] = x[:n]

    return numpy.clip(y[..., None, None] * [-1, +1] + 1, 0, 1)

def stereo_gain_weights(gain):

    if gain is None:
        gain = numpy.ones(len(STEMS))

    x = numpy.atleast_1d(gain).ravel()
    y = numpy.ones(len(STEMS))
    n = min(len(x), len(y))

    y[:n] = x[:n]

    return numpy.clip(y[..., None, None], -10, +10)

def shiftpitch(x, *, samplerate, factor, quefrency, framesize, hopsize, normalize=True):

    x = numpy.atleast_2d(x)
    y = numpy.zeros_like(x)
    assert len(x.shape) == 2 and x.shape[-1] == 2

    pitchshifter = stftpitchshift.StftPitchShift(
        framesize=framesize,
        hopsize=hopsize,
        samplerate=samplerate)

    for i in range(x.shape[-1]):

        y[:, i] = pitchshifter.shiftpitch(
            x[:, i],
            factors=factor,
            quefrency=quefrency,
            normalization=normalize)

    return y

def synthesize(file, data, opts):

    suffix = file.suffix
    model  = opts.model

    src = [data / model / (stem + suffix) for stem in sorted(STEMS)]
    dst = file

    norm = opts.norm
    mono = opts.mono

    bala = stereo_balance_weights(opts.bala)
    gain = stereo_gain_weights(opts.gain)

    pitch = opts.pitch

    if not opts.quiet:
        click.echo(f'Synthesizing {dst.resolve()}')

    x, sr = zip(*[soundfile.read(stem) for stem in src])

    assert len(list(set(sr))) == 1
    sr = sr[0]
    x  = numpy.array(x)
    assert len(x.shape) == 3 and x.shape[-1] == 2

    if pitch and pitch > 0 and pitch != 1:

        if not opts.quiet:
            click.echo(f'Applying pitch shifting by factor {pitch}')

        stems       = [STEMS.index(stem) for stem in ['bass', 'other', 'vocals']]
        factors     = [pitch] * len(stems)
        quefrencies = [0, 0, opts.quefrency]
        framesizes  = [opts.framesize] * len(stems)
        hopsizes    = [opts.hopsize] * len(stems)

        for i, stem in enumerate(stems):
            x[stem] = shiftpitch(x[stem],
                samplerate=sr,
                factor=factors[i],
                quefrency=quefrencies[i],
                framesize=framesizes[i],
                hopsize=hopsizes[i])

    if not opts.quiet:
        if mono:
            click.echo('Converting input to mono')
        if not numpy.all(numpy.equal(numpy.unique(bala), 1)):
            click.echo(f'Applying balance weights {bala.tolist()}')
        if not numpy.all(numpy.equal(numpy.unique(gain), 1)):
            click.echo(f'Applying gain weights {gain.tolist()}')
        if norm:
            click.echo('Normalizing output')

    if mono:
        x = numpy.mean(x, axis=-1)
        x = numpy.repeat(x[..., None], 2, axis=-1)

    y = numpy.sum(x * bala * gain, axis=0)

    if norm:
        y /= numpy.max(numpy.abs(y)) or 1

    y = numpy.clip(y, -1, +1)

    soundfile.write(dst, y, sr)
