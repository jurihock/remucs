# pylint: disable=import-error
# pylint: disable=too-many-locals
# pylint: disable=too-many-statements
# pylint: disable=fixme

import matplotlib.pyplot as plot
import numpy as np
import numpy.lib.stride_tricks as tricks
import scipy.signal
import soundfile

from qdft import QDFT
from qdft.fafe import QFAFE
from qdft.scale import Scale
from synth import synth

np.set_printoptions(suppress=True)


def findpeaks(x, n):

    if n == 1:
        return np.argmax(x, axis=-1)

    a = x[..., 0:-3]
    y = x[..., 1:-2]
    b = x[..., 2:-1]

    i = (y > a) & (y > b)
    j = np.argsort(y * i, axis=-1)[..., ::-1]

    assert np.all(np.argmax(y, axis=-1) == j[..., 0])

    return j[..., :n] + 1


def smooth_savgol(x, seconds, samplerate):

    return scipy.signal.savgol_filter(x,
        window_length=int(seconds * samplerate),
        polyorder=1,
        mode='mirror')


def smooth_polyfit(x, degree):

    i = np.arange(len(x))
    p = np.polyfit(i, x, degree)
    y = np.poly1d(p)(i)

    print(f'polyfit degree {degree} coeffs {p}')

    return y


def main():

    cp = 440
    test = f'test.{cp}.wav'
    synth(test, a4=cp, tenuto=1)

    samples, samplerate = soundfile.read(test)

    samples = np.mean(samples, axis=-1) \
              if len(np.shape(samples)) > 1 \
              else np.asarray(samples)

    print(f'old length {len(samples)} {len(samples)/samplerate}s')
    length = int(np.ceil(samples.size / samplerate) * samplerate)
    samples.resize(length)
    print(f'new length {len(samples)} {len(samples)/samplerate}s')

    scale = Scale()
    bw    = (scale.frequency('A0'), scale.frequency('C#8'))
    qdft  = QDFT(samplerate=samplerate, bandwidth=bw, resolution=12*4)

    chunks = tricks.sliding_window_view(samples, samplerate)[::samplerate]
    data   = np.empty((0, qdft.size))

    for i, chunk in enumerate(chunks):

        if not i:
            print('0%')

        data = np.vstack((data, qdft.qdft(chunk)))

        print(f'{int(100 * (i + 1) / len(chunks))}%')

    fafe  = QFAFE(qdft)
    mags  = np.abs(data)
    freqs = fafe.hz(data)
    data  = mags + 1j * freqs

    # TODO use qdft.latencies in the next qdft release
    latency = int(np.max(qdft.periods[0] - qdft.offsets))
    print(f'max. qdft latency {latency}')

    print(f'old shape {data.shape}')
    data = data[latency:-samplerate]
    print(f'new shape {data.shape}')

    cp0 = scale.concertpitch
    cp1 = np.full(len(data), cp0, float)

    r = np.real(data)
    f = np.imag(data)

    i = np.arange(len(data))
    j = findpeaks(r, 3)
    k = 1 # int(100e-3 * samplerate)

    for n, m in zip(i, j):

        e = np.median(np.roll(cp1, k)[:k]) \
            if k > 1 else cp1[n-1]

        s = np.round(12 * np.log2(f[n, m] / e))

        a = f[n, m]
        b = np.power(2, s / 12)
        c = np.power(2, s / 6)

        cp1[n] = np.sum(a * b) / np.sum(c)
        cp1[n] = cp1[n-1] if np.isnan(cp1[n]) else cp1[n]

    cp2 = smooth_savgol(cp1, 100e-3, samplerate)
    cp3 = smooth_polyfit(cp2, 1)

    vals = np.round(cp1).astype(int)
    vmin = np.min(vals)
    vmax = np.max(vals)
    bins = np.arange(vmin, vmax + 1)
    edge = np.arange(vmin, vmax + 2) - 0.5
    weig = np.prod(r[i[..., None], j], axis=-1)
    hist = np.histogram(vals, bins=edge, weights=weig)
    hmax = np.argmax(hist[0])
    bmax = bins[hmax]
    assert hist[0].shape == bins.shape
    assert hist[1].shape == edge.shape

    res = np.round([
        cp1[0],
        cp1[-1],
        np.min(cp1),
        np.max(cp1),
        np.mean(cp1),
        np.median(cp1),
        bmax
    ]).astype(int)

    lab = [
        'first',
        'last',
        'min',
        'max',
        'avg',
        'med',
        'hist',
    ]

    print('\n'.join(map(str, zip(lab, res))))

    plot.figure(test + ' hist')
    plot.hist(vals, bins=edge)
    plot.axvline(bmax, label='bmax', linestyle='--')
    plot.axvline(cp, label='cp', linestyle='--', color='m')
    plot.legend()
    plot.tight_layout()

    plot.figure(test)
    plot.plot(cp1, label='cp1')
    plot.plot(cp2, label='cp2', linestyle='--')
    plot.plot(cp3, label='cp3', linestyle='--')
    plot.axhline(cp, label='cp', linestyle='--', color='m')
    plot.legend()
    plot.tight_layout()

    plot.show()

    assert res[-1] == cp


if __name__ == '__main__':

    main()
