# pylint: disable=import-error
# pylint: disable=fixme

import matplotlib.pyplot as plot
import numpy as np
import numpy.lib.stride_tricks as tricks
import scipy.signal
import soundfile

from qdft import Chroma
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

    print(f'poly degree {degree} coeffs {p}')

    return y


def main():

    cp = 442
    test = f'test.{cp}.wav'
    synth(test, a4=cp)

    samples, samplerate = soundfile.read(test)

    samples = np.mean(samples, axis=-1) \
              if len(np.shape(samples)) > 1 \
              else np.asarray(samples)

    print(f'old length {len(samples)} {len(samples)/samplerate}s')
    length = int(np.ceil(samples.size / samplerate) * samplerate)
    samples.resize(length)
    print(f'new length {len(samples)} {len(samples)/samplerate}s')

    chunks = tricks.sliding_window_view(samples, samplerate)[::samplerate]
    chroma = Chroma(samplerate, decibel=False, feature='hz')

    chromagram = np.empty((0, chroma.size))

    for i, chunk in enumerate(chunks):

        if not i:
            print('0%')

        chromagram = np.vstack((chromagram, chroma.chroma(chunk)))

        print(f'{int(100 * (i + 1) / len(chunks))}%')

    # TODO use chroma.qdft.latencies in the next qdft release
    latency = int(np.max(chroma.qdft.periods[0] - chroma.qdft.offsets))
    print(f'max. qdft latency {latency}')

    print(f'old shape {chromagram.shape}')
    chromagram = chromagram[latency:-samplerate]
    print(f'new shape {chromagram.shape}')

    cp0 = chroma.concertpitch
    cp1 = np.full(len(chromagram), cp0, float)

    r = np.real(chromagram)
    f = np.imag(chromagram)

    i = np.arange(len(chromagram))
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

    # TODO try a lowpass instead
    cp2 = smooth_savgol(cp1, 100e-3, samplerate)
    cp3 = smooth_polyfit(cp2, 1)

    # TODO improve estimation precision
    res = np.round([
        cp3[0],
        cp3[-1],
        np.mean(cp3),
        np.median(cp3)
    ])

    print(f'fist {res[0]} last {res[1]} avg {res[2]} med {res[3]}')

    plot.figure(test)
    plot.plot(cp1)
    plot.plot(cp2, '--')
    plot.plot(cp3, '--')
    plot.show()

    assert res[-1] == cp


if __name__ == '__main__':

    main()
