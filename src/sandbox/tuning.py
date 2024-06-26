# pylint: disable=import-error

from pathlib import Path

import matplotlib.pyplot as plot
import numpy as np
import scipy.signal
import soundfile

from synth import synth
from remucs import RemucsOptions
from remucs.tuning import analyze

np.set_printoptions(suppress=True)


def smooth_savgol(x, seconds, samplerate):

    return scipy.signal.savgol_filter(x,
        window_length=int(seconds * samplerate),
        polyorder=1,
        mode='mirror')


def main():

    cp0 = 440
    test = Path(f'test.{cp0}.wav')
    synth(test, a4=cp0, tenuto=1)
    sr = soundfile.info(test).samplerate

    cp2, weights = analyze(test, RemucsOptions())

    values = np.round(cp2).astype(int)
    bounds = np.min(values), np.max(values)
    bins   = np.arange(bounds[0], bounds[1] + 1)
    edges  = np.arange(bounds[0], bounds[1] + 2) - 0.5
    hist   = np.histogram(values, bins=edges, weights=weights)

    assert hist[0].shape == bins.shape
    assert hist[1].shape == edges.shape

    cp1 = bins[np.argmax(hist[0])]
    cp3 = smooth_savgol(cp2, 100e-3, sr)

    histweights = weights / np.max(hist[0])
    plotweights = weights * np.ptp(cp2) / np.max(weights) + np.min(cp2)

    cents = round(1200 * np.log2(cp1 / cp0))

    print(f'a4 orig {cp0} est {cp1} cents {cents}')

    plot.figure(f'{test} hist')
    plot.hist(values, bins=edges, weights=histweights)
    plot.axvline(cp0, label='cp0', linestyle='-', color='c')
    plot.axvline(cp1, label='cp1', linestyle='--', color='m')
    plot.legend()
    plot.tight_layout()

    plot.figure(f'{test} plot')
    plot.plot(cp2)
    plot.plot(cp3, linestyle='-')
    plot.plot(plotweights, linestyle=':')
    plot.axhline(cp0, label='cp0', linestyle='-', color='c')
    plot.axhline(cp1, label='cp1', linestyle='--', color='m')
    plot.legend()
    plot.tight_layout()

    plot.show()

    assert cp1 == cp0


if __name__ == '__main__':

    main()
