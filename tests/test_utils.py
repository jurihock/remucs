from typing import Tuple
from numpy.typing import ArrayLike, NDArray

import numpy


def issame(x: float, y: float, tol: float = 1) -> bool:

    return abs(x - y) <= tol


def isless(x: float, y: float) -> bool:

    return x <= y


def find(x: NDArray, y: ArrayLike) -> NDArray:

    return numpy.abs(x[..., None] - y).argmin(axis=0)


def freqs(x: NDArray, sr: int) -> Tuple[NDArray, NDArray]:

    n = min(4096, len(x))

    w = numpy.hanning(n)
    y = numpy.fft.rfft(w[..., None] * x[-n:], axis=0)

    db = 20 * numpy.log10(2 * numpy.abs(y) / numpy.sum(w))
    hz = numpy.fft.rfftfreq(n, 1/sr)

    db = numpy.ceil(db).astype(int)

    return db, hz


def time(d: int, sr: int) -> NDArray:

    return numpy.arange(0, d, 1/sr)


def wave(f: ArrayLike, t: NDArray) -> NDArray:

    return numpy.sin(t[..., None] * f * numpy.pi * 2)
