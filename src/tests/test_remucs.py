from dataclasses import dataclass
from pathlib import Path
from typing import List

# pylint: disable=import-error
from test_utils import find, freqs, isless, issame, time, wave

import numpy
import pytest
import soundfile

import remucs

DEBUG = False

numpy.set_printoptions(suppress=True)


@dataclass
class Session:
    data: Path
    src:  Path
    dst:  Path
    sr:   int
    f:    List[int]


@pytest.fixture(name='session', scope='session')
def create_test_session(tmpdir_factory) -> Session:

    if DEBUG:
        data = Path(__file__).resolve().parent.parent
    else:
        data = Path(tmpdir_factory.mktemp('remucs'))

    src = data / 'test.wav'
    dst = data / 'test.remucs.wav'

    sr = 44100
    f  = [1000, 2000]

    hz = numpy.fft.rfftfreq(4096, 1/sr)
    f  = hz[find(hz, f)]

    f  = numpy.floor(f).astype(int).tolist()

    return Session(data=data, src=src, dst=dst, sr=sr, f=f)


def probe(session: Session, **kwargs):

    data = session.data

    src = session.src
    dst = session.dst

    sr = session.sr
    f  = session.f

    remucs.remucs(src, data, remucs.RemucsOptions(**kwargs))
    y = numpy.array(soundfile.read(dst)[0])

    db, hz = freqs(y, sr)
    idx    = find(hz, f)

    return db[idx]


def test_setup(session: Session):

    data = session.data

    src = session.src
    dst = session.dst

    sr = session.sr
    f  = session.f

    t  = time(1, sr)

    x = wave(f, t)
    assert x.ndim == 2
    assert x.shape[0] == len(t)
    assert x.shape[1] == 2

    soundfile.write(src, x, sr)
    assert src.is_file()
    assert soundfile.info(src).samplerate == sr

    remucs.remucs(src, data)
    assert dst.is_file()
    assert soundfile.info(dst).samplerate == sr

    y = numpy.array(soundfile.read(dst)[0])
    assert y.ndim == 2
    assert y.shape[0] == len(t)
    assert y.shape[1] == 2


def test_debug(session: Session):

    data = session.data

    src = session.src
    dst = data / '.remucs' / src.stem / 'htdemucs' / 'other.wav'

    sr = session.sr
    f  = session.f

    x = numpy.array(soundfile.read(src)[0])
    y = numpy.array(soundfile.read(dst)[0])

    print('f debug', f)

    db, hz = freqs(x, sr)
    assert len(db) == len(hz)

    i = db.argmax(axis=0)
    j = numpy.floor(hz[i]).astype(int)
    print('x debug', db[i[0]], db[i[1]], '@', j)

    db, hz = freqs(y, sr)
    assert len(db) == len(hz)

    i = db.argmax(axis=0)
    j = numpy.floor(hz[i]).astype(int)
    print('y debug', db[i[0]], db[i[1]], '@', j)


def test_stereo(session: Session):

    db = probe(session, mono=False)
    print('y stereo', db[0], db[1])

    assert issame(db[0, 0],  0)
    assert isless(db[0, 1], -40)
    assert isless(db[1, 0], -40)
    assert issame(db[1, 1],  0)


def test_mono(session: Session):

    db = probe(session, mono=True)
    print('y mono', db[0], db[1])

    assert issame(db[0, 0], -6)
    assert issame(db[0, 1], -6)
    assert issame(db[1, 0], -6)
    assert issame(db[1, 1], -6)


def test_gain(session: Session):

    db = probe(session, norm=False, gain=[1, 1, 0.5, 1])
    print('y gain', db[0], db[1])

    assert issame(db[0, 0], -6)
    assert isless(db[0, 1], -40)
    assert isless(db[1, 0], -40)
    assert issame(db[1, 1], -6)


def test_norm(session: Session):

    db = probe(session, norm=True, gain=[1, 1, 0.5, 1])
    print('y norm', db[0], db[1])

    assert issame(db[0, 0],  0)
    assert isless(db[0, 1], -40)
    assert isless(db[1, 0], -40)
    assert issame(db[1, 1],  0)


def test_balance_left(session: Session):

    db = probe(session, bala=[0, 0, -1, 0])
    print('y balance left', db[0], db[1])

    assert issame(db[0, 0],  0)
    assert isless(db[0, 1], -40)
    assert isless(db[1, 0], -40)
    assert isless(db[1, 1], -40)


def test_balance_right(session: Session):

    db = probe(session, bala=[0, 0, +1, 0])
    print('y balance right', db[0], db[1])

    assert isless(db[0, 0], -40)
    assert isless(db[0, 1], -40)
    assert isless(db[1, 0], -40)
    assert issame(db[1, 1],  0)
