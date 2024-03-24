import os, sys
src = os.path.join(os.path.dirname(__file__), '..', 'src')
sys.path.insert(0, src)

import numpy
import pathlib
import pytest
import remucs
import soundfile

numpy.set_printoptions(suppress=True)

def time(d, sr):

    return numpy.arange(0, d, 1/sr)

def wave(f, t):

    return numpy.sin(t[..., None] * f * numpy.pi * 2)

def freqs(x, sr):

    n = min(4096, len(x))

    w = numpy.hanning(n)
    y = numpy.fft.rfft(w[..., None] * x[-n:], axis=0)

    db = 20 * numpy.log10(2 * numpy.abs(y) / numpy.sum(w))
    hz = numpy.fft.rfftfreq(n, 1/sr)

    db = numpy.ceil(db).astype(int)

    return db, hz

def find(hz, f):

    return numpy.abs(hz[..., None] - f).argmin(axis=0)

def issame(x, y, tol=1):

    return abs(x - y) <= 1

def isless(x, y):

    return x <= y

@pytest.fixture(scope='session')
def session(tmpdir_factory):

    if False:
        data = pathlib.Path(__file__).resolve().parent.parent
    else:
        data = pathlib.Path(tmpdir_factory.mktemp('remucs'))

    src = data / 'test.wav'
    dst = data / 'test.remucs.wav'

    sr = 44100
    f  = [1000, 2000]

    hz = numpy.fft.rfftfreq(4096, 1/sr)
    f  = hz[find(hz, f)]

    f  = numpy.floor(f).astype(int)

    return dict(sr=sr, f=f, src=src, dst=dst, data=data)

def probe(session, **kwargs):

    data = session['data']

    src = session['src']
    dst = session['dst']

    sr = session['sr']
    f  = session['f']

    remucs.remucs(src, data=data, **kwargs)
    y = numpy.array(soundfile.read(dst)[0])

    db, hz = freqs(y, sr)
    i = find(hz, f)

    return db[i]

def test_setup(session):

    data = session['data']

    src = session['src']
    dst = session['dst']

    sr = session['sr']
    f  = session['f']

    t  = time(1, sr)

    x = wave(f, t)
    assert x.ndim == 2
    assert x.shape[0] == len(t)
    assert x.shape[1] == 2

    soundfile.write(src, x, sr)
    assert src.is_file()
    assert soundfile.info(src).samplerate == sr

    remucs.remucs(src, data=data)
    assert dst.is_file()
    assert soundfile.info(dst).samplerate == sr

    y = numpy.array(soundfile.read(dst)[0])
    assert y.ndim == 2
    assert y.shape[0] == len(t)
    assert y.shape[1] == 2

def test_debug(session):

    data = session['data']

    src = session['src']
    dst = data / '.remucs' / src.stem / 'htdemucs' / 'other.wav'

    sr = session['sr']
    f  = session['f']

    x = numpy.array(soundfile.read(src)[0])
    y = numpy.array(soundfile.read(dst)[0])

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

def test_stereo(session):

    db = probe(session, mono=False)
    print('y stereo', db[0], db[1])

    assert issame(db[0, 0],  0)
    assert isless(db[0, 1], -40)
    assert isless(db[1, 0], -40)
    assert issame(db[1, 1],  0)

def test_mono(session):

    db = probe(session, mono=True)
    print('y mono', db[0], db[1])

    assert issame(db[0, 0], -6)
    assert issame(db[0, 1], -6)
    assert issame(db[1, 0], -6)
    assert issame(db[1, 1], -6)

def test_gain(session):

    db = probe(session, norm=False, gain=[1, 1, 0.5, 1])
    print('y gain', db[0], db[1])

    assert issame(db[0, 0], -6)
    assert isless(db[0, 1], -40)
    assert isless(db[1, 0], -40)
    assert issame(db[1, 1], -6)

def test_norm(session):

    db = probe(session, norm=True, gain=[1, 1, 0.5, 1])
    print('y norm', db[0], db[1])

    assert issame(db[0, 0],  0)
    assert isless(db[0, 1], -40)
    assert isless(db[1, 0], -40)
    assert issame(db[1, 1],  0)

def test_balance_left(session):

    db = probe(session, balance=[0, 0, -1, 0])
    print('y balance left', db[0], db[1])

    assert issame(db[0, 0],  0)
    assert isless(db[0, 1], -40)
    assert isless(db[1, 0], -40)
    assert isless(db[1, 1], -40)

def test_balance_right(session):

    db = probe(session, balance=[0, 0, +1, 0])
    print('y balance right', db[0], db[1])

    assert isless(db[0, 0], -40)
    assert isless(db[0, 1], -40)
    assert isless(db[1, 0], -40)
    assert issame(db[1, 1],  0)
