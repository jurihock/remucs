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

    return numpy.linspace(0, d, sr)

def wave(f, t):

    return numpy.sin(t[..., None] * f * numpy.pi * 2)

def freqs(x, sr):

    n = min(4096, len(x))

    w = numpy.hanning(n)
    y = numpy.fft.rfft(w[..., None] * x[-n:], axis=0)

    db = 20 * numpy.log10(2 * numpy.abs(y) / numpy.sum(w))
    hz = numpy.fft.rfftfreq(n, 1/sr)

    return db, hz

def find(hz, f):

    return numpy.abs(hz[..., None] - f).argmin(axis=0)

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
    f  = numpy.floor(hz[find(hz, f)])

    return dict(sr=sr, f=f, src=src, dst=dst, data=data)

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
    j = numpy.floor(hz[i])
    print('x', j, db[i[0]], db[i[1]])

    db, hz = freqs(y, sr)
    assert len(db) == len(hz)

    i = db.argmax(axis=0)
    j = numpy.floor(hz[i])
    print('y', j, db[i[0]], db[i[1]])

def test_stereo(session):

    data = session['data']

    src = session['src']
    dst = session['dst']

    sr = session['sr']
    f  = session['f']

    remucs.remucs(src, mono=False, data=data)
    y = numpy.array(soundfile.read(dst)[0])

    db, hz = freqs(y, sr)
    i = find(hz, f)
    print('y', 'stereo', f, db[i[0]], db[i[1]])

    assert numpy.isclose(db[i[0]][0],   0, atol=1)
    assert numpy.isclose(db[i[0]][1], -55, atol=15)
    assert numpy.isclose(db[i[1]][0], -55, atol=15)
    assert numpy.isclose(db[i[1]][1],   0, atol=1)

def test_mono(session):

    data = session['data']

    src = session['src']
    dst = session['dst']

    sr = session['sr']
    f  = session['f']

    remucs.remucs(src, mono=True, data=data)
    y = numpy.array(soundfile.read(dst)[0])

    db, hz = freqs(y, sr)
    i = find(hz, f)
    print('y', 'mono', f, db[i[0]], db[i[1]])

    assert numpy.isclose(db[i[0]][0], -6, atol=1)
    assert numpy.isclose(db[i[0]][1], -6, atol=1)
    assert numpy.isclose(db[i[1]][0], -6, atol=1)
    assert numpy.isclose(db[i[1]][1], -6, atol=1)

def test_gain(session):

    data = session['data']

    src = session['src']
    dst = session['dst']

    sr = session['sr']
    f  = session['f']

    remucs.remucs(src, norm=False, gain=[1, 1, 0.5, 1], data=data)
    y = numpy.array(soundfile.read(dst)[0])

    db, hz = freqs(y, sr)
    i = find(hz, f)
    print('y', 'gain', f, db[i[0]], db[i[1]])

    assert numpy.isclose(db[i[0]][0],  -6, atol=1)
    assert numpy.isclose(db[i[0]][1], -55, atol=15)
    assert numpy.isclose(db[i[1]][0], -55, atol=15)
    assert numpy.isclose(db[i[1]][1],  -6, atol=1)

def test_norm(session):

    data = session['data']

    src = session['src']
    dst = session['dst']

    sr = session['sr']
    f  = session['f']

    remucs.remucs(src, norm=True, gain=[1, 1, 0.5, 1], data=data)
    y = numpy.array(soundfile.read(dst)[0])

    db, hz = freqs(y, sr)
    i = find(hz, f)
    print('y', 'norm', f, db[i[0]], db[i[1]])

    assert numpy.isclose(db[i[0]][0],   0, atol=1)
    assert numpy.isclose(db[i[0]][1], -55, atol=15)
    assert numpy.isclose(db[i[1]][0], -55, atol=15)
    assert numpy.isclose(db[i[1]][1],   0, atol=1)

def test_balance_left(session):

    data = session['data']

    src = session['src']
    dst = session['dst']

    sr = session['sr']
    f  = session['f']

    remucs.remucs(src, balance=[0, 0, -1, 0], data=data)
    y = numpy.array(soundfile.read(dst)[0])

    db, hz = freqs(y, sr)
    i = find(hz, f)
    print('y', 'balance left', f, db[i[0]], db[i[1]])

    assert numpy.isclose(db[i[0]][0],   0, atol=1)
    assert numpy.isclose(db[i[0]][1], -55, atol=15)
    assert numpy.isclose(db[i[1]][0], -55, atol=15)
    assert numpy.isclose(db[i[1]][1], -55, atol=15)

def test_balance_right(session):

    data = session['data']

    src = session['src']
    dst = session['dst']

    sr = session['sr']
    f  = session['f']

    remucs.remucs(src, balance=[0, 0, +1, 0], data=data)
    y = numpy.array(soundfile.read(dst)[0])

    db, hz = freqs(y, sr)
    i = find(hz, f)
    print('y', 'balance right', f, db[i[0]], db[i[1]])

    assert numpy.isclose(db[i[0]][0], -55, atol=15)
    assert numpy.isclose(db[i[0]][1], -55, atol=15)
    assert numpy.isclose(db[i[1]][0], -55, atol=15)
    assert numpy.isclose(db[i[1]][1],   0, atol=1)
