# pylint: disable=import-error
# pylint: disable=fixme

import matplotlib.pyplot as plot
import numpy as np
import numpy.lib.stride_tricks as tricks
import scipy.signal
import soundfile

from qdft import Chroma
from synth import synth

CP = 440
test = f'test.{CP}.wav'
synth(test, a4=CP)

samples, samplerate = soundfile.read(test)

samples = np.mean(samples, axis=-1) \
          if len(np.shape(samples)) > 1 \
          else np.asarray(samples)

print(f'old samples {len(samples)} {len(samples)/samplerate}s')
length = int(np.ceil(samples.size / samplerate) * samplerate)
samples.resize(length)
print(f'new samples {len(samples)} {len(samples)/samplerate}s')

chunks  = tricks.sliding_window_view(samples, samplerate)[::samplerate]
chroma  = Chroma(samplerate, feature='hz')

chromagram = np.empty((0, chroma.size))

for i, chunk in enumerate(chunks):

    if not i:
        print('0%')

    chromagram = np.vstack((chromagram, chroma.chroma(chunk)))

    print(f'{int(100 * (i + 1) / len(chunks))}%')

# TODO chroma.qdft.latencies in the next release
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
j = np.argmax(r, axis=-1)

for n, m in zip(i, j):

    # TODO peak picking
    s = np.round(12 * np.log2(f[n, m] / cp1[n-1]))

    cp1[n] = (f[n, m] * 2**(s/12)) / (2**(s/6))
    cp1[n] = cp1[n-1] if np.isnan(cp1[n]) else cp1[n]

kernel = int(100e-3 * samplerate)
cp1 = scipy.signal.savgol_filter(cp1, kernel, polyorder=1, mode='mirror')

# TODO better estimation precision
stats = np.ceil([
    cp1[0],
    cp1[-1],
    np.mean(cp1),
    np.median(cp1)
])

print(f'fist {stats[0]} last {stats[1]} avg {stats[2]} med {stats[3]}')

plot.figure(test)
plot.plot(cp1)
plot.show()

assert stats[-1] == CP
