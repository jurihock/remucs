import os
import warnings

import click
import numpy
import tqdm

# pylint: disable=wildcard-import,unused-wildcard-import
from remucs.common import *
from remucs.utils import filehash

DEMUCS = None

try:
    import demucs.separate # >= 4.0
    DEMUCS = 'demucs.separate'
except ModuleNotFoundError:
    pass

try:
    import demucs.api # >= 4.1
    DEMUCS = 'demucs.api'
except ModuleNotFoundError:
    pass

if not DEMUCS:
    warnings.warn('In order to use remucs, you also need to install demucs!')

def analyze_demucs_separate(src, dst, opts):

    dst = next(iter(dst.values()))
    dst = dst.parent.parent

    args = ['-n', opts.model, '-o', str(dst), '--filename', '{stem}.{ext}', str(src)]

    if not opts.quiet:
        click.echo(f'Executing demucs with args \"{" ".join(args)}\"')

    demucs.separate.main(args)

def analyze_demucs_api(src, dst, opts):

    def callback(args):

        length = args['audio_length']
        models = args['models']
        model  = args['model_idx_in_bag']
        offset = args['segment_offset']
        state  = args['state']
        prog   = args['progress']

        if state.lower() == 'start' and prog is not None:

            y = length * models
            x = length * model + offset

            n = numpy.clip(numpy.ceil(100 * x / y), 0, 100)
            m = numpy.clip(n - prog.n, 0, 100 - prog.n)

            prog.update(m)

    progress  = tqdm.tqdm(total=100) \
                if not opts.quiet else None

    separator = demucs.api.Separator(
        model=opts.model,
        callback=callback,
        callback_arg={'progress': progress})

    # WORKAROUND
    # The `separate_audio_file` function throws the following error when dealing with .wav files:
    #   RuntimeError: unsupported operation:
    #   More than one element of the written-to tensor refers to a single memory location.
    #   Please clone() the tensor before performing the operation.
    # Therefore, load the input file manually and clone the resulting tensor as suggested.
    original  = separator._load_audio(src).clone()
    separated = separator.separate_tensor(original, separator.samplerate)[-1]
    assert sorted(separated.keys()) == sorted(STEMS)

    if progress is not None:
        progress.update(numpy.clip(100 - progress.n, 0, 100))
        progress.close()

    for stem, samples in separated.items():

        if not opts.quiet:
            click.echo(f'Writing {dst[stem].resolve()}')

        dst[stem].parent.mkdir(parents=True, exist_ok=True)
        demucs.api.save_audio(samples, dst[stem], samplerate=separator.samplerate)

def analyze(file, data, opts):

    suffix = file.suffix
    model  = opts.model

    src = file
    dst = {stem: data / model / (stem + suffix) for stem in STEMS}

    check = data / (DIGEST + suffix)
    hash0 = check.read_text().strip() if check.exists() else None
    hash1 = filehash(src, DIGEST).strip()

    if hash0 != hash1:

        check.unlink(missing_ok=True)

        for stem in data.glob(os.path.join('**', '*') + suffix):

            if not opts.quiet:
                click.echo(f'Dropping {stem.resolve()}')

            stem.unlink()

        check.write_text(hash1)

    complete = list(set((data / model / (stem + suffix)).exists() for stem in STEMS))
    complete = complete[0] if len(complete) == 1 else False

    if complete:
        return

    if not opts.quiet:
        click.echo(f'Analyzing {src.resolve()}')

    if DEMUCS == 'demucs.separate':

        analyze_demucs_separate(src, dst, opts)

    elif DEMUCS == 'demucs.api':

        analyze_demucs_api(src, dst, opts)

    else:

        raise ModuleNotFoundError(
            'Unable to perform analysis! ' +
            'Please install demucs and try again.')
