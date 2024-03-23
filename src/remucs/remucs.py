import importlib.metadata

__version__ = importlib.metadata.version(__package__) \
              if __package__ else None

import click
import hashlib
import numpy
import os
import pathlib
import soundfile
import tqdm
import traceback
import warnings

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

REMUCS = '.remucs'
DIGEST = 'sha256'
MODELS = ['htdemucs', 'htdemucs_ft']
STEMS  = ['bass', 'drums', 'other', 'vocals']

def checksum(file, digest):

    with open(file, 'rb') as stream:
        return hashlib.file_digest(stream, digest).hexdigest()

def analyze(file, data, *, model=MODELS[0], quiet=False):

    suffix = file.suffix

    src = file
    dst = {stem: data / model / (stem + suffix) for stem in STEMS}

    model = model.lower()
    assert model in MODELS, f'Invalid model name "{model}"! Valid model names are: {", ".join(MODELS)}.'

    check = data / (DIGEST + suffix)
    hash0 = check.read_text().strip() if check.exists() else None
    hash1 = checksum(src, DIGEST).strip()

    if hash0 != hash1:

        check.unlink(missing_ok=True)

        for stem in data.glob(os.path.join('**', '*') + suffix):

            if not quiet:
                click.echo(f'Dropping {stem.resolve()}')

            stem.unlink()

        check.write_text(hash1)

    complete = list(set((data / model / (stem + suffix)).exists() for stem in STEMS))
    complete = complete[0] if len(complete) == 1 else False

    if complete:
        return

    if not quiet:
        click.echo(f'Analyzing {src.resolve()}')

    if DEMUCS == 'demucs.separate':

        dst = data
        dst.parent.mkdir(parents=True, exist_ok=True)

        args = ['-n', model, '-o', str(dst), '--filename', '{stem}.{ext}', str(src)]

        if not quiet:
            click.echo(f'Executing demucs with args \"{" ".join(args)}\"')

        demucs.separate.main(args)

    elif DEMUCS == 'demucs.api':

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

        progress  = tqdm.tqdm(total=100) if not quiet else None
        separator = demucs.api.Separator(model=model, callback=callback, callback_arg=dict(progress=progress))

        # WORKAROUND
        # The `separate_audio_file` function throws the following error when dealing with .wav files:
        #   RuntimeError: unsupported operation:
        #   More than one element of the written-to tensor refers to a single memory location.
        #   Please clone() the tensor before performing the operation.
        # Therefore, load the input file manually and clone the resulting tensor as suggested.
        original  = separator._load_audio(src).clone()
        separated = separator.separate_tensor(original, separator.samplerate)[-1]

        if progress is not None:
            progress.update(numpy.clip(100 - progress.n, 0, 100))
            progress.close()

        obtained_stems = sorted(separated.keys())
        expected_stems = sorted(STEMS)
        assert obtained_stems == expected_stems

        for stem, samples in separated.items():

            if not quiet:
                click.echo(f'Writing {dst[stem].resolve()}')

            dst[stem].parent.mkdir(parents=True, exist_ok=True)
            demucs.api.save_audio(samples, dst[stem], samplerate=separator.samplerate)

    else:

        raise RuntimeError('Unable to perform analysis! Please install demucs and try again.')

def synthesize(file, data, *, model=MODELS[0], norm=False, mono=False, balance=[0]*len(STEMS), gain=[1]*len(STEMS), quiet=False):

    suffix = file.suffix

    src = [data / model / (stem + suffix) for stem in sorted(STEMS)]
    dst = file

    if not quiet:
        click.echo(f'Synthesizing {dst.resolve()}')

    balance = numpy.atleast_1d(balance).ravel()
    gain    = numpy.atleast_1d(gain).ravel()

    b, nb = numpy.zeros(len(STEMS)), min(len(STEMS), len(balance))
    g, ng = numpy.ones(len(STEMS)),  min(len(STEMS), len(gain))

    b[:nb] = balance[:nb]
    g[:ng] = gain[:ng]

    b = numpy.clip(b[..., None, None] * [-1, +1] + 1, 0, 1)
    g = numpy.clip(g[..., None, None], -10, +10)

    x, sr = zip(*[soundfile.read(stem) for stem in src])

    assert len(list(set(sr))) == 1
    sr = sr[0]
    x  = numpy.array(x)
    assert x.ndim == 3 and x.shape[-1] == 2

    if not quiet:
        if mono:
            click.echo(f'Converting input to mono')
        if not numpy.all(numpy.equal(numpy.unique(b), 1)):
            click.echo(f'Applying balance weights {b.tolist()}')
        if not numpy.all(numpy.equal(numpy.unique(g), 1)):
            click.echo(f'Applying gain weights {g.tolist()}')
        if norm:
            click.echo(f'Normalizing output')

    if mono:
        x = numpy.mean(x, axis=-1)
        x = numpy.repeat(x[..., None], 2, axis=-1)

    y = numpy.sum(x * b * g, axis=0)

    if norm:
        y /= numpy.max(numpy.abs(y)) or 1

    y = numpy.clip(y, -1, +1)

    soundfile.write(dst, y, sr)

def remucs(file, *, fine=False, norm=False, mono=False, balance=[0]*len(STEMS), gain=[1]*len(STEMS), data='~', quiet=True):

    file = pathlib.Path(file)
    assert file.is_file(), f'Specified file "{file}" does not exist!'

    data = pathlib.Path(data).expanduser()
    assert data.is_dir(), f'Specified data path "{data}" does not exist!'

    if not quiet:
        click.echo(f'Processing {file.resolve()}')

    data = data / REMUCS / file.stem
    data.mkdir(parents=True, exist_ok=True)

    src = file
    dst = file.with_suffix(REMUCS + file.suffix)

    model = MODELS[fine]

    analyze(src, data, model=model, quiet=quiet)
    synthesize(dst, data, model=model, norm=norm, mono=mono, balance=balance, gain=gain, quiet=quiet)

@click.command(no_args_is_help=True, context_settings=dict(help_option_names=['-h', '--help']))
@click.argument('files',       nargs=-1, required=True, type=click.Path(exists=True, file_okay=True, dir_okay=False, path_type=pathlib.Path))
@click.option('-f', '--fine',  default=False, is_flag=True, help=f'Use fine-tuned "{MODELS[1]}" model.')
@click.option('-n', '--norm',  default=False, is_flag=True, help='Normalize output amplitude.')
@click.option('-m', '--mono',  default=False, is_flag=True, help='Convert stereo input to mono.')
@click.option('-b', '--bala',  default=','.join(["0"]*len(STEMS)), show_default=True, help=f'Balance of individual stems [{",".join(sorted(STEMS))}].')
@click.option('-g', '--gain',  default=','.join(["1"]*len(STEMS)), show_default=True, help=f'Gain of individual stems [{",".join(sorted(STEMS))}].')
@click.option('-d', '--data',  default=pathlib.Path().home(), show_default=True, type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=pathlib.Path), help='Directory where to store intermediate files.')
@click.option('-q', '--quiet', default=False, is_flag=True, help='Don\'t trash stdout.')
@click.version_option(__version__ or 'n.a.', '-V', '--version', message='%(version)s')
def cli(files, fine, norm, mono, bala, gain, data, quiet):

    try:

        balance = [float(_) for _ in bala.split(',')]
        gain    = [float(_) for _ in gain.split(',')]

        for file in list(set(files)):
            remucs(file, fine=fine, norm=norm, mono=mono, balance=balance, gain=gain, data=data, quiet=quiet)

        exit(0)

    except Exception as e:

        click.echo(str(e), err=True)
        click.echo(traceback.format_exc(), err=True)

        exit(1)

if __name__ == '__main__':

    cli()
