import click
import demucs.api
import numpy
import pathlib
import shutil
import soundfile
import tqdm
import traceback

REMUCS = '.remucs'
INPUT  = 'input'
OUTPUT = 'output'
MODELS = ['htdemucs', 'htdemucs_ft']
STEMS  = ['bass', 'drums', 'other', 'vocals']

def analyze(stems, suffix, *, model=MODELS[0], quiet=False):

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

    src = stems / (INPUT + suffix)

    model = model.lower()
    assert model in MODELS

    if not quiet:
        click.echo(f'Analyzing {src.resolve()}')

    progress = tqdm.tqdm(total=100) if not quiet else None
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

        dst = stems / model / (stem + suffix)

        if not quiet:
            click.echo(f'Writing {dst.resolve()}')

        dst.parent.mkdir(parents=True, exist_ok=True)
        demucs.api.save_audio(samples, dst, samplerate=separator.samplerate)

def synthesize(stems, suffix, *, model=MODELS[0], norm=False, mono=False, balance=[0]*len(STEMS), gain=[1]*len(STEMS), quiet=False):

    src = [stems / model / (stem + suffix) for stem in sorted(STEMS)]
    dst = stems / (OUTPUT + suffix)

    if not quiet:
        click.echo(f'Synthesizing {dst.resolve()}')

    balance = numpy.atleast_1d(balance).ravel()
    gain    = numpy.atleast_1d(gain).ravel()

    b, nb = numpy.zeros(len(STEMS)), min(len(STEMS), len(balance))
    g, ng = numpy.ones(len(STEMS)),  min(len(STEMS), len(gain))

    b[:nb] = balance[:nb]
    g[:ng] = gain[:ng]

    b = numpy.clip(b[..., None, None] * [-1, +1] + 1, 0, 1)
    g = numpy.clip(g[..., None, None], 0, 1)

    x = [soundfile.read(stem) for stem in src]
    x, sr = zip(*x)

    assert len(list(set(sr))) == 1
    sr = sr[0]
    x = numpy.array(x)
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

def remucs(file, *, fine=False, norm=False, mono=False, balance=[0]*len(STEMS), gain=[1]*len(STEMS), data='~', quiet=False):

    if not quiet:
        click.echo(f'Processing {file.resolve()}')

    model = MODELS[fine]

    name   = file.stem
    suffix = file.suffix

    data = pathlib.Path(data).expanduser()
    assert data.is_dir()

    stems = data / REMUCS / name
    stems.mkdir(parents=True, exist_ok=True)
    shutil.copy(file, stems / (INPUT + suffix))

    has_all_stems = list(set((stems / model / (stem + suffix)).exists() for stem in STEMS))
    has_all_stems = has_all_stems[0] if len(has_all_stems) == 1 else False

    if not has_all_stems:
        analyze(stems, suffix, model=model, quiet=quiet)

    synthesize(stems, suffix, model=model, norm=norm, mono=mono, balance=balance, gain=gain, quiet=quiet)

if __name__ == '__main__':

    @click.command(context_settings=dict(help_option_names=['-h', '--help']))
    @click.argument('files',         nargs=-1, required=True, type=click.Path(exists=True, file_okay=True, dir_okay=False, path_type=pathlib.Path))
    @click.option('-f', '--fine',    default=False, is_flag=True, help=f'Use fine-tuned "{MODELS[1]}" model.')
    @click.option('-n', '--norm',    default=False, is_flag=True, help='Normalize output.')
    @click.option('-m', '--mono',    default=False, is_flag=True, help='Convert stereo source to mono.')
    @click.option('-b', '--balance', default=','.join(["0"]*len(STEMS)), show_default=True, help=f'Balance of individual stems [{",".join(sorted(STEMS))}].')
    @click.option('-g', '--gain',    default=','.join(["1"]*len(STEMS)), show_default=True, help=f'Gain of individual stems [{",".join(sorted(STEMS))}].')
    @click.option('-d', '--data',    default=pathlib.Path().home(), show_default=True, type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=pathlib.Path), help='Directory where to store intermediate files.')
    @click.option('-q', '--quiet',   default=False, is_flag=True, help='Don\'t trash stdout.')
    def cli(files, fine, norm, mono, balance, gain, data, quiet):

        try:

            balance = [float(_) for _ in balance.split(',')]
            gain    = [float(_) for _ in gain.split(',')]

            for file in list(set(files)):
                remucs(file, fine=fine, norm=norm, mono=mono, balance=balance, gain=gain, data=data, quiet=quiet)

        except Exception as e:

            click.echo(str(e), err=True)
            click.echo(traceback.format_exc(), err=True)

    cli()
