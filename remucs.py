import click
import demucs.api
import numpy
import pathlib
import shutil
import soundfile
import tqdm

REMUCS = '.remucs'
INPUT  = 'input'
OUTPUT = 'output'
MODELS = ['htdemucs', 'htdemucs_ft']
STEMS  = ['bass', 'drums', 'other', 'vocals']

def analyze(stems, suffix, *, model=MODELS[0]):

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

    with tqdm.tqdm(total=100) as progress:

        model = model.lower()
        assert model in MODELS

        src = stems / (INPUT + suffix)

        separator = demucs.api.Separator(model=model, callback=callback, callback_arg=dict(progress=progress))

        # WORKAROUND
        # The `separate_audio_file` function throws the following error when dealing with .wav files:
        #   RuntimeError: unsupported operation:
        #   More than one element of the written-to tensor refers to a single memory location.
        #   Please clone() the tensor before performing the operation.
        # Therefore, load the input file manually and clone the resulting tensor as suggested.
        original  = separator._load_audio(src).clone()
        separated = separator.separate_tensor(original, separator.samplerate)[-1]

        obtained_stems = sorted(separated.keys())
        expected_stems = sorted(STEMS)
        assert obtained_stems == expected_stems

        for stem, data in separated.items():
            dst = stems / (stem + suffix)
            demucs.api.save_audio(data, dst, samplerate=separator.samplerate)

def synthesize(stems, suffix, *, norm=False, mono=False, balance=[0]*len(STEMS), gain=[1]*len(STEMS)):

    src = [stems / (stem + suffix) for stem in sorted(STEMS)]
    dst = stems / (OUTPUT + suffix)

    balance = numpy.atleast_1d(balance).ravel()
    gain    = numpy.atleast_1d(gain).ravel()

    b, nb = numpy.zeros(len(STEMS)), min(len(STEMS), len(balance))
    g, ng = numpy.ones(len(STEMS)),  min(len(STEMS), len(gain))

    b[:nb] = balance[:nb]
    g[:ng] = gain[:ng]

    b = numpy.repeat(b[..., None, None], 2, axis=-1)
    b[..., 0] = numpy.clip(1 - b[..., 0], 0, 1)
    b[..., 1] = numpy.clip(1 + b[..., 1], 0, 1)

    g = g[..., None, None]
    g = numpy.clip(g, 0, 1)

    x = [soundfile.read(stem) for stem in src]
    x, sr = zip(*x)

    assert len(list(set(sr))) == 1
    sr = sr[0]
    x = numpy.array(x)
    assert x.ndim == 3 and x.shape[-1] == 2

    if mono:
        x = numpy.mean(x, axis=-1)
        x = numpy.repeat(x[..., None], 2, axis=-1)

    y = numpy.sum(x * b * g, axis=0)

    if norm:
        y /= numpy.max(numpy.abs(y)) or 1

    y = numpy.clip(y, -1, +1)

    soundfile.write(dst, y, sr)

def remucs(file, *, fine=False, norm=False, mono=False, balance=[0]*len(STEMS), gain=[1]*len(STEMS), data=None):

    model = MODELS[fine]
    overwrite = False

    name   = file.stem
    suffix = file.suffix

    stems = (data or pathlib.Path().cwd()) / REMUCS / name
    stems.mkdir(parents=True, exist_ok=True)
    shutil.copy(file, stems / (INPUT + suffix))

    has_all_stems = list(set((stems / (stem + suffix)).exists() for stem in STEMS))
    has_all_stems = has_all_stems[0] if len(has_all_stems) == 1 else False

    if overwrite or not has_all_stems:
        analyze(stems, suffix, model=model)

    synthesize(stems, suffix, norm=norm, mono=mono, balance=balance, gain=gain)

if __name__ == '__main__':

    @click.command(context_settings=dict(help_option_names=['-h', '--help']))
    @click.argument('files', nargs=-1, required=True, type=click.Path(exists=True, file_okay=True, dir_okay=False, path_type=pathlib.Path))
    @click.option('-f', '--fine', default=False, is_flag=True, help=f'Use fine-tuned "{MODELS[1]}" model.')
    @click.option('-n', '--norm', default=False, is_flag=True, help='Normalize output.')
    @click.option('-m', '--mono', default=False, is_flag=True, help='Convert stereo source to mono.')
    @click.option('-b', '--balance', default=','.join(["0"]*len(STEMS)), show_default=True, help=f'Balance of individual stems [{",".join(sorted(STEMS))}].')
    @click.option('-g', '--gain', default=','.join(["1"]*len(STEMS)), show_default=True, help=f'Gain of individual stems [{",".join(sorted(STEMS))}].')
    @click.option('-d', '--data', default=pathlib.Path().cwd(), show_default=True, type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=pathlib.Path), help='Directory where to store intermediate files.')
    def cli(files, fine, norm, mono, balance, gain, data):

        balance = [float(_) for _ in balance.split(',')]
        gain    = [float(_) for _ in gain.split(',')]

        for file in list(set(files)):
            remucs(file, fine=fine, norm=norm, mono=mono, balance=balance, gain=gain, data=data)

    cli()
