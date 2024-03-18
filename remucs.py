import click
import demucs.api
import itertools
import math
import os
import pathlib
import shutil
import subprocess
import tqdm

REMUCS = '.remucs'
INPUT  = 'input'
OUTPUT = 'output'
STEMS  = ['drums', 'bass', 'other', 'vocals']

def analyze(stems, suffix, model):

    def callback(args):

        length = args['audio_length']
        models = args['models']
        model  = args['model_idx_in_bag']
        offset = args['segment_offset']
        state  = args['state']
        prog   = args['progress']

        if state.lower() == 'start':

            y = length * models
            x = length * model + offset
            n = min(max(math.ceil(100 * x / y), 0), 100)

            prog.update(n - prog.n)

    with tqdm.tqdm(total=100) as progress:

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

def synthesize(stems, suffix, gains, pans, mono):

    def interlace(*args):
        return list(itertools.chain(*zip(*args)))

    def stringify(args):
        def str_or_path(arg):
            return arg if isinstance(arg, pathlib.PurePath) else str(arg)
        return list(map(str_or_path, args))

    sox = ['sox', '-m']
    src = [stems / (stem + suffix) for stem in STEMS]
    dst = [stems / (OUTPUT + suffix)]

    src = interlace(['-v']*len(STEMS), gains, src)

    subprocess.run(stringify(sox + src + dst), cwd=stems)

if __name__ == '__main__':

    model = 'htdemucs' # + '_ft'
    force = False

    gains = [1]*4
    pans  = [0]*4
    mono  = False

    file   = pathlib.Path('./test.wav')
    name   = file.stem
    suffix = file.suffix

    stems = pathlib.Path().cwd() / REMUCS / name
    stems.mkdir(parents=True, exist_ok=True)
    shutil.copy(file, stems / (INPUT + suffix))

    has_all_stems = list(set((stems / (stem + suffix)).exists() for stem in STEMS))
    has_all_stems = has_all_stems[0] if len(has_all_stems) == 1 else False

    if force or not has_all_stems:
        analyze(stems, suffix, model)

    synthesize(stems, suffix, gains, pans, mono)
