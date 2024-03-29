# pylint: disable=import-error

from os import PathLike
from pathlib import Path
from subprocess import run
from typing import Union

from mido import Message, MetaMessage, MidiFile, MidiTrack, bpm2tempo
from pytuning.scales import create_edo_scale
from pytuning.tuning_tables import create_timidity_tuning

import click


def synth(file: Union[str, PathLike], *, a4:   int  = 440,
                                         bpm:  int  = 120,
                                         prog: int  = 0,
                                         play: bool = False):

    file = Path(file)

    midi  = MidiFile()
    track = MidiTrack()

    midi.tracks.append(track)

    track.append(MetaMessage('key_signature', key='C'))
    track.append(MetaMessage('time_signature', numerator=4, denominator=4))
    track.append(MetaMessage('set_tempo', tempo=bpm2tempo(bpm)))
    track.append(Message('program_change', program=prog))

    notes = [60, 62, 64, 65, 67, 69, 71, 72, 67, 64, 60]

    for note in notes:

        track.append(Message('note_on',  note=note, velocity=100, time=0))
        track.append(Message('note_off', note=note, velocity=100, time=240))

    track.append(MetaMessage('end_of_track'))
    midi.save(file.with_suffix('.mid'))

    scale  = create_edo_scale(12)
    tuning = create_timidity_tuning(scale, reference_note=69, reference_frequency=a4)

    with open(file.with_suffix('.tab'), 'w', encoding='ascii') as table:
        table.write(tuning)

    timidity = [
        'timidity',
        '-Ow',
        '-Z', file.with_suffix('.tab'),
        '-o', file.with_suffix('.wav'),
        file.with_suffix('.mid')]

    run(timidity, check=True)

    if file.suffix.lower() != '.wav':
        run(['sox', file.with_suffix('.wav'), file], check=True)

    if play:
        run(['play', file], check=True)


@click.command(
                               context_settings={'help_option_names': ['-h', '--help']},
                               no_args_is_help=True)
@click.argument('file',        nargs=1,
                               required=True,
                               type=click.Path(exists=False, file_okay=True, dir_okay=False, path_type=Path))
@click.option('-a', '--a4',
                               default=440,
                               show_default=True,
                               help='Tuning frequency.')
@click.option('-b', '--bpm',
                               default=120,
                               show_default=True,
                               help='Beats per minute.')
@click.option('-p', '--prog',
                               default=0,
                               show_default=True,
                               help='MIDI program number.')
@click.option('-y', '--play',
                               default=False,
                               is_flag=True,
                               help='Play generated file.')
def main(file, a4, bpm, prog, play):

    synth(file, a4=a4, bpm=bpm, prog=prog, play=play)


if __name__ == '__main__':

    main()  # pylint: disable=no-value-for-parameter
