# Remucs

![language](https://img.shields.io/badge/languages-Python-blue)
![license](https://img.shields.io/github/license/jurihock/remucs?color=blue)
![test](https://img.shields.io/github/actions/workflow/status/jurihock/remucs/test.yml?branch=main&label=test)
![pypi](https://img.shields.io/pypi/v/remucs?color=gold)

The purpose of the _remucs_ command line tool is to extract the individual stems from a mix and remix them again in a certain way, e.g. by adjusting the volume gain, left-right channel balance and last but not least, applying transient-preserving pitch shifting. Since the stem extraction is based on the [adefossez/demucs](https://github.com/adefossez/demucs) engine, the stem choice is restricted to the _drums_, _bass_, _vocals_ and _other_ sources.

## Usage

```
Usage: remucs [OPTIONS] FILES...

Options:
  -f, --fine            Use fine-tuned “htdemucs_ft” model.
  -n, --norm            Normalize output amplitude.
  -m, --mono            Convert stereo input to mono.
  -b, --bala TEXT       Balance of individual stems "bass,drums,other,vocals",
                        e.g. "0,0.5,1,-1". [default: 0,0,0,0]
  -g, --gain TEXT       Gain of individual stems "bass,drums,other,vocals",
                        e.g. "2,1,0.5,0". [default: 1,1,1,1]
  -a, --a4 INTEGER      Target tuning reference frequency, to automatically
                        estimate the pitch shifting factor (experimental).
  -p, --pitch TEXT      Pitch shifting factor in semitones followed by cents,
                        e.g -12 or +12 or +3-50. [default: 0]
  -d, --data DIRECTORY  Directory where to store the intermediate files.
                        [default: <user’s home directory>]
  -q, --quiet           Don't trash stdout.
  -V, --version         Show the version and exit.
  -h, --help            Show this message and exit.
```

## Install

Choose between the latest _remucs_ release or the bleeding edge version:

```
pip install -U remucs
pip install -U git+https://github.com/jurihock/remucs#egg=remucs
```

Don't forget to install one of the available versions of _demucs_ as well:

```
pip install -U demucs
pip install -U git+https://github.com/adefossez/demucs#egg=demucs
```

## License

*remucs* is licensed under the terms of the MIT license.
For details please refer to the accompanying [LICENSE](LICENSE) file distributed with *remucs*.
