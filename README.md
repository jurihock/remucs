# Remucs

![language](https://img.shields.io/badge/languages-Python-blue)
![license](https://img.shields.io/github/license/jurihock/remucs?color=green)
![pypi](https://img.shields.io/pypi/v/remucs?color=gold)

The purpose of the _remucs_ command line tool is to extract the individual stems from a mix and remix them again in a certain way.
Since the stem extraction is based on the [adefossez/demucs](https://github.com/adefossez/demucs) engine, the choice is restricted to the _drum_, _bass_, _vocal_ and _other_ sources.

## Usage

```
Usage: remucs [OPTIONS] FILES...

Options:
  -f, --fine            Use fine-tuned “htdemucs_ft” model.
  -n, --norm            Normalize output amplitude.
  -m, --mono            Convert stereo input to mono.
  -b, --bala TEXT       Balance of individual stems [bass,drums,other,vocals].
                        [default: 0,0,0,0]
  -g, --gain TEXT       Gain of individual stems [bass,drums,other,vocals].
                        [default: 1,1,1,1]
  -d, --data DIRECTORY  Directory where to store intermediate files.
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
