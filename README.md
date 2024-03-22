# Remucs

![language](https://img.shields.io/badge/language-Python-blue)
![license](https://img.shields.io/github/license/jurihock/remucs?color=blue)

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

```
pip install -U git+https://github.com/jurihock/remucs#egg=remucs
```

Remarks:

* Remucs is not compatible with the version _4.0.1_ of [facebookresearch/demucs](https://github.com/facebookresearch/demucs), as published on [pypi](https://pypi.org/project/demucs).
* The required [adefossez/demucs](https://github.com/adefossez/demucs) dependency is currently not published on _pypi_.
* Due to [this issue](https://github.com/pypi/warehouse/issues/7136), there is no way to publish remucs on _pypi_ too.
* Due to specific [torchaudio](https://pypi.org/project/torchaudio/2.1.2) dependency, the _python_ version is restricted to _3.11_.

## License

*remucs* is licensed under the terms of the MIT license.
For details please refer to the accompanying [LICENSE](LICENSE) file distributed with *remucs*.
