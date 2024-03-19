# REMUCS

The purpose of the _remucs_ command line tool is to extract the individual stems from a mix and remix them again in a certain way.
Since the stem extraction is based on the [demucs](https://github.com/adefossez/demucs) engine, the choice is restricted to the _drum_, _bass_, _vocal_ and _other_ sources.

## Usage

```
Usage: remucs.py [OPTIONS] FILES...

Options:
  -f, --fine            Use fine-tuned "htdemucs_ft" model.
  -n, --norm            Normalize output.
  -m, --mono            Convert stereo source to mono.
  -b, --balance TEXT    Balance of individual stems [bass,drums,other,vocals].
                        [default: 0,0,0,0]
  -g, --gain TEXT       Gain of individual stems [bass,drums,other,vocals].
                        [default: 1,1,1,1]
  -d, --data DIRECTORY  Directory where to store intermediate files.
                        [default: <user home directory>]
  -q, --quiet           Don't trash stdout.
  -h, --help            Show this message and exit.
```
