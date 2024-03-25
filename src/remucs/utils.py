from pathlib import PurePath
from typing import Union

import hashlib
import re

def semitone(value: str) -> float:

    match = re.match('([+,-]?\\d+){1}([+,-]\\d+){0,1}', value)
    assert match is not None

    return pow(2, float(match[1]) / 12)

def cent(value: str) -> float:

    match = re.match('([+,-]?\\d+){1}([+,-]\\d+){0,1}', value)
    assert match is not None

    return pow(2, float(match[2] or 0) / 1200)

def kilo(value: str) -> int:

    if value.lower().endswith('k'):
        return int(value[:-1]) * 1024

    return int(value)

def filehash(file: Union[str, PurePath], digest: str) -> str:

    with open(file, 'rb') as stream:
        return hashlib.file_digest(stream, digest).hexdigest()
