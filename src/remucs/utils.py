import hashlib
import re

def semitone(value):

    match = re.match('([+,-]?\\d+){1}([+,-]\\d+){0,1}', value)
    assert match is not None

    return pow(2, float(match[1]) / 12)

def cent(value):

    match = re.match('([+,-]?\\d+){1}([+,-]\\d+){0,1}', value)
    assert match is not None

    return pow(2, float(match[2] or 0) / 1200)

def kilo(value):

    if value.lower().endswith('k'):
        return int(value[:-1]) * 1024

    return int(value)

def filehash(file, digest):

    with open(file, 'rb') as stream:
        return hashlib.file_digest(stream, digest).hexdigest()
