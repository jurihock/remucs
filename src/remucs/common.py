import importlib.metadata

VERSION = importlib.metadata.version(__package__)
REMUCS  = '.remucs'
DIGEST  = 'sha256'
MODELS  = ['htdemucs', 'htdemucs_ft']
STEMS   = ['bass', 'drums', 'other', 'vocals']
