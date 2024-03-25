import importlib.metadata

__version__ = importlib.metadata.version('remucs')

from remucs.remucs import remucs
from remucs.options import RemucsOptions
