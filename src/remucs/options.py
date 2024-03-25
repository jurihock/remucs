from dataclasses import dataclass, field
from typing import List

# pylint: disable=wildcard-import,unused-wildcard-import
from remucs.common import *

@dataclass
class RemucsOptions:

    quiet: bool = True

    fine: bool = False
    norm: bool = False
    mono: bool = False

    bala: List[float] = field(default_factory=lambda: [0]*len(STEMS))
    gain: List[float] = field(default_factory=lambda: [1]*len(STEMS))

    pitch:     float = 1
    quefrency: float = 1e-3

    order:   int = 13
    overlap: int = 4

    @property
    def model(self) -> str:
        return MODELS[self.fine]

    @property
    def framesize(self) -> int:
        return 1 << self.order

    @property
    def hopsize(self) -> int:
        return self.framesize // self.overlap
