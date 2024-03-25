from dataclasses import dataclass, field
from typing import List

@dataclass
class RemucsOptions:

    quiet: bool = True

    fine: bool = False
    norm: bool = False
    mono: bool = False

    bala: List[float] = field(default_factory=lambda: [0, 0, 0, 0])
    gain: List[float] = field(default_factory=lambda: [1, 1, 1, 1])

    pitch: float = 1

    @property
    def model(self):
        return 'htdemucs_ft' if self.fine else 'htdemucs'
