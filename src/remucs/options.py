from dataclasses import dataclass, field
from typing import List, Union


@dataclass
class RemucsOptions:

    quiet: bool = True

    fine: bool = False
    norm: bool = False
    mono: bool = False

    bala: List[float] = field(default_factory=lambda: [0]*4)
    gain: List[float] = field(default_factory=lambda: [1]*4)

    a4:        Union[int, None] = None
    pitch:     float = 1
    quefrency: float = 1e-3

    order:   int = 13
    overlap: int = 4

    remucs: str = '.remucs'
    digest: str = 'sha256'

    @property
    def stems(self) -> List[str]:
        return ['bass', 'drums', 'other', 'vocals']

    @property
    def models(self) -> List[str]:
        return ['htdemucs', 'htdemucs_ft']

    @property
    def model(self) -> str:
        return self.models[self.fine]

    @property
    def framesize(self) -> int:
        return 1 << self.order

    @property
    def hopsize(self) -> int:
        return self.framesize // self.overlap
