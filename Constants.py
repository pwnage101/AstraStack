"""Enums and small value types shared across modules. Imports nothing of ours, so any module may import it."""
from dataclasses import dataclass
from enum import Enum

class DemosaicMode(Enum):
    """Whether a source's frames are demosaiced. The values are display names."""
    AUTO = "Auto"
    OFF = "Off"
    ON = "On"

class MosaicPattern(Enum):
    """The 2×2 color filter pattern of a mosaic. The values are display names.

    UNSPECIFIED means the source's library decides (LibRaw), and is never offered to the user.
    """
    RGGB = "RGGB"
    GRBG = "GRBG"
    GBRG = "GBRG"
    BGGR = "BGGR"
    UNSPECIFIED = "LibRaw"

class DemosaicAlgorithm(Enum):
    """The demosaic algorithms OpenCV and LibRaw offer in their standard builds. The values are display names.

    UNSPECIFIED means the source's library decides (LibRaw), and is never offered to the user.
    """
    EDGE_AWARE = "Edge-aware"
    BILINEAR = "Bilinear"
    VNG = "VNG"
    PPG = "PPG"
    AHD = "AHD"
    DCB = "DCB"
    DHT = "DHT"
    AAHD = "AAHD"
    UNSPECIFIED = "LibRaw"

@dataclass(frozen=True)
class DemosaicIntent:
    """A resolved demosaic choice: the frames are a mosaic of this pattern, demosaiced by this algorithm."""
    pattern: MosaicPattern
    algorithm: DemosaicAlgorithm
