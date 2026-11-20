from __future__ import annotations
from collections.abc import Callable
from dataclasses import dataclass
from functools import cache
from typing import Literal
from lazy import lazy
np = lazy("numpy")
colour = lazy("colour")
import warnings

# colour-science warns on import that its optional plotting and SciPy features are missing
warnings.filterwarnings("ignore", message='"(Matplotlib|SciPy)" related API features are not available')

type Transfer = (
    tuple[Literal["H273"], int]  # ("H273", 1), the encoding curve of an ITU-T H.273 code, undone to scene light
    | tuple[str, None]  # SRGB_CURVE, a curve of colour.OETF_INVERSES
    | tuple[Literal["Gamma"], float]  # ("Gamma", 2.2)
)

# CIE XYZ of D65
D65_XYZ = (0.9504559270516716, 1.0, 1.0890577507598784)

# The white point of the CICP (ITU-T H.273) color primaries that aren't D65, whose primaries colour-science has
CICP_WHITES = {4: "C", 8: "C", 10: "E", 11: "DCI-P3"}

SRGB_CURVE = ("ITU-T H.273 IEC 61966-2", None)  # colour's curve, which continues above 1
BT709_CURVE = ("H273", 1)

# The ITU-T H.273 transfer characteristics codes that are curves; 2 (unspecified) and 8 (linear) have none
CICP_CURVES = {1, 4, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18}
CICP_TRANSFER_LINEAR = 8
CICP_TRANSFER_UNSPECIFIED = 2

# Lookup tables for linearizing integer images, by transfer function and size
LUTS = {}


class ColorWarning(UserWarning):
    """An input's color description can't be fully honored."""


@dataclass(frozen=True, eq=False)
class WorkingSpace:
    """The working data's colors: linear RGB whose (1, 1, 1) is D65 white, or unknown primaries."""
    toXYZ: np.ndarray | None  # 3×3 float64, working RGB (column) → XYZ, with toXYZ @ (1, 1, 1) = D65 XYZ;
    # None when the input states no primaries (untagged), displayed as sRGB

    def __eq__(self, other: object) -> bool:
        """Returns whether both describe the same colors (within the precision of tagged numbers).

        Args:
            other: The other working space.

        Returns:
            True when both have no primaries, or their matrices match within 1e-4.
        """
        if(not isinstance(other, WorkingSpace)):
            return NotImplemented
        if(self.toXYZ is None or other.toXYZ is None):
            return self.toXYZ is None and other.toXYZ is None
        return bool(np.allclose(a=self.toXYZ, b=other.toXYZ, atol=1e-4))

    __hash__ = None


UNTAGGED = WorkingSpace(toXYZ=None)
SRGB = WorkingSpace(
    toXYZ=np.array(
        [
            [0.4123907992659595, 0.3575843393838780, 0.1804807884018343],
            [0.2126390058715104, 0.7151686787677559, 0.0721923153607337],
            [0.0193308187155918, 0.1191947797946259, 0.9505321522496608],
        ],
    ),
)


def fromPrimaries(primaries: np.ndarray | tuple, white: np.ndarray | tuple) -> WorkingSpace:
    """Returns the working space of RGB primaries and a white point, adapted to D65.

    Args:
        primaries: CIE xy of the red, green and blue primaries.
        white: CIE xy of the white point.

    Returns:
        The working space.
    """
    space = colour.RGB_Colourspace(name="", primaries=primaries, whitepoint=white)
    adapted = space.chromatically_adapt(
        whitepoint=colour.XYZ_to_xy(D65_XYZ),
        whitepoint_name=None,
        chromatic_adaptation_transform="Bradford",
    )
    return WorkingSpace(toXYZ=adapted.matrix_RGB_to_XYZ)


def encodingCurve(code: int) -> Callable[[np.ndarray], np.ndarray]:
    """Returns the curve from scene light to signal for an ITU-T H.273 transfer characteristics code.

    Args:
        code: The transfer characteristics code, one of CICP_CURVES.

    Returns:
        colour's BT.2100 PQ OETF (scene light) for PQ (code 16), else the H.273 table's curve.
    """
    return colour.models.oetf_BT2100_PQ if code == 16 else colour.models.TRANSFER_CHARACTERISTICS_ITUTH273[code]


@cache
def h273Table(code: int) -> tuple[np.ndarray, np.ndarray]:
    """Samples an ITU-T H.273 encoding curve, to be inverted by interpolation.

    Args:
        code: The transfer characteristics code, one of CICP_CURVES.

    Returns:
        The encoded values (increasing) and the scene linear values that give them. PQ and HLG put BT.2408's reference
        white (203 cd/m² for PQ, 75% HLG signal) at 1.0; brighter values stay above 1.
    """
    # Dense near black, where curves are steepest, up to the linear value of full signal
    linear = np.linspace(0, 1, 1 << 16, dtype=np.float64)**4*(52.37/48 if code == 17 else 1)
    with np.errstate(invalid="ignore", divide="ignore"):
        encoded = np.asarray(encodingCurve(code)(linear), dtype=np.float64)
    encoded = np.maximum.accumulate(encoded)
    white = {16: colour.eotf_inverse(203, function="ST 2084"), 18: 0.75}.get(code)
    if(white is not None):
        linear /= np.interp(white, encoded, linear)
    return (encoded, linear)


def transferToLinear(x: np.ndarray, transfer: Transfer | None) -> np.ndarray:
    """Converts encoded values (0-1) to linear values.

    Args:
        x: Encoded values.
        transfer: The curve to undo, None for no curve.

    Returns:
        The linear values, same dtype as x.
    """
    dtype = x.dtype
    if(transfer is None):
        return x
    name, param = transfer
    if(name == "Gamma"):
        y = colour.gamma_function(a=x, exponent=param, negative_number_handling="Mirror")
    elif(name == "H273"):
        y = np.interp(x, *h273Table(param))
    else:
        y = colour.oetf_inverse(np.maximum(x, 0), function=name)
    return np.asarray(y).astype(dtype)


def linearizeInteger(img: np.ndarray, transfer: Transfer | None) -> np.ndarray:
    """Converts an 8/16-bit image to linear float32 (0-1) through a lookup table.

    Args:
        img: The integer image, gray or with 3 channels.
        transfer: The curve to undo, None for no curve.

    Returns:
        The linear float32 image.
    """
    n = np.iinfo(img.dtype).max + 1
    key = (transfer, n)
    if(key not in LUTS):
        codes = np.arange(n, dtype=np.float64)/(n - 1)
        LUTS[key] = transferToLinear(x=codes, transfer=transfer).astype(np.float32)
    return LUTS[key][img]


def toSRGBMatrix(workingSpace: WorkingSpace | None) -> np.ndarray | None:
    """Returns the matrix from working RGB to linear sRGB.

    Args:
        workingSpace: The working data's colors.

    Returns:
        The 3×3 matrix, or None when no conversion is needed (sRGB primaries, or no primaries).
    """
    if(workingSpace is None or workingSpace.toXYZ is None or workingSpace == SRGB):
        return None
    return np.linalg.solve(SRGB.toXYZ, workingSpace.toXYZ)


def toSRGBPrimaries(img: np.ndarray, workingSpace: WorkingSpace | None) -> np.ndarray:
    """Converts linear working RGB to linear sRGB primaries.

    Args:
        img: Linear RGB image (channels last, RGB order).
        workingSpace: The working data's colors.

    Returns:
        The image in linear sRGB primaries, unclipped.
    """
    M = toSRGBMatrix(workingSpace)
    if(M is None):
        return img
    return (img.reshape(-1, 3) @ M.T.astype(np.float32)).reshape(img.shape)


def displayEncode(img: np.ndarray, workingSpace: WorkingSpace | None) -> np.ndarray:
    """Encodes linear working data as display-referred sRGB.

    Args:
        img: Linear RGB image (channels last, RGB order) or gray image.
        workingSpace: The working data's colors; None or no primaries are shown as sRGB.

    Returns:
        float32 sRGB-encoded values, clipped to 0-1.
    """
    if(img.ndim == 3):
        img = toSRGBPrimaries(img=img, workingSpace=workingSpace)
    img = np.clip(a=img, a_min=0, a_max=1).astype(np.float32)
    return colour.models.eotf_inverse_sRGB(img).astype(np.float32)


def quantize(img: np.ndarray, bits: int = 8) -> np.ndarray:
    """Quantizes encoded values (0-1) to integers.

    Args:
        img: Encoded values.
        bits: 8 or 16.

    Returns:
        The uint8 or uint16 image.
    """
    if(bits == 16):
        return np.around(np.clip(a=img, a_min=0, a_max=1)*65535).astype(np.uint16)
    return np.around(np.clip(a=img, a_min=0, a_max=1)*255).astype(np.uint8)


def displayImage(img: np.ndarray, workingSpace: WorkingSpace | None, bits: int = 8) -> np.ndarray:
    """Returns linear working data as display-referred, quantized sRGB.

    Args:
        img: Linear RGB image (channels last, RGB order).
        workingSpace: The working data's colors.
        bits: 8 or 16.

    Returns:
        The sRGB-encoded uint8 or uint16 image.
    """
    return quantize(img=displayEncode(img=img, workingSpace=workingSpace), bits=bits)


def cicpTransfer(code: int | None) -> Transfer | None:
    """Returns the transfer function of a CICP transfer characteristics code.

    Args:
        code: The code; None and 2 (unspecified) count as untagged.

    Returns:
        The curve to undo, None for linear or untagged. Reserved codes warn.
    """
    if(code == 13):
        return SRGB_CURVE
    if(code in CICP_CURVES):
        return ("H273", code)
    if(code not in (None, CICP_TRANSFER_UNSPECIFIED, CICP_TRANSFER_LINEAR)):
        warnings.warn(message=f"Reserved transfer characteristics {code}, treated as untagged", category=ColorWarning)
    return None


def cicpPrimaries(code: int | None) -> WorkingSpace:
    """Returns the working space of a CICP color primaries code.

    Args:
        code: The code; None, 2 (unspecified) or an unknown code count as untagged.

    Returns:
        The working space, UNTAGGED when the primaries aren't stated.
    """
    primaries = np.asarray(colour.models.COLOUR_PRIMARIES_ITUTH273.get(code, ""))
    if(primaries.dtype.kind != "f"):
        return UNTAGGED
    white = colour.CCS_ILLUMINANTS["CIE 1931 2 Degree Standard Observer"][CICP_WHITES.get(code, "D65")]
    return fromPrimaries(primaries=primaries, white=white)


def fromCICP(primaries: int, transfer: int) -> tuple[Transfer | None, WorkingSpace]:
    """Describes CICP (ITU-T H.273) color primaries and transfer characteristics codes, e.g. a PNG cICP chunk.

    Args:
        primaries: The color primaries code.
        transfer: The transfer characteristics code.

    Returns:
        The transfer curve to undo (None for linear or untagged) and the working space.
    """
    return (cicpTransfer(transfer), cicpPrimaries(primaries))


def fromVideoTags(primaries: int | None, transfer: int | None, isYUV: bool) -> tuple[Transfer | None, WorkingSpace]:
    """Describes a video stream's colors from its tags, each tag falling back on its own when it is missing.

    Args:
        primaries: CICP color primaries code.
        transfer: CICP transfer characteristics code.
        isYUV: Whether the pixel format is YUV (otherwise RGB or gray).

    Returns:
        The transfer curve to undo and the working space. Untagged YUV is BT.709 (curve and primaries); untagged
        RGB/gray is sRGB, as Windows takes bitmaps.
    """
    curve = cicpTransfer(transfer)
    if(curve is None and transfer != CICP_TRANSFER_LINEAR):
        curve = BT709_CURVE if isYUV else SRGB_CURVE
    workingSpace = cicpPrimaries(primaries)
    if(workingSpace.toXYZ is None):
        workingSpace = SRGB
    return (curve, workingSpace)
