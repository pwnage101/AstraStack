from __future__ import annotations
from dataclasses import dataclass
from functools import cache
from typing import Literal
from lazy import lazy
np = lazy("numpy")
colour = lazy("colour")
Image = lazy("PIL.Image")
ImageCms = lazy("PIL.ImageCms")
imagecodecs = lazy("imagecodecs")
import io
import struct
import warnings
import zlib

# colour-science warns on import that its optional plotting features are missing
warnings.filterwarnings("ignore", message='"Matplotlib" related API features are not available')

Transfer = (
    tuple[Literal["sRGB", "BT.709", "BT.601", "BT.2020", "PQ", "HLG"], None]  # ("sRGB", None)
    | tuple[Literal["Gamma"], float]  # ("Gamma", 2.2)
    | tuple[Literal["ICC"], bytes]  # ("ICC", profile bytes), a matrix/TRC RGB or gray TRC profile
)
BayerPattern = Literal["RGGB", "GRBG", "GBRG", "BGGR"]

# CIE xy of D65 and XYZ of D65 and of the ICC PCS white (D50)
D65 = (0.3127, 0.3290)
D65_XYZ = (0.9504559270516716, 1.0, 1.0890577507598784)
D50_XYZ = (0.9642, 1.0, 0.8249)

# Adobe RGB (1998)'s gamma, 2 51/256, which cameras signal with the DCF Interoperability Index "R03"
ADOBE_RGB_GAMMA = 563/256

# CICP (ITU-T H.273) color primaries, as colour-science color space names
CICP_PRIMARIES = {1: "ITU-R BT.709",
                  4: "ITU-R BT.470 - 525",
                  5: "ITU-R BT.470 - 625",
                  6: "SMPTE C",
                  7: "SMPTE 240M",
                  9: "ITU-R BT.2020",
                  11: "DCI-P3",
                  12: "Display P3",
                  22: "EBU Tech. 3213-E"}

# CICP transfer characteristics that are decoded, and the linear one (no curve)
CICP_TRANSFERS = {1: ("BT.709", None),
                  6: ("BT.601", None),
                  13: ("sRGB", None),
                  14: ("BT.2020", None),
                  15: ("BT.2020", None),
                  16: ("PQ", None),
                  18: ("HLG", None)}
CICP_TRANSFER_LINEAR = 8

# CICP transfer characteristics that are recognized but not decoded, so they are treated as untagged
CICP_UNSUPPORTED_TRANSFERS = {4: "BT.470 M (gamma 2.2)",
                              5: "BT.470 BG (gamma 2.8)",
                              7: "SMPTE 240M",
                              9: "logarithmic (100:1)",
                              10: "logarithmic (316:1)",
                              11: "IEC 61966-2-4",
                              12: "BT.1361",
                              17: "SMPTE 428"}

# CICP matrix coefficients, as the YUV -> RGB matrix used for them
CICP_MATRICES = {1: "BT.709",
                 4: "BT.601",
                 5: "BT.601",
                 6: "BT.601",
                 9: "BT.2020",
                 10: "BT.2020"}

# Lookup tables for linearizing integer images, by transfer function and size
LUTS = {}


class ColorWarning(UserWarning):
    """An input's color description can't be fully honored."""


@dataclass(frozen=True)
class DecodeParams:
    """What the loader needs to turn stored values into linear 0-1 working data."""
    transfer: Transfer | None = None  # ("sRGB", None)
    yuvMatrix: Literal["BT.601", "BT.709", "BT.2020"] | None = None  # "BT.709"
    range: Literal["limited", "full"] | None = None  # "limited"
    bayerPattern: BayerPattern | None = None  # "RGGB"
    # (1008.0, 1008.0, 1008.0), one per output channel (RGB), as the file states it; the Process black level
    # subtracts it, not the loader
    blackLevel: tuple[float, ...] | None = None
    whiteLevel: float | None = None  # 16383.0


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


def bradford(src: np.ndarray | tuple, dst: np.ndarray | tuple) -> np.ndarray:
    """Returns the Bradford chromatic adaptation matrix from one white to another.

    Args:
        src: XYZ of the source white.
        dst: XYZ of the destination white.

    Returns:
        The 3×3 matrix converting XYZ under src to XYZ under dst.
    """
    return colour.adaptation.matrix_chromatic_adaptation_VonKries(
        XYZ_w=np.asarray(src, dtype=np.float64),
        XYZ_wr=np.asarray(dst, dtype=np.float64),
        transform="Bradford",
    )


def toD65(toXYZ: np.ndarray, white: np.ndarray | tuple) -> WorkingSpace:
    """Returns the working space of an RGB -> XYZ matrix, adapted from its white to D65.

    The adapted matrix is then scaled per primary so (1, 1, 1) maps exactly to D65, which removes the rounding of
    tagged numbers.

    Args:
        toXYZ: RGB -> XYZ matrix whose (1, 1, 1) is the given white.
        white: XYZ of the matrix's white.

    Returns:
        The working space, whose toXYZ maps (1, 1, 1) to D65.
    """
    M = bradford(white, D65_XYZ) @ np.asarray(toXYZ, dtype=np.float64)
    return WorkingSpace(toXYZ=M @ np.diag(np.linalg.solve(M, D65_XYZ)))


def fromPrimaries(primaries: np.ndarray | tuple, white: np.ndarray | tuple) -> WorkingSpace:
    """Returns the working space of RGB primaries and a white point, adapted to D65.

    Args:
        primaries: CIE xy of the red, green and blue primaries.
        white: CIE xy of the white point.

    Returns:
        The working space.
    """
    toXYZ = colour.normalised_primary_matrix(np.asarray(primaries, dtype=np.float64), np.asarray(white))
    return toD65(toXYZ=toXYZ, white=colour.xy_to_XYZ(np.asarray(white, dtype=np.float64)))


def named(name: str) -> WorkingSpace:
    """Returns the working space of a colour-science RGB color space.

    Args:
        name: Name of the color space, e.g. "ITU-R BT.2020".

    Returns:
        The working space, sRGB's exactly for Rec.709 primaries.
    """
    if(name in ("sRGB", "ITU-R BT.709")):
        return SRGB
    cs = colour.RGB_COLOURSPACES[name]
    return fromPrimaries(primaries=cs.primaries, white=cs.whitepoint)


REC2020_NAME = "ITU-R BT.2020"

# colour-science's name for each named curve. Every curve is undone as a camera curve (OETF), to scene light.
NAMED_TRANSFERS = {"sRGB": "ITU-T H.273 IEC 61966-2",
                   "BT.709": "ITU-R BT.709",
                   "BT.601": "ITU-R BT.601",
                   "BT.2020": "ITU-R BT.2020",
                   "HLG": "ITU-R BT.2100 HLG",
                   "PQ": "ITU-R BT.2100 PQ"}


@cache
def referenceWhite(name: str) -> float:
    """Returns the scene light of a named curve's reference white, which is decoded to 1.0.

    Args:
        name: The curve, a key of NAMED_TRANSFERS.

    Returns:
        For PQ and HLG, BT.2408's: the scene light that PQ shows at 203 cd/m², or that HLG encodes as 75% signal.
        For the SDR curves, 1.0, the scene light of full signal.
    """
    if(name == "PQ"):
        return float(colour.oetf_inverse(colour.eotf_inverse(203, function="ST 2084"), function=NAMED_TRANSFERS[name]))
    elif(name == "HLG"):
        return float(colour.oetf_inverse(0.75, function=NAMED_TRANSFERS[name]))
    return 1.0


def transferToLinear(x: np.ndarray, transfer: Transfer | None, bgr: bool = False) -> np.ndarray:
    """Converts encoded values (0-1) to linear values.

    Args:
        x: Encoded values; ICC transfers need the channels on the last axis.
        transfer: The curve to undo, None for no curve.
        bgr: Whether the channels are in BGR order (ICC transfers only).

    Returns:
        The linear values, same dtype as x.
    """
    dtype = x.dtype
    if(transfer is None):
        return x
    name, params = transfer
    if(name == "Gamma"):
        y = colour.gamma_function(a=x, exponent=params, negative_number_handling="Mirror")
    elif(name == "ICC"):
        y = iccToLinear(x=x[..., ::-1] if bgr else x, icc=params)
        y = y[..., ::-1] if bgr else y
    else:
        with np.errstate(invalid="ignore", divide="ignore"):
            y = colour.oetf_inverse(np.maximum(x, 0), function=NAMED_TRANSFERS[name])
            # Normalize so that "paper white" level is always 1.0 in working linear space.
            # Yes, that means some values in linear space are higher than 1.0.
            y = y/referenceWhite(name)
    return np.asarray(y).astype(dtype)


def linearizeInteger(img: np.ndarray, transfer: Transfer | None, bgr: bool = False) -> np.ndarray:
    """Converts an 8/16-bit image to linear float32 (0-1) through a lookup table.

    Args:
        img: The integer image, gray or with 3 channels.
        transfer: The curve to undo, None for no curve.
        bgr: Whether the channels are in BGR order (ICC transfers only).

    Returns:
        The linear float32 image.
    """
    n = np.iinfo(img.dtype).max + 1
    key = (transfer, n)
    if(key not in LUTS):
        codes = np.arange(n, dtype=np.float64)/(n - 1)
        if(transfer is not None and transfer[0] == "ICC"):
            LUTS[key] = transferToLinear(x=np.stack([codes]*3, axis=-1), transfer=transfer).T.astype(np.float32)
        else:
            LUTS[key] = transferToLinear(x=codes, transfer=transfer).astype(np.float32)
    lut = LUTS[key]
    if(lut.ndim == 1):
        return lut[img]
    if(img.ndim == 2):
        return lut[0][img]
    if(bgr):
        lut = lut[::-1]
    return np.dstack([lut[i][img[:, :, i]] for i in range(3)])


def linearize(img: np.ndarray, params: DecodeParams, bgr: bool = True) -> np.ndarray:
    """Converts stored image values to linear 0-1 working data.

    Args:
        img: The stored (integer, or float for FITS) image, gray or with 3 channels.
        params: How the values are encoded.
        bgr: Whether the channels are in BGR order.

    Returns:
        The linear float32 image. Sensor data is scaled by its white level and keeps its black level.
    """
    if(params.whiteLevel is not None):
        return (img.astype(np.float32)/np.float32(params.whiteLevel)).astype(np.float32)
    if(np.issubdtype(img.dtype, np.floating)):
        return transferToLinear(x=img.astype(np.float32), transfer=params.transfer, bgr=bgr)
    return linearizeInteger(img=img, transfer=params.transfer, bgr=bgr)


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


def iccProfile(icc: bytes) -> ImageCms.core.CmsProfile:
    """Opens an ICC profile with LittleCMS.

    Args:
        icc: The profile bytes.

    Returns:
        The profile.
    """
    return ImageCms.ImageCmsProfile(io.BytesIO(icc)).profile


@cache
def linearProfile(icc: bytes) -> bytes:
    """Returns a profile with a matrix/TRC RGB profile's colorants but linear curves, or a linear gray profile.

    Args:
        icc: The profile bytes.

    Returns:
        The linear profile's bytes.
    """
    profile = iccProfile(icc)
    if(profile.xcolor_space == "GRAY"):
        return imagecodecs.cms_profile("gray", gamma=1.0)
    # Colorants are D50-adapted, so giving their xyY with the D50 white keeps them exactly.
    primaries = [v for xyY in (profile.red_primary, profile.green_primary, profile.blue_primary) for v in xyY[1]]
    return imagecodecs.cms_profile(
        "rgb",
        whitepoint=[*colour.XYZ_to_xyY(np.array(D50_XYZ))],
        primaries=primaries,
        gamma=1.0,
    )


def iccToLinear(x: np.ndarray, icc: bytes) -> np.ndarray:
    """Converts encoded values (0-1) through an ICC profile's tone curves, with LittleCMS.

    Args:
        x: Encoded values; RGB profiles need the channels (RGB order) on the last axis.
        icc: A matrix/TRC RGB or gray TRC profile's bytes.

    Returns:
        The linear values, in the profile's own primaries.
    """
    gray = iccProfile(icc).xcolor_space == "GRAY"
    colorspace = "gray" if gray else "rgb"
    y = imagecodecs.cms_transform(
        x.reshape((1, -1) if gray else (1, -1, 3)),
        profile=icc,
        outprofile=linearProfile(icc),
        colorspace=colorspace,
        outcolorspace=colorspace,
    )
    return y.reshape(x.shape)


def isMatrixProfile(icc: bytes) -> bool:
    """Returns whether an ICC profile is a matrix/TRC RGB profile; any LUTs it also has are ignored.

    Args:
        icc: The profile bytes.

    Returns:
        True for a matrix/TRC RGB profile.
    """
    profile = iccProfile(icc)
    return profile.xcolor_space == "RGB " and profile.is_matrix_shaper


def isGrayProfile(icc: bytes) -> bool:
    """Returns whether an ICC profile is a gray TRC profile.

    Args:
        icc: The profile bytes.

    Returns:
        True for a gray TRC profile.
    """
    profile = iccProfile(icc)
    return profile.xcolor_space == "GRAY" and profile.is_matrix_shaper


def fromGrayICC(icc: bytes) -> tuple[DecodeParams, WorkingSpace]:
    """Describes a gray TRC ICC profile: its tone curve, and no primaries.

    Args:
        icc: The profile bytes.

    Returns:
        The decode parameters and working space.
    """
    return (DecodeParams(transfer=("ICC", icc)), UNTAGGED)


def fromICC(icc: bytes) -> tuple[DecodeParams, WorkingSpace]:
    """Describes a matrix/TRC RGB ICC profile.

    The colorants are relative to the D50 PCS, so they are adapted D50 -> D65.

    Args:
        icc: The profile bytes.

    Returns:
        The decode parameters (the ICC curves, or the sRGB curve when they match it) and working space.
    """
    profile = iccProfile(icc)
    colorants = np.column_stack([c[0] for c in (profile.red_colorant, profile.green_colorant, profile.blue_colorant)])
    workingSpace = toD65(toXYZ=colorants, white=D50_XYZ)
    x = np.linspace(start=0, stop=1, num=4096)
    linear = iccToLinear(x=np.stack([x]*3, axis=-1), icc=icc)
    if(np.abs(linear - colour.models.eotf_sRGB(x)[:, np.newaxis]).max() < 1e-4):
        transfer = ("sRGB", None)
    else:
        transfer = ("ICC", icc)
    return (DecodeParams(transfer=transfer), workingSpace)


def describeICC(icc: bytes, channels: int) -> tuple[DecodeParams, WorkingSpace] | None:
    """Describes an embedded ICC profile.

    Args:
        icc: The profile bytes.
        channels: Number of channels of the image it's embedded in.

    Returns:
        The decode parameters and working space, or None when the profile isn't supported (with a warning).
    """
    try:
        if(isGrayProfile(icc)):
            return fromGrayICC(icc)
        elif(isMatrixProfile(icc)):
            return fromICC(icc)
    except Exception as error:
        warnings.warn("Unsupported ICC profile, the image is treated as untagged: " + str(error), ColorWarning)
        return None
    warnings.warn("Unsupported ICC profile, the image is treated as untagged", ColorWarning)
    return None


def cicpTransfer(code: int | None) -> Transfer | None:
    """Returns the transfer function of a CICP transfer characteristics code.

    Args:
        code: The code; None, 2 (unspecified) or an unknown code count as untagged.

    Returns:
        The curve to undo, None for linear or untagged. Recognized but unsupported curves warn.
    """
    if(code in CICP_TRANSFERS):
        return CICP_TRANSFERS[code]
    if(code in CICP_UNSUPPORTED_TRANSFERS):
        warnings.warn(
            message=f"Unsupported transfer characteristics {CICP_UNSUPPORTED_TRANSFERS[code]}, treated as untagged",
            category=ColorWarning,
        )
    return None


def cicpPrimaries(code: int | None) -> WorkingSpace:
    """Returns the working space of a CICP color primaries code.

    Args:
        code: The code; None, 2 (unspecified) or an unknown code count as untagged.

    Returns:
        The working space, UNTAGGED when the primaries aren't stated.
    """
    if(code in CICP_PRIMARIES):
        return named(CICP_PRIMARIES[code])
    return UNTAGGED


def fromCICP(primaries: int, transfer: int) -> tuple[DecodeParams, WorkingSpace]:
    """Describes CICP (ITU-T H.273) color primaries and transfer characteristics codes, e.g. a PNG cICP chunk.

    Args:
        primaries: The color primaries code.
        transfer: The transfer characteristics code.

    Returns:
        The decode parameters and working space.
    """
    return (DecodeParams(transfer=cicpTransfer(transfer)), cicpPrimaries(primaries))


def fromVideoTags(
    primaries: int | None,
    transfer: int | None,
    matrix: int | None,
    colorRange: int | None,
    isYUV: bool,
    fullRangeFormat: bool = False,
) -> tuple[DecodeParams, WorkingSpace]:
    """Describes a video stream from its color tags, each tag falling back on its own when it is missing.

    Args:
        primaries: CICP color primaries code.
        transfer: CICP transfer characteristics code.
        matrix: CICP matrix coefficients code.
        colorRange: FFmpeg color range (1 = limited, 2 = full, 0 or None = unspecified).
        isYUV: Whether the pixel format is YUV (otherwise RGB or gray).
        fullRangeFormat: Whether the pixel format itself is full range (yuvj*).

    Returns:
        The decode parameters and working space. Untagged YUV uses the BT.709 matrix and limited range, untagged
        RGB/gray is full range.
    """
    yuvMatrix = None
    if(isYUV):
        yuvMatrix = CICP_MATRICES.get(matrix, "BT.709")
    if(colorRange == 1):
        rangeName = "limited"
    elif(colorRange == 2 or fullRangeFormat or not isYUV):
        rangeName = "full"
    else:
        rangeName = "limited"
    params = DecodeParams(transfer=cicpTransfer(transfer), yuvMatrix=yuvMatrix, range=rangeName)
    return (params, cicpPrimaries(primaries))


# Luminance weights (Kr, Kb) of the YUV matrices
YUV_WEIGHTS = {"BT.601": (0.299, 0.114),
               "BT.709": (0.2126, 0.0722),
               "BT.2020": (0.2627, 0.0593)}


def yuvToRGB(
    yuv: np.ndarray,
    bits: int,
    matrix: Literal["BT.601", "BT.709", "BT.2020"],
    colorRange: Literal["limited", "full"],
) -> np.ndarray:
    """Converts 4:4:4 YUV to full range RGB, expanding limited range.

    Limited range Y 16-235 and Cb/Cr 16-240 (scaled for higher bit depths) become 0-1; values outside the nominal
    range are clipped.

    Args:
        yuv: The Y, Cb and Cr planes, stacked on the first axis, as stored integers.
        bits: Bit depth of the stored values.
        matrix: The YUV -> RGB matrix.
        colorRange: Whether the values are limited or full range.

    Returns:
        The float32 RGB image (channels last), within 0-1.
    """
    scale = 2**(bits - 8)
    y, cb, cr = [plane.astype(np.float32) for plane in yuv]
    if(colorRange == "limited"):
        y = (y - 16*scale)/(219*scale)
        cb = (cb - 128*scale)/(224*scale)
        cr = (cr - 128*scale)/(224*scale)
    else:
        maxVal = 2**bits - 1
        y = y/maxVal
        cb = (cb - 2**(bits - 1))/maxVal
        cr = (cr - 2**(bits - 1))/maxVal
    kr, kb = YUV_WEIGHTS[matrix]
    r = y + 2*(1 - kr)*cr
    b = y + 2*(1 - kb)*cb
    g = (y - kr*r - kb*b)/(1 - kr - kb)
    return np.clip(a=np.dstack([r, g, b]), a_min=0, a_max=1).astype(np.float32)


def rawRGBCam(raw: object) -> np.ndarray | None:
    """Returns the camera -> linear sRGB matrix of a RAW file, whose rows sum to 1.

    Args:
        raw: The rawpy.RawPy.

    Returns:
        The 3×3 matrix from LibRaw's color_matrix (DNG) or rgb_xyz_matrix (its per-model table), or None when
        LibRaw has neither.
    """
    colorMatrix = np.array(raw.color_matrix, dtype=np.float64)[:3, :3]
    xyzToCam = np.array(raw.rgb_xyz_matrix, dtype=np.float64)[:3, :3]
    if(np.any(colorMatrix)):
        return colorMatrix
    elif(np.any(xyzToCam)):
        camRGB = xyzToCam @ SRGB.toXYZ
        camRGB /= camRGB.sum(axis=1, keepdims=True)
        return np.linalg.inv(camRGB)
    return None


def fromRAW(raw: object) -> tuple[DecodeParams, WorkingSpace]:
    """Describes a RAW file decoded linear in camera RGB, with its black level and without white balance.

    Args:
        raw: The rawpy.RawPy.

    Returns:
        The sensor levels and the camera's working space; UNTAGGED with a warning when LibRaw has no matrix.
    """
    black = [float(v) for v in raw.black_level_per_channel]
    if(len(black) >= 4 and raw.num_colors == 3):
        black = [black[0], (black[1] + black[3])/2, black[2]]
    params = DecodeParams(blackLevel=tuple(black[:3]), whiteLevel=float(raw.white_level))
    rgbCam = rawRGBCam(raw)
    if(rgbCam is None):
        warnings.warn("LibRaw has no color matrix for this camera, its colors are shown as sRGB", ColorWarning)
        return (params, UNTAGGED)
    return (params, WorkingSpace(toXYZ=SRGB.toXYZ @ rgbCam))


def normalizeBayerPattern(
    pattern: str,
    xOffset: int = 0,
    yOffset: int = 0,
    bottomUp: bool = False,
    height: int = 0,
) -> BayerPattern | None:
    """Normalizes a FITS Bayer pattern to the pattern at the first stored pixel.

    Args:
        pattern: BAYERPAT, e.g. "RGGB".
        xOffset: XBAYROFF, columns the pattern is shifted by.
        yOffset: YBAYROFF, rows the pattern is shifted by.
        bottomUp: Whether ROWORDER is BOTTOM-UP, so the pattern describes the image flipped vertically.
        height: Image height, used for bottom-up images.

    Returns:
        The 2×2 pattern of the stored data, or None when the pattern isn't a 2×2 RGB one.
    """
    pattern = pattern.strip().upper()
    if(len(pattern) != 4 or sorted(pattern) != ["B", "G", "G", "R"]):
        return None
    rows = [pattern[0:2], pattern[2:4]]
    if(yOffset % 2):
        rows = rows[::-1]
    if(xOffset % 2):
        rows = [row[::-1] for row in rows]
    if(bottomUp and height % 2 == 0):
        rows = rows[::-1]
    result = rows[0] + rows[1]
    if(result not in ("RGGB", "GRBG", "GBRG", "BGGR")):
        return None
    return result


def fromFITS(header: object, data: np.ndarray) -> DecodeParams:
    """Describes FITS data as sensor data: its Bayer pattern and levels.

    Args:
        header: The FITS header (astropy.io.fits.Header or a dict).
        data: The (first frame's) data.

    Returns:
        The decode parameters: no black level, since FITS has no standard keyword for one; white from DATAMAX or
        SATURATE, else the integer type's maximum, else 1.0 for float data within 0-1 and 65535 otherwise.
    """
    bayerPattern = None
    if("BAYERPAT" in header):
        bayerPattern = normalizeBayerPattern(
            pattern=str(header["BAYERPAT"]),
            xOffset=int(header.get("XBAYROFF", 0)),
            yOffset=int(header.get("YBAYROFF", 0)),
            bottomUp=str(header.get("ROWORDER", "")).strip().upper() == "BOTTOM-UP",
            height=data.shape[0],
        )
    if("DATAMAX" in header and float(header["DATAMAX"]) > 0):
        white = float(header["DATAMAX"])
    elif("SATURATE" in header and float(header["SATURATE"]) > 0):
        white = float(header["SATURATE"])
    elif(np.issubdtype(data.dtype, np.integer)):
        white = float(np.iinfo(data.dtype).max)
    elif(np.nanmin(data) >= 0 and np.nanmax(data) <= 1):
        white = 1.0
    else:
        white = 65535.0
    return DecodeParams(bayerPattern=bayerPattern, whiteLevel=white)


def pngChunks(file: str) -> list[tuple[bytes, bytes]]:
    """Returns the chunks of a PNG file before its first IDAT.

    Args:
        file: Path of the file.

    Returns:
        The chunks as (type, data), empty when the file isn't a PNG.
    """
    chunks = []
    with open(file, "rb") as f:
        if(f.read(8) != b"\x89PNG\r\n\x1a\n"):
            return chunks
        while(True):
            header = f.read(8)
            if(len(header) < 8 or header[4:8] == b"IDAT"):
                break
            length = struct.unpack(">I", header[:4])[0]
            chunks.append((header[4:8], f.read(length)))
            f.seek(4, 1)
    return chunks


def fromPNGChunks(chunks: list[tuple[bytes, bytes]], channels: int) -> tuple[DecodeParams, WorkingSpace] | None:
    """Describes a PNG from its color chunks, with the precedence cICP > iCCP > sRGB > cHRM/gAMA.

    Args:
        chunks: The PNG's chunks before IDAT.
        channels: Number of color channels of the image.

    Returns:
        The decode parameters and working space, or None when the PNG is untagged.
    """
    chunks = dict(chunks)
    if(b"cICP" in chunks):
        return fromCICP(primaries=chunks[b"cICP"][0], transfer=chunks[b"cICP"][1])
    if(b"iCCP" in chunks):
        data = chunks[b"iCCP"]
        icc = zlib.decompress(data[data.index(b"\0") + 2:])
        description = describeICC(icc=icc, channels=channels)
        if(description is not None):
            return description
    if(b"sRGB" in chunks):
        return (DecodeParams(transfer=("sRGB", None)), SRGB)
    if(b"gAMA" in chunks or b"cHRM" in chunks):
        workingSpace = UNTAGGED
        if(b"cHRM" in chunks):
            v = np.array(struct.unpack(">8I", chunks[b"cHRM"][:32]))/100000
            workingSpace = fromPrimaries(primaries=v[2:8].reshape(3, 2), white=v[0:2])
        transfer = None
        if(b"gAMA" in chunks):
            gAMA = struct.unpack(">I", chunks[b"gAMA"][:4])[0]
            if(gAMA == 45455 and workingSpace.toXYZ is None):
                return (DecodeParams(transfer=("sRGB", None)), SRGB)
            elif(gAMA == 45455 and workingSpace == SRGB):
                transfer = ("sRGB", None)
            elif(gAMA != 100000 and gAMA > 0):
                transfer = ("Gamma", 100000/gAMA)
        return (DecodeParams(transfer=transfer), workingSpace)
    return None


def readICC(file: str) -> bytes | None:
    """Returns the embedded ICC profile of a JPEG/TIFF file.

    Args:
        file: Path of the file.

    Returns:
        The profile bytes, or None.
    """
    try:
        with Image.open(file) as img:
            return img.info.get("icc_profile") or None
    except Exception:
        pass
    try:
        tifffile = lazy("tifffile")
        with tifffile.TiffFile(file) as tif:
            tag = tif.pages[0].tags.get("InterColorProfile")
            return None if tag is None else bytes(tag.value)
    except Exception:
        return None


def readEXIFColorSpace(file: str) -> tuple[int | None, str | None]:
    """Returns the EXIF ColorSpace tag and Interoperability Index of an image file.

    Args:
        file: Path of the file.

    Returns:
        The tags' values (1 = sRGB; 0xFFFF with "R03" = Adobe RGB), each None when there is none.
    """
    try:
        with Image.open(file) as img:
            exif = img.getexif()
            return (exif.get_ifd(0x8769).get(0xA001), exif.get_ifd(0xA005).get(1))
    except Exception:
        return (None, None)


def describeImage(file: str, channels: int) -> tuple[DecodeParams, WorkingSpace]:
    """Describes a (non RAW, non FITS) image file from its color tags.

    PNG: cICP > iCCP > sRGB > cHRM/gAMA. JPEG/TIFF: an embedded ICC profile, then EXIF ColorSpace = 1 as sRGB, or
    ColorSpace = 0xFFFF with Interoperability Index "R03" as Adobe RGB, which is how cameras mark it.

    Args:
        file: Path of the file.
        channels: Number of color channels of the image.

    Returns:
        The decode parameters and working space; untagged files have no transfer and no primaries.
    """
    chunks = pngChunks(file)
    if(chunks):
        description = fromPNGChunks(chunks=chunks, channels=channels)
    else:
        description = None
        icc = readICC(file)
        if(icc):
            description = describeICC(icc=icc, channels=channels)
        if(description is None):
            colorSpace, interopIndex = readEXIFColorSpace(file)
            if(colorSpace == 1):
                description = (DecodeParams(transfer=("sRGB", None)), SRGB)
            elif(colorSpace == 0xFFFF and interopIndex == "R03"):
                description = (DecodeParams(transfer=("Gamma", ADOBE_RGB_GAMMA)), named("Adobe RGB (1998)"))
    if(description is None):
        return (DecodeParams(), UNTAGGED)
    return description


@cache
def srgbProfile() -> bytes:
    """Returns LittleCMS's sRGB ICC profile, embedded in every saved file.

    Returns:
        The profile bytes.
    """
    return ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes()


def srgbPNGChunks() -> list[tuple[bytes, bytes]]:
    """Returns the chunks that tag a PNG as sRGB: cICP, iCCP, sRGB and gAMA.

    Returns:
        The chunks as (type, data).
    """
    return [(b"cICP", bytes([1, 13, 0, 1])),
            (b"iCCP", b"sRGB\0\0" + zlib.compress(srgbProfile())),
            (b"sRGB", bytes([0])),
            (b"gAMA", struct.pack(">I", 45455))]


def insertPNGChunks(png: bytes, chunks: list[tuple[bytes, bytes]]) -> bytes:
    """Inserts chunks into a PNG right after its IHDR.

    Args:
        png: The PNG file's bytes.
        chunks: The chunks to insert as (type, data).

    Returns:
        The new PNG file's bytes.
    """
    pypng = lazy("png")
    existing = list(pypng.Reader(bytes=png).chunks())
    out = io.BytesIO()
    pypng.write_chunks(out, existing[:1] + chunks + existing[1:])
    return out.getvalue()


SAVE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".tif", ".tiff")


def saveSRGB(file: str, img: np.ndarray, workingSpace: WorkingSpace | None, bits: int) -> None:
    """Saves linear working data as display-referred sRGB, tagged as sRGB.

    PNGs carry cICP, iCCP, sRGB and gAMA chunks; JPEGs and TIFFs carry the sRGB ICC profile.

    Args:
        file: Path of the file; its extension must be one of SAVE_EXTENSIONS.
        img: Linear RGB image (channels last, RGB order).
        workingSpace: The working data's colors.
        bits: 8 or 16 (16 only for PNG and TIFF).

    Raises:
        ValueError: The extension or bit depth isn't supported.
    """
    extension = file.lower()[file.rfind("."):] if "." in file else ""
    if(extension not in SAVE_EXTENSIONS):
        raise ValueError("Unsupported file extension " + repr(extension))
    if(bits == 16 and extension in (".jpg", ".jpeg")):
        raise ValueError("JPEG can't store 16-bit images")
    rgb = displayImage(img=img, workingSpace=workingSpace, bits=bits)
    if(extension == ".png"):
        cv2 = lazy("cv2")
        success, png = cv2.imencode(".png", np.ascontiguousarray(rgb[:, :, ::-1]))
        if(not success):
            raise ValueError("The PNG could not be encoded")
        with open(file, "wb") as f:
            f.write(insertPNGChunks(png=png.tobytes(), chunks=srgbPNGChunks()))
    elif(extension in (".tif", ".tiff") and bits == 16):
        tifffile = lazy("tifffile")
        tifffile.imwrite(file, data=rgb, photometric="rgb", iccprofile=srgbProfile())
    elif(extension in (".tif", ".tiff")):
        Image.fromarray(rgb, "RGB").save(fp=file, icc_profile=srgbProfile())
    else:
        Image.fromarray(rgb, "RGB").save(fp=file, quality=95, icc_profile=srgbProfile())
