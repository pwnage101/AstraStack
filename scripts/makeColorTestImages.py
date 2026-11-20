"""Generates the color test inputs and their references: one scene, stored in many encodings.

Every input decodes to the same scene-referred linear light, so the app should save each of them as the same image.
The scene is a chart of flat 32 px patches, so JPEG's 8x8 blocks and 4:2:0 chroma never straddle two colors. Most of
its colors lie inside the sRGB gamut, so every encoding holds them. The gray highlights above white (1.0) are only held
by the float TIFF master and the HLG and PQ videos, and the Rec.2020 primaries and secondaries outside sRGB only by
the BT.2020 encodings; the other encodings clip them per channel, as the app does when it saves sRGB.

The color science (color spaces, transfer functions and YCbCr matrices) is colour-science's; this script only
quantizes the encoded values and writes them into files.

Run from the repository root: ``make color-test-images``, or ``python scripts/makeColorTestImages.py``.
"""
import io
from fractions import Fraction
from pathlib import Path
import struct
from typing import Literal

import av
import colour
from av.video.reformatter import ColorPrimaries, ColorRange, Colorspace, ColorTrc, Interpolation
import cv2
import numpy as np
import png
import rawpy
import tifffile

TEST_FILES_DIR = Path("features")/"testFiles"
INPUTS_COLOR_DIR = TEST_FILES_DIR/"inputs"/"color"
REFERENCE_COLOR_DIR = TEST_FILES_DIR/"reference"/"color"

SRGB = colour.RGB_COLOURSPACES["sRGB"]
REC2020 = colour.RGB_COLOURSPACES["ITU-R BT.2020"]


def fromSRGB(colors: list[tuple[float, float, float]]) -> list[list[float]]:
    """Decodes sRGB-encoded colors, like those of hex codes and color pickers, to linear sRGB."""
    return colour.cctf_decoding(np.array(colors), function="sRGB").tolist()


def fromRec2020(colors: list[tuple[float, float, float]]) -> list[list[float]]:
    """Converts linear Rec.2020 colors to linear sRGB, which may be negative or above 1."""
    return colour.RGB_to_RGB(RGB=np.array(colors), input_colourspace=REC2020, output_colourspace=SRGB).tolist()


def halfSaturated(colors: list[tuple[float, float, float]]) -> list[list[float]]:
    """Mixes linear sRGB colors halfway to the gray of their own luminance."""
    rgb = np.array(colors)
    luminance = colour.RGB_luminance(RGB=rgb, primaries=SRGB.primaries, whitepoint=SRGB.whitepoint)[:, np.newaxis]
    return (luminance + 0.5*(rgb - luminance)).tolist()


PATCH_WIDTH = 32
# Pixels this close to a patch's edge aren't compared, since 4:2:0 chroma and JPEG compression blur across edges
MASK_MARGIN = 8

# Gray intensities UP TO reference white.
GRAYSCALE_BELOW = fromSRGB([(v, v, v) for v in (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)])
# Gray intensities ABOVE reference white.
# HLG can encode values up to 1.78 in the sRGB scale, so going up to 1.7
# decently covers the full range of the most common delivery HDR format.
GRAYSCALE_ABOVE = fromSRGB([(v, v, v) for v in (1.2, 1.3, 1.4, 1.5, 1.6, 1.7)])
# R,G,B,C,M,Y
COLORS_INSIDE_SRGB = fromSRGB([
    (1.0, 0.0, 0.0),
    (0.0, 1.0, 0.0),
    (0.0, 0.0, 1.0),
    (0.0, 1.0, 1.0),
    (1.0, 0.0, 1.0),
    (1.0, 1.0, 0.0),
])
# Less saturated versions, with the same luminances.
COLORS_HALF_SATURATED = halfSaturated(COLORS_INSIDE_SRGB)
# The same colors in Rec.2020, the primaries and secondaries outside sRGB.
COLORS_INSIDE_REC2020 = fromRec2020(COLORS_INSIDE_SRGB)
# The assembled scene in linear sRGB.
SCENE_ROWS = [
    GRAYSCALE_BELOW,
    COLORS_HALF_SATURATED,
    COLORS_INSIDE_SRGB,
    COLORS_INSIDE_REC2020,
    GRAYSCALE_ABOVE,
]
VIDEO_FRAMES = 3

# BT.2408 reference whites, which decode to 1.0: the scene light of 75% HLG signal, and that shown at 203 cd/m² by PQ
HLG_REFERENCE_WHITE = colour.oetf_inverse(0.75, function="ITU-R BT.2100 HLG")
PQ_REFERENCE_WHITE = colour.oetf_inverse(colour.eotf_inverse(203, function="ST 2084"), function="ITU-R BT.2100 PQ")


def makeScene() -> np.ndarray:
    """Returns the scene: linear sRGB, float64, (rows, columns, RGB)."""
    patches = np.array(SCENE_ROWS, dtype=np.float64)
    rows = np.repeat(a=patches, repeats=PATCH_WIDTH, axis=0)
    return np.repeat(a=rows, repeats=PATCH_WIDTH, axis=1)


def convertPrimaries(img: np.ndarray, colourspace: colour.RGB_Colourspace) -> np.ndarray:
    """Converts linear sRGB to another linear RGB color space, with negatives clipped to 0.

    Args:
        img: Linear sRGB image, (rows, columns, RGB).
        colourspace: The destination's color space.

    Returns:
        The linear image in the destination's color space.
    """
    return np.maximum(colour.RGB_to_RGB(RGB=img, input_colourspace=SRGB, output_colourspace=colourspace), 0)


def applySRGBCurve(linear: np.ndarray) -> np.ndarray:
    """Applies the sRGB transfer curve to linear values, extending its power curve above 1."""
    return colour.cctf_encoding(linear, function="sRGB")


def toInteger(img: np.ndarray, dtype: type) -> np.ndarray:
    """Quantizes 0-1 values to an integer type's full range.

    Note: values above 1 are permitted, but are clamped to 1.
    """
    maximum = np.iinfo(dtype).max
    return np.around(np.clip(a=img, a_min=0, a_max=1)*maximum).astype(dtype)


# Each transfer curve, from linear light to its 0-1 signal. HLG and PQ put 1.0 at their BT.2408 reference whites.
TRANSFER_CURVES = {
    "linear": lambda linear: linear,
    "sRGB": applySRGBCurve,
    "BT.709": lambda linear: colour.oetf(linear, function="ITU-R BT.709"),
    "gamma2.8": lambda linear: linear**(1/2.8),
    "HLG": lambda linear: colour.oetf(linear*HLG_REFERENCE_WHITE, function="ITU-R BT.2100 HLG"),
    "PQ": lambda linear: colour.oetf(linear*PQ_REFERENCE_WHITE, function="ITU-R BT.2100 PQ"),
}


def encode(img: np.ndarray, dtype: type, primaries: str, tc: str) -> np.ndarray:
    """Encodes linear sRGB in other primaries and transfer curve, then stores it as dtype.

    Args:
        img: Linear sRGB image, (rows, columns, RGB), or a gray one, (rows, columns).
        dtype: uint8 or uint16, which clip to 0-1, or float32, which keeps values above 1.
        primaries: "sRGB" or "BT.2020".
        tc: The transfer curve, one of TRANSFER_CURVES.

    Returns:
        The encoded image.
    """
    linear = np.maximum(img, 0) if primaries == "sRGB" else convertPrimaries(img=img, colourspace=REC2020)
    encoded = TRANSFER_CURVES[tc](linear)
    return encoded.astype(np.float32) if dtype == np.float32 else toInteger(encoded, dtype)


# The PNG chunks that tag each (primaries, transfer curve). cICP's are (primaries, transfer, matrix, full range), with
# matrix 0 for RGB.
PNG_CHUNKS = {
    (None, None): [],
    ("sRGB", "sRGB"): [(b"cICP", bytes([1, 13, 0, 1])), (b"sRGB", bytes([0]))],
    ("sRGB", "linear"): [
        (b"cICP", bytes([1, 8, 0, 1])),
        (b"gAMA", struct.pack(">I", 100000)),
        (b"cHRM", struct.pack(">8I", *(round(v*100000) for v in [*SRGB.whitepoint, *np.ravel(SRGB.primaries)]))),
    ],
    ("BT.2020", "linear"): [(b"cICP", bytes([9, 8, 0, 1]))],
}


def writePNG(
    img: np.ndarray,
    path: Path,
    dtype: type,
    encodingPrimaries: str,
    encodingTc: str,
    metadataPrimaries: str | None,
    metadataTc: str | None,
) -> None:
    """Writes an RGB PNG, with its color chunks right after IHDR.

    Args:
        img: Linear sRGB image, (rows, columns, RGB).
        path: Where to write it.
        dtype: uint8 or uint16.
        encodingPrimaries: The primaries the values are encoded in.
        encodingTc: The transfer curve the values are encoded with.
        metadataPrimaries: The primaries the file is tagged with; None for untagged.
        metadataTc: The transfer curve the file is tagged with; None for untagged.
    """
    pixels = encode(img=img, dtype=dtype, primaries=encodingPrimaries, tc=encodingTc)
    height, width = pixels.shape[:2]
    data = io.BytesIO()
    writer = png.Writer(width=width, height=height, greyscale=False, bitdepth=pixels.dtype.itemsize*8)
    writer.write(data, pixels.reshape(height, -1))
    existing = list(png.Reader(bytes=data.getvalue()).chunks())
    with open(path, "wb") as f:
        png.write_chunks(f, existing[:1] + PNG_CHUNKS[(metadataPrimaries, metadataTc)] + existing[1:])


def writeTIFF(
    img: np.ndarray,
    path: Path,
    dtype: type,
    encodingPrimaries: str,
    encodingTc: str,
    metadataPrimaries: None,
    metadataTc: None,
) -> None:
    """Writes an RGB TIFF; the app reads no TIFF color tags, so none are written.

    Args:
        img: Linear sRGB image, (rows, columns, RGB).
        path: Where to write it.
        dtype: uint8, uint16 or float32.
        encodingPrimaries: The primaries the values are encoded in.
        encodingTc: The transfer curve the values are encoded with.
        metadataPrimaries: Must be None (untagged).
        metadataTc: Must be None (untagged).
    """
    assert metadataPrimaries is None and metadataTc is None, "TIFF color tags aren't written"
    tifffile.imwrite(
        path,
        data=encode(img=img, dtype=dtype, primaries=encodingPrimaries, tc=encodingTc),
        photometric="rgb",
        compression="zlib",
        metadata=None,
        software=None,
    )


def writeJPEG(
    img: np.ndarray,
    path: Path,
    encodingPrimaries: str,
    encodingTc: str,
    metadataPrimaries: None,
    metadataTc: None,
) -> None:
    """Writes an 8-bit RGB JPEG at quality 95 without chroma subsampling; the app reads no JPEG color tags, so none
    are written.

    Args:
        img: Linear sRGB image, (rows, columns, RGB).
        path: Where to write it.
        encodingPrimaries: The primaries the values are encoded in.
        encodingTc: The transfer curve the values are encoded with.
        metadataPrimaries: Must be None (untagged).
        metadataTc: Must be None (untagged).
    """
    assert metadataPrimaries is None and metadataTc is None, "JPEG color tags aren't written"
    cv2.imwrite(
        path,
        encode(img=img, dtype=np.uint8, primaries=encodingPrimaries, tc=encodingTc)[..., ::-1],
        [cv2.IMWRITE_JPEG_QUALITY, 95, cv2.IMWRITE_JPEG_SAMPLING_FACTOR, cv2.IMWRITE_JPEG_SAMPLING_FACTOR_444],
    )


def writeImages(scene: np.ndarray) -> None:
    """Writes the TIFF, PNG and JPEG encodings of the scene.

    The untagged images have the sRGB curve, as the app assumes for them whatever their type.

    Args:
        scene: The scene in linear sRGB.
    """
    # Integer TIFF is always assumed to be sRGB/sRGB
    writeTIFF(
        img=scene,
        path=INPUTS_COLOR_DIR/"tiff16_none_none.tif",
        dtype=np.uint16,
        encodingPrimaries="sRGB",
        encodingTc="sRGB",
        metadataPrimaries=None,
        metadataTc=None,
    )
    writeTIFF(
        img=scene,
        path=INPUTS_COLOR_DIR/"tiff8_none_none.tif",
        dtype=np.uint8,
        encodingPrimaries="sRGB",
        encodingTc="sRGB",
        metadataPrimaries=None,
        metadataTc=None,
    )
    
    # Float TIFF is too, and its sRGB curve continues above white
    writeTIFF(
        img=scene,
        path=INPUTS_COLOR_DIR/"tiff32f_none_none.tif",
        dtype=np.float32,
        encodingPrimaries="sRGB",
        encodingTc="sRGB",
        metadataPrimaries=None,
        metadataTc=None,
    )

    # PNG samples that are untagged and should be assumed to have sRGB primaries / sRGB transfer curve.
    writePNG(
        img=scene,
        path=INPUTS_COLOR_DIR/"png16_none_none.png",
        dtype=np.uint16,
        encodingPrimaries="sRGB",
        encodingTc="sRGB",
        metadataPrimaries=None,
        metadataTc=None,
    )
    writePNG(
        img=scene,
        path=INPUTS_COLOR_DIR/"png8_none_none.png",
        dtype=np.uint8,
        encodingPrimaries="sRGB",
        encodingTc="sRGB",
        metadataPrimaries=None,
        metadataTc=None,
    )

    # PNG samples that are accurately tagged:
    writePNG(
        img=scene,
        path=INPUTS_COLOR_DIR/"png16_srgb_linear.png",
        dtype=np.uint16,
        encodingPrimaries="sRGB",
        encodingTc="linear",
        metadataPrimaries="sRGB",
        metadataTc="linear",
    )
    writePNG(
        img=scene,
        path=INPUTS_COLOR_DIR/"png16_srgb_srgb.png",
        dtype=np.uint16,
        encodingPrimaries="sRGB",
        encodingTc="sRGB",
        metadataPrimaries="sRGB",
        metadataTc="sRGB",
    )
    writePNG(
        img=scene,
        path=INPUTS_COLOR_DIR/"png16_bt2020_linear.png",
        dtype=np.uint16,
        encodingPrimaries="BT.2020",
        encodingTc="linear",
        metadataPrimaries="BT.2020",
        metadataTc="linear",
    )

    # Untagged JPEG is assumed to be encoded with sRGB primaries / sRGB transfer curve.
    writeJPEG(
        img=scene,
        path=INPUTS_COLOR_DIR/"jpg_none_none.jpg",
        encodingPrimaries="sRGB",
        encodingTc="sRGB",
        metadataPrimaries=None,
        metadataTc=None,
    )


def toYUV(pixels: np.ndarray, pixFmt: str, matrix: Colorspace, colorRange: ColorRange) -> av.VideoFrame:
    """Converts encoded RGB to a 4:2:0 YUV video frame with FFmpeg.

    Area interpolation averages chroma within each 2x2 block, so it doesn't bleed into neighboring patches, and RGB
    of the frame's own depth leaves FFmpeg nothing to dither.

    Args:
        pixels: Encoded RGB, uint8 or uint16, (rows, columns, RGB).
        pixFmt: The YUV pixel format.
        matrix: The YUV matrix.
        colorRange: The range of the YUV samples.

    Returns:
        The frame.
    """
    frame = av.VideoFrame.from_ndarray(pixels, format="rgb48le" if pixels.dtype == np.uint16 else "rgb24")
    return frame.reformat(
        format=pixFmt,
        dst_colorspace=matrix,
        dst_color_range=colorRange,
        interpolation=Interpolation.AREA,
    )


# The lossless encoder for each container and bit depth, with its options and YUV pixel format; AVI stores the RGB or
# gray pixels uncompressed, like capture programs' AVIs
CODECS = {
    (".mp4", 8): ("libx264", {"qp": "0"}, "yuv420p"),
    (".mp4", 10): ("libx265", {"x265-params": "lossless=1:log-level=error"}, "yuv420p10le"),
    (".avi", 8): ("rawvideo", {}, None),
}
# The YUV matrix that goes with each primaries
MATRICES = {"sRGB": Colorspace.ITU709, "BT.2020": Colorspace.BT2020}
# The video tag of each primaries and transfer curve; BT.709's primaries are sRGB's
PRIMARIES_TAGS = {None: ColorPrimaries.UNSPECIFIED, "sRGB": ColorPrimaries.BT709, "BT.2020": ColorPrimaries.BT2020}
TC_TAGS = {
    None: ColorTrc.UNSPECIFIED,
    "sRGB": ColorTrc.IEC61966_2_1,
    "BT.709": ColorTrc.BT709,
    "gamma2.8": ColorTrc.GAMMA28,
    "HLG": ColorTrc.ARIB_STD_B67,
    "PQ": ColorTrc.SMPTE2084,
}
# The video tag of each range; "limited" is FFmpeg's MPEG range, and "full" its JPEG range
RANGE_TAGS = {None: ColorRange.UNSPECIFIED, "limited": ColorRange.MPEG, "full": ColorRange.JPEG}


def writeVideo(
    img: np.ndarray,
    path: Path,
    bitDepth: int,
    encodingPrimaries: str,
    encodingTc: str,
    encodingRange: Literal["limited", "full"] | None,
    metadataPrimaries: str | None = None,
    metadataTc: str | None = None,
    metadataRange: Literal["limited", "full"] | None = None,
) -> None:
    """Writes VIDEO_FRAMES copies of an image, encoded and tagged.

    Args:
        img: Linear sRGB image, (rows, columns, RGB), or a gray one, (rows, columns).
        path: Where to write it; its extension picks the container.
        bitDepth: Bits per sample; with the container, it picks the encoder from CODECS.
        encodingPrimaries: The primaries the values are encoded in, which pick the YUV matrix from MATRICES.
        encodingTc: The transfer curve the values are encoded with.
        encodingRange: The range the YUV values are encoded in; None for AVI, which stores RGB or gray.
        metadataPrimaries: The primaries the file is tagged with, along with their YUV matrix; None for untagged.
        metadataTc: The transfer curve the file is tagged with; None for untagged.
        metadataRange: The range the file is tagged with; None for untagged.
    """
    codec, options, yuvFormat = CODECS[(path.suffix, bitDepth)]
    pixels = encode(
        img=img,
        dtype=np.uint16 if bitDepth > 8 else np.uint8,
        primaries=encodingPrimaries,
        tc=encodingTc,
    )
    if(yuvFormat is not None):
        frame = toYUV(
            pixels=pixels,
            pixFmt=yuvFormat,
            matrix=MATRICES[encodingPrimaries],
            colorRange=RANGE_TAGS[encodingRange],
        )
    elif(pixels.ndim == 2):
        frame = av.VideoFrame.from_ndarray(pixels, format="gray")
    else:
        frame = av.VideoFrame.from_ndarray(pixels, format="rgb24").reformat(format="bgr24")
    with av.open(path, "w") as container:
        stream = container.add_stream(codec_name=codec, rate=10, options=options)
        stream.width = frame.width
        stream.height = frame.height
        stream.pix_fmt = frame.format.name
        stream.codec_context.color_primaries = PRIMARIES_TAGS[metadataPrimaries]
        stream.codec_context.color_trc = TC_TAGS[metadataTc]
        if(metadataPrimaries is not None):
            stream.codec_context.colorspace = MATRICES[metadataPrimaries]
        stream.codec_context.color_range = RANGE_TAGS[metadataRange]
        for index in range(VIDEO_FRAMES):
            frame.pts = index
            container.mux(stream.encode(frame))
        container.mux(stream.encode())


def writeVideos(scene: np.ndarray) -> None:
    """Writes the video encodings of the scene.

    Untagged YUV stores BT.709 values, and the untagged AVIs sRGB values, as the app assumes for them.

    Args:
        scene: The scene in linear sRGB.
    """
    writeVideo(
        img=scene,
        path=INPUTS_COLOR_DIR/"mp4_8_none_none_full.mp4",
        bitDepth=8,
        encodingPrimaries="sRGB",
        encodingTc="BT.709",
        encodingRange="full",
        metadataPrimaries=None,
        metadataTc=None,
        metadataRange="full",
    )
    writeVideo(
        img=scene,
        path=INPUTS_COLOR_DIR/"mp4_8_none_none_none.mp4",
        bitDepth=8,
        encodingPrimaries="sRGB",
        encodingTc="BT.709",
        encodingRange="limited",
        metadataPrimaries=None,
        metadataTc=None,
        metadataRange=None,
    )
    writeVideo(
        img=scene,
        path=INPUTS_COLOR_DIR/"mp4_8_bt709_bt709_limited.mp4",
        bitDepth=8,
        encodingPrimaries="sRGB",
        encodingTc="BT.709",
        encodingRange="limited",
        metadataPrimaries="sRGB",
        metadataTc="BT.709",
        metadataRange="limited",
    )
    writeVideo(
        img=scene,
        path=INPUTS_COLOR_DIR/"mp4_8_bt709_none_limited.mp4",
        bitDepth=8,
        encodingPrimaries="sRGB",
        encodingTc="BT.709",
        encodingRange="limited",
        metadataPrimaries="sRGB",
        metadataTc=None,
        metadataRange="limited",
    )
    writeVideo(
        img=scene,
        path=INPUTS_COLOR_DIR/"mp4_8_bt709_srgb_full.mp4",
        bitDepth=8,
        encodingPrimaries="sRGB",
        encodingTc="sRGB",
        encodingRange="full",
        metadataPrimaries="sRGB",
        metadataTc="sRGB",
        metadataRange="full",
    )
    # A legacy transfer characteristic, BT.470 BG's gamma 2.8
    writeVideo(
        img=scene,
        path=INPUTS_COLOR_DIR/"mp4_8_bt709_gamma28_limited.mp4",
        bitDepth=8,
        encodingPrimaries="sRGB",
        encodingTc="gamma2.8",
        encodingRange="limited",
        metadataPrimaries="sRGB",
        metadataTc="gamma2.8",
        metadataRange="limited",
    )
    # Like a Nikon Z5II's HLG footage: 10-bit, HLG, BT.2020, full range
    writeVideo(
        img=scene,
        path=INPUTS_COLOR_DIR/"mp4_10_bt2020_hlg_full.mp4",
        bitDepth=10,
        encodingPrimaries="BT.2020",
        encodingTc="HLG",
        encodingRange="full",
        metadataPrimaries="BT.2020",
        metadataTc="HLG",
        metadataRange="full",
    )
    # Like HDR10 common for distribution: 10-bit, PQ, BT.2020, limited range
    writeVideo(
        img=scene,
        path=INPUTS_COLOR_DIR/"mp4_10_bt2020_pq_limited.mp4",
        bitDepth=10,
        encodingPrimaries="BT.2020",
        encodingTc="PQ",
        encodingRange="limited",
        metadataPrimaries="BT.2020",
        metadataTc="PQ",
        metadataRange="limited",
    )
    writeVideo(
        img=scene,
        path=INPUTS_COLOR_DIR/"avi8_rgb.avi",
        bitDepth=8,
        encodingPrimaries="sRGB",
        encodingTc="sRGB",
        encodingRange=None,
    )
    writeVideo(
        img=toGray(scene, outputChannels=1),
        path=INPUTS_COLOR_DIR/"avi8_gray.avi",
        bitDepth=8,
        encodingPrimaries="sRGB",
        encodingTc="sRGB",
        encodingRange=None,
    )

# A camera's DNG ColorMatrix at D65 (XYZ to camera RGB; Canon EOS 5D Mark III's, from dcraw)
CAMERA_XYZ_TO_CAM = np.array([[6722, -635, -963], [-4287, 12460, 2028], [-908, 2162, 5668]])/10000
# The sensor's levels; neither is 0 or the 16-bit maximum
RAW_BLACK_LEVEL = 512
RAW_WHITE_LEVEL = 15000
# The as-shot white balance, relative to the D65 reference's: warmer than daylight
RAW_AS_SHOT_SHIFT = np.array([0.75, 1.0, 1.1])
# The scene is stored at this exposure, since a sensor holds no values above its white level; the scene's
# brightest patches are 1.7. The test restores it with the Process Value slider.
RAW_EXPOSURE = 0.25


def toRationals(values: np.ndarray) -> list[int]:
    """Flattens numbers to the numerator, denominator pairs of a DNG (S)RATIONAL tag."""
    return [part for v in np.ravel(values) for part in Fraction(float(v)).limit_denominator(10000).as_integer_ratio()]


def writeRaw(scene: np.ndarray) -> None:
    """Writes the scene as a Bayer RGGB DNG, as a camera would store it.

    The stored values undo what LibRaw and the app do: black and white levels, then the as-shot white balance
    swapped for the D65 reference's, then the camera matrix. So the app, decoding by that model, gets the scene back.
    The file states the D65 ColorMatrix and a real AsShotNeutral, not D65.

    Args:
        scene: The scene in linear sRGB.
    """
    # LibRaw's camera -> sRGB matrix, normalized so its rows sum to 1, and its D65 reference multipliers
    camRGB = CAMERA_XYZ_TO_CAM @ SRGB.matrix_RGB_to_XYZ
    daylight = 1/camRGB.sum(axis=1)
    daylight /= daylight[1]
    camRGB /= camRGB.sum(axis=1, keepdims=True)
    # The camera's D65-balanced values, then what the sensor records before the white balance
    balanced = RAW_EXPOSURE*scene @ camRGB.T
    raw = RAW_BLACK_LEVEL + (RAW_WHITE_LEVEL - RAW_BLACK_LEVEL)*balanced/daylight
    mosaic = np.zeros(raw.shape[:2])
    mosaic[0::2, 0::2] = raw[0::2, 0::2, 0]
    mosaic[0::2, 1::2] = raw[0::2, 1::2, 1]
    mosaic[1::2, 0::2] = raw[1::2, 0::2, 1]
    mosaic[1::2, 1::2] = raw[1::2, 1::2, 2]
    # LibRaw clips what the as-shot multipliers scale beyond the smallest one's white level
    asShot = daylight*RAW_AS_SHOT_SHIFT
    limits = RAW_WHITE_LEVEL*asShot.min()/asShot[[0, 1, 1, 2]].reshape(2, 2)
    assert np.all(mosaic < np.tile(limits, (mosaic.shape[0]//2, mosaic.shape[1]//2))), "LibRaw would clip the scene"
    path = INPUTS_COLOR_DIR/"dng16_camera_none.dng"
    tifffile.imwrite(
        path,
        data=np.around(np.clip(mosaic, 0, 65535)).astype(np.uint16),
        photometric=32803,
        extratags=[
            (271, 2, 0, "AstraStack", True),  # Make
            (272, 2, 0, "Color test camera", True),  # Model
            (50706, 1, 4, (1, 4, 0, 0), True),  # DNGVersion
            (50708, 2, 0, "AstraStack Color test camera", True),  # UniqueCameraModel
            (33421, 3, 2, (2, 2), True),  # CFARepeatPatternDim
            (33422, 1, 4, (0, 1, 1, 2), True),  # CFAPattern: RGGB
            (50714, 3, 1, RAW_BLACK_LEVEL, True),
            (50717, 3, 1, RAW_WHITE_LEVEL, True),
            (50721, 10, 9, toRationals(CAMERA_XYZ_TO_CAM), True),  # ColorMatrix1
            (50778, 3, 1, 21, True),  # CalibrationIlluminant1: D65
            (50728, 5, 3, toRationals(1/asShot), True),  # AsShotNeutral
        ],
    )
    # The file reads back as written
    with rawpy.imread(str(path)) as dng:
        assert np.allclose(dng.camera_whitebalance[:3]/np.array(dng.camera_whitebalance[1]), asShot, atol=1e-3)
        assert np.allclose(np.array(dng.daylight_whitebalance[:3])/dng.daylight_whitebalance[1], daylight, atol=1e-3)
        assert np.allclose(dng.color_matrix[:3, :3], np.linalg.inv(camRGB), atol=1e-3)
        assert dng.black_level_per_channel == [RAW_BLACK_LEVEL]*4 and dng.white_level == RAW_WHITE_LEVEL


def toGray(img: np.ndarray, outputChannels: Literal[1, 3]) -> np.ndarray:
    """Convert to grayscale (using luminance), as a 2-D image or with the gray in all three channels."""
    luminance = colour.RGB_luminance(RGB=img, primaries=SRGB.primaries, whitepoint=SRGB.whitepoint)
    return luminance if outputChannels == 1 else np.repeat(a=luminance[..., None], repeats=3, axis=2)


def clipInRec2020(img: np.ndarray) -> np.ndarray:
    """Clips a linear sRGB image to 0-1 in Rec.2020's channels, as a BT.2020 integer encoding does."""
    rec2020 = np.clip(convertPrimaries(img=img, colourspace=REC2020), 0, 1)
    return colour.RGB_to_RGB(RGB=rec2020, input_colourspace=REC2020, output_colourspace=SRGB)


def toMask(img: np.ndarray) -> np.ndarray:
    """Returns a mono mask that's 255 inside the patches, away from their edges, and 0 elsewhere."""
    height, width = img.shape[:2]
    rows = np.arange(height) % PATCH_WIDTH
    columns = np.arange(width) % PATCH_WIDTH
    inside = np.outer((rows >= MASK_MARGIN) & (rows < PATCH_WIDTH - MASK_MARGIN),
                      (columns >= MASK_MARGIN) & (columns < PATCH_WIDTH - MASK_MARGIN))
    return inside.astype(np.uint8)*255


def writeReferences(scene: np.ndarray) -> None:
    """Writes what the app saves as 8-bit sRGB PNG for the color inputs, and where to compare them.

    Args:
        scene: The scene in linear sRGB.
    """
    # What every input saves as by default, with values above white clipped
    writePNG(
        img=scene,
        path=REFERENCE_COLOR_DIR/"master_sRGB_+0EV.png",
        dtype=np.uint8,
        encodingPrimaries="sRGB",
        encodingTc="sRGB",
        metadataPrimaries="sRGB",
        metadataTc="sRGB",
    )
    # What the gray input saves as
    writePNG(
        img=toGray(scene, outputChannels=3),
        path=REFERENCE_COLOR_DIR/"master_gray_sRGB_+0EV.png",
        dtype=np.uint8,
        encodingPrimaries="sRGB",
        encodingTc="sRGB",
        metadataPrimaries="sRGB",
        metadataTc="sRGB",
    )
    # At -2EV (the Value slider at 25%), the brightest patch falls below white, so these show what each input clipped
    # What the inputs that hold values above white save as
    writePNG(
        img=0.25*scene,
        path=REFERENCE_COLOR_DIR/"master_sRGB_-2EV.png",
        dtype=np.uint8,
        encodingPrimaries="sRGB",
        encodingTc="sRGB",
        metadataPrimaries="sRGB",
        metadataTc="sRGB",
    )
    # What the inputs that clipped each sRGB channel at white save as
    writePNG(
        img=0.25*np.clip(scene, 0, 1),
        path=REFERENCE_COLOR_DIR/"master_sRGB_clippedSRGB_-2EV.png",
        dtype=np.uint8,
        encodingPrimaries="sRGB",
        encodingTc="sRGB",
        metadataPrimaries="sRGB",
        metadataTc="sRGB",
    )
    # What the BT.2020 PNG saves as, having clipped each Rec.2020 channel at white
    writePNG(
        img=0.25*clipInRec2020(scene),
        path=REFERENCE_COLOR_DIR/"master_sRGB_clippedRec2020_-2EV.png",
        dtype=np.uint8,
        encodingPrimaries="sRGB",
        encodingTc="sRGB",
        metadataPrimaries="sRGB",
        metadataTc="sRGB",
    )
    # What the gray input saves as, having clipped its luminance at white
    writePNG(
        img=0.25*np.clip(toGray(scene, outputChannels=3), 0, 1),
        path=REFERENCE_COLOR_DIR/"master_gray_sRGB_clipped_-2EV.png",
        dtype=np.uint8,
        encodingPrimaries="sRGB",
        encodingTc="sRGB",
        metadataPrimaries="sRGB",
        metadataTc="sRGB",
    )
    # Mask used by tests to avoid comparing edges which contain compression/subsampling artifacts.
    cv2.imwrite(REFERENCE_COLOR_DIR/"mask.png", toMask(scene))


def main() -> None:
    """Writes every color input and reference."""
    scene = makeScene()
    writeImages(scene)
    writeVideos(scene)
    writeRaw(scene)
    writeReferences(scene)


if __name__ == "__main__":
    main()
