"""Generates the color test inputs and their references: one scene, stored in many encodings.

Every input decodes to the same scene-referred linear light, so the app should save each of them as the same image.
The scene is a chart of flat 32 px patches, so JPEG's 8x8 blocks and 4:2:0 chroma never straddle two colors. Its
colors lie inside the sRGB gamut, so every encoding holds them, except for the gray highlights above white (1.0),
which only the float TIFF master and the HLG and PQ videos hold; the other encodings clip them to white.

The color science (color spaces, transfer functions and YCbCr matrices) is colour-science's, and the ICC profiles are
checked in; this script only quantizes the encoded values and writes them into files. The profiles are:

- Rec2020-elle-V4-g10.icc, "Rec2020-elle-V4-g10" by Elle Stone (https://github.com/ellelstone/elles_icc_profiles),
  licensed under CC BY-SA 3.0 (https://creativecommons.org/licenses/by-sa/3.0/), unmodified.
- DisplayP3-v4.icc, from Compact ICC Profiles by Clinton Ingram (https://github.com/saucecontrol/Compact-ICC-Profiles),
  dedicated to the public domain under CC0 1.0.

Run from anywhere: ``python features/testFiles/inputs/color/makeColorInputs.py``.
"""
import io
import os
import struct

import av
import colour
from av.video.reformatter import ColorPrimaries, ColorRange, Colorspace, ColorTrc, Interpolation
import numpy as np
import png
import tifffile
from PIL import ExifTags, Image

COLOR_DIR = os.path.dirname(os.path.abspath(__file__))
REFERENCE_DIR = os.path.join(COLOR_DIR, "..", "..", "reference")

SRGB = colour.RGB_COLOURSPACES["sRGB"]
DISPLAY_P3 = colour.RGB_COLOURSPACES["Display P3"]
REC2020 = colour.RGB_COLOURSPACES["ITU-R BT.2020"]
ADOBE_RGB = colour.RGB_COLOURSPACES["Adobe RGB (1998)"]


def halfSaturated(colors: list[tuple[float, float, float]]) -> list[list[float]]:
    """Mixes linear sRGB colors halfway to the gray of their own luminance."""
    rgb = np.array(colors)
    luminance = colour.RGB_luminance(RGB=rgb, primaries=SRGB.primaries, whitepoint=SRGB.whitepoint)[:, np.newaxis]
    return (luminance + 0.5*(rgb - luminance)).tolist()


# R,G,B,C,M,Y, then brown and navy blue.
SATURATED = [
    (1.0,  0.0,  0.0),
    (0.0,  1.0,  0.0),
    (0.0,  0.0,  1.0),
    (0.0,  1.0,  1.0),
    (1.0,  0.0,  1.0),
    (1.0,  1.0,  0.0),
    (0.2,  0.1,  0.02),
    (0.02, 0.05, 0.1),
]

PATCH_WIDTH = 32
# The scene in linear sRGB, one row of patches per list; 1.0 is diffuse white
SCENE_ROWS = [
    # Gray intensities BELOW reference white.
    [(v, v, v) for v in (0.0, 0.02, 0.05, 0.1, 0.2, 0.4, 0.7, 1.0)],
    # Less saturated versions of the next row, with the same luminances.
    halfSaturated(SATURATED),
    SATURATED,
    # Gray intensities ABOVE reference white.
    [(v, v, v) for v in (1.0, 1.15, 1.3, 1.5, 1.7, 1.9, 2.5, 3.5)],
]
VIDEO_FRAMES = 3

# Adobe RGB (1998)'s gamma, 2 51/256
ADOBE_RGB_GAMMA = 563/256

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


def encodeSRGB(linear: np.ndarray) -> np.ndarray:
    """Applies the sRGB encoding to linear 0-1 values."""
    return colour.cctf_encoding(linear, function="sRGB")


def toInteger(img: np.ndarray, dtype: type) -> np.ndarray:
    """Quantizes 0-1 values to an integer type's full range."""
    maximum = np.iinfo(dtype).max
    return np.around(np.clip(a=img, a_min=0, a_max=1)*maximum).astype(dtype)


def writePNG(name: str, img: np.ndarray, chunks: list[tuple[bytes, bytes]], folder: str = COLOR_DIR) -> None:
    """Writes an RGB PNG with extra chunks right after IHDR.

    Args:
        name: File name.
        img: RGB image, uint8 or uint16.
        chunks: The extra chunks as (type, data).
        folder: Where to write it.
    """
    height, width = img.shape[:2]
    data = io.BytesIO()
    writer = png.Writer(width=width, height=height, greyscale=False, bitdepth=img.dtype.itemsize*8)
    writer.write(data, img.reshape(height, -1))
    existing = list(png.Reader(bytes=data.getvalue()).chunks())
    with open(os.path.join(folder, name), "wb") as f:
        png.write_chunks(f, existing[:1] + chunks + existing[1:])


def readProfile(name: str) -> bytes:
    """Returns the bytes of a checked-in ICC profile."""
    with open(os.path.join(COLOR_DIR, name), "rb") as f:
        return f.read()


def writeImages(scene: np.ndarray) -> None:
    """Writes the float TIFF master and the PNG and JPEG encodings of the scene.

    Args:
        scene: The scene in linear sRGB.
    """
    rec2020 = convertPrimaries(img=scene, colourspace=REC2020).astype(np.float32)
    tifffile.imwrite(
        os.path.join(COLOR_DIR, "tiff32f_rec2020_linear.tif"),
        data=rec2020,
        photometric="rgb",
        compression="zlib",
        iccprofile=readProfile("Rec2020-elle-V4-g10.icc"),
        metadata=None,
        software=None,
    )

    white = np.clip(a=scene, a_min=0, a_max=1)
    gAMA = (b"gAMA", struct.pack(">I", 100000))
    chromaticities = np.concatenate([SRGB.whitepoint, np.ravel(SRGB.primaries)])
    cHRM = (b"cHRM", struct.pack(">8I", *(round(v*100000) for v in chromaticities)))
    writePNG(name="png16_srgb_linear.png", img=toInteger(white, np.uint16), chunks=[gAMA, cHRM])
    writePNG(name="png16_untagged.png", img=toInteger(white, np.uint16), chunks=[])
    writePNG(name="png16_srgb_srgb.png", img=toInteger(encodeSRGB(white), np.uint16), chunks=[(b"sRGB", bytes([0]))])

    exif = Image.Exif()
    exif.get_ifd(ExifTags.IFD.Exif)[ExifTags.Base.ColorSpace] = 1
    Image.fromarray(toInteger(encodeSRGB(white), np.uint8)).save(
        fp=os.path.join(COLOR_DIR, "jpg_srgb_srgb.jpg"),
        quality=95,
        subsampling=0,
        exif=exif,
    )
    p3 = encodeSRGB(np.clip(a=convertPrimaries(img=white, colourspace=DISPLAY_P3), a_min=0, a_max=1))
    Image.fromarray(toInteger(p3, np.uint8)).save(
        fp=os.path.join(COLOR_DIR, "jpg_p3_srgb.jpg"),
        quality=95,
        subsampling=0,
        icc_profile=readProfile("DisplayP3-v4.icc"),
    )

    # Cameras mark Adobe RGB with EXIF rather than a profile: ColorSpace uncalibrated, and the DCF option file's
    # Interoperability Index
    exif = Image.Exif()
    exifIFD = exif.get_ifd(ExifTags.IFD.Exif)
    exifIFD[ExifTags.Base.ColorSpace] = 0xFFFF
    exifIFD[ExifTags.IFD.Interop] = {ExifTags.Interop.InteropIndex: "R03"}
    adobeLinear = np.clip(a=convertPrimaries(img=white, colourspace=ADOBE_RGB), a_min=0, a_max=1)
    adobe = colour.gamma_function(a=adobeLinear, exponent=1/ADOBE_RGB_GAMMA)
    Image.fromarray(toInteger(adobe, np.uint8)).save(
        fp=os.path.join(COLOR_DIR, "jpg_adobergb_gamma22.jpg"),
        quality=95,
        subsampling=0,
        exif=exif,
    )


def toYUV(rgb: np.ndarray, pixFmt: str, matrix: Colorspace, colorRange: ColorRange) -> av.VideoFrame:
    """Converts encoded RGB to a 4:2:0 YUV video frame with FFmpeg.

    Area interpolation averages chroma within each 2x2 block, so it doesn't bleed into neighboring patches, and RGB
    of the frame's own depth leaves FFmpeg nothing to dither.

    Args:
        rgb: Encoded RGB, 0-1, (rows, columns, RGB).
        pixFmt: The YUV pixel format.
        matrix: The YUV matrix.
        colorRange: The range of the YUV samples.

    Returns:
        The frame.
    """
    if(pixFmt.endswith("10le")):
        frame = av.VideoFrame.from_ndarray(toInteger(rgb, np.uint16), format="rgb48le")
    else:
        frame = av.VideoFrame.from_ndarray(toInteger(rgb, np.uint8), format="rgb24")
    return frame.reformat(
        format=pixFmt,
        dst_colorspace=matrix,
        dst_color_range=colorRange,
        interpolation=Interpolation.AREA,
    )


def writeVideo(
    name: str,
    frame: av.VideoFrame,
    codec: str,
    options: dict[str, str],
    primaries: ColorPrimaries = ColorPrimaries.UNSPECIFIED,
    transfer: ColorTrc = ColorTrc.UNSPECIFIED,
    matrix: Colorspace | None = None,
    colorRange: ColorRange = ColorRange.UNSPECIFIED,
) -> None:
    """Writes VIDEO_FRAMES copies of a frame, in its pixel format, with color tags.

    Args:
        name: File name; its extension picks the container.
        frame: The frame.
        codec: The encoder.
        options: The encoder's options.
        primaries: The colour primaries tag.
        transfer: The transfer characteristics tag.
        matrix: The matrix coefficients tag; None leaves it unspecified.
        colorRange: The color range tag.
    """
    with av.open(os.path.join(COLOR_DIR, name), "w") as container:
        stream = container.add_stream(codec_name=codec, rate=10, options=options)
        stream.width = frame.width
        stream.height = frame.height
        stream.pix_fmt = frame.format.name
        stream.codec_context.color_primaries = primaries
        stream.codec_context.color_trc = transfer
        if(matrix is not None):
            stream.codec_context.colorspace = matrix
        stream.codec_context.color_range = colorRange
        for index in range(VIDEO_FRAMES):
            frame.pts = index
            container.mux(stream.encode(frame))
        container.mux(stream.encode())


def writeVideos(scene: np.ndarray) -> None:
    """Writes the video encodings of the scene.

    Untagged YUV stores linear values with the BT.709 matrix, as the app assumes for it.

    Args:
        scene: The scene in linear sRGB.
    """
    white = np.clip(a=scene, a_min=0, a_max=1)
    h264 = {"qp": "0"}
    h265 = {"x265-params": "lossless=1:log-level=error"}
    writeVideo(
        name="mp4_none_none_full.mp4",
        frame=toYUV(rgb=white, pixFmt="yuvj420p", matrix=Colorspace.ITU709, colorRange=ColorRange.JPEG),
        codec="libx264",
        options=h264,
    )
    writeVideo(
        name="mp4_none_none_none.mp4",
        frame=toYUV(rgb=white, pixFmt="yuv420p", matrix=Colorspace.ITU709, colorRange=ColorRange.MPEG),
        codec="libx264",
        options=h264,
    )
    writeVideo(
        name="mp4_bt709_bt709_limited.mp4",
        frame=toYUV(
            rgb=colour.oetf(white, function="ITU-R BT.709"),
            pixFmt="yuv420p",
            matrix=Colorspace.ITU709,
            colorRange=ColorRange.MPEG,
        ),
        codec="libx264",
        options=h264,
        primaries=ColorPrimaries.BT709,
        transfer=ColorTrc.BT709,
        matrix=Colorspace.ITU709,
        colorRange=ColorRange.MPEG,
    )
    writeVideo(
        name="mp4_bt709_none_limited.mp4",
        frame=toYUV(rgb=white, pixFmt="yuv420p", matrix=Colorspace.ITU709, colorRange=ColorRange.MPEG),
        codec="libx264",
        options=h264,
        primaries=ColorPrimaries.BT709,
        matrix=Colorspace.ITU709,
        colorRange=ColorRange.MPEG,
    )
    writeVideo(
        name="mp4_bt709_srgb_full.mp4",
        frame=toYUV(rgb=encodeSRGB(white), pixFmt="yuv420p", matrix=Colorspace.ITU709, colorRange=ColorRange.JPEG),
        codec="libx264",
        options=h264,
        primaries=ColorPrimaries.BT709,
        transfer=ColorTrc.IEC61966_2_1,
        matrix=Colorspace.ITU709,
        colorRange=ColorRange.JPEG,
    )
    # Like a Nikon Z5II's HLG footage: 10-bit HEVC, BT.2020, full range
    hlg = colour.oetf(
        convertPrimaries(img=scene, colourspace=REC2020)*HLG_REFERENCE_WHITE,
        function="ITU-R BT.2100 HLG",
    )
    writeVideo(
        name="mp4_10_bt2020_hlg_full.mp4",
        frame=toYUV(rgb=hlg, pixFmt="yuv420p10le", matrix=Colorspace.BT2020, colorRange=ColorRange.JPEG),
        codec="libx265",
        options=h265,
        primaries=ColorPrimaries.BT2020,
        transfer=ColorTrc.ARIB_STD_B67,
        matrix=Colorspace.BT2020,
        colorRange=ColorRange.JPEG,
    )
    # Like HDR10: 10-bit HEVC, BT.2020, limited range
    pq = colour.oetf(convertPrimaries(img=scene, colourspace=REC2020)*PQ_REFERENCE_WHITE, function="ITU-R BT.2100 PQ")
    writeVideo(
        name="mp4_10_bt2020_pq_limited.mp4",
        frame=toYUV(rgb=pq, pixFmt="yuv420p10le", matrix=Colorspace.BT2020, colorRange=ColorRange.MPEG),
        codec="libx265",
        options=h265,
        primaries=ColorPrimaries.BT2020,
        transfer=ColorTrc.SMPTE2084,
        matrix=Colorspace.BT2020,
        colorRange=ColorRange.MPEG,
    )
    # FFV1 has no 8-bit planar RGB, so the pixels are packed with a padding byte
    writeVideo(
        name="avi8_rgb.avi",
        frame=av.VideoFrame.from_ndarray(toInteger(white, np.uint8), format="rgb24").reformat(format="bgr0"),
        codec="ffv1",
        options={},
    )
    writeVideo(
        name="avi8_gray.avi",
        frame=av.VideoFrame.from_ndarray(toInteger(toGray(scene), np.uint8), format="gray"),
        codec="ffv1",
        options={},
    )


def toGray(img: np.ndarray) -> np.ndarray:
    """Convert to grayscale (using luminance)."""
    return colour.RGB_luminance(RGB=img, primaries=SRGB.primaries, whitepoint=SRGB.whitepoint)


def writeReferences(scene: np.ndarray) -> None:
    """Writes what the app saves as 8-bit sRGB PNG for the color inputs.

    By default every input saves as color_master_sRGB.png, where values above white are clipped. With the Value
    slider at 50%, the inputs that hold values above white save as color_master_sRGB_value50.png. The gray input
    saves as color_master_gray_sRGB.png.

    Args:
        scene: The scene in linear sRGB.
    """
    srgb = [(b"sRGB", bytes([0]))]
    for name, linear in (("color_master_sRGB.png", scene),
                         ("color_master_sRGB_value50.png", 0.5*scene),
                         ("color_master_gray_sRGB.png", np.repeat(a=toGray(scene)[..., None], repeats=3, axis=2))):
        writePNG(
            name=name,
            img=toInteger(encodeSRGB(np.clip(a=linear, a_min=0, a_max=1)), np.uint8),
            chunks=srgb,
            folder=REFERENCE_DIR,
        )


def main() -> None:
    """Writes every color input and reference."""
    scene = makeScene()
    writeImages(scene)
    writeVideos(scene)
    writeReferences(scene)


if __name__ == "__main__":
    main()
