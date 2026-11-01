"""Generates the color-tagged, FITS and RAW test inputs from Jupiter.mp4 and a Google Pixel 8a DNG.

Run from the repository root: ``python features/testFiles/makeInputs.py [source.dng]``. The source DNG can also be
given with the ASTRASTACK_SOURCE_DNG environment variable; the RAW fixtures are skipped when it isn't present.
"""
import os
import struct
import subprocess
import sys
import zlib

import av
import colour
import cv2
import numpy as np
import png
import rawpy
import tifffile
from astropy.io import fits
from PIL import ExifTags, Image

INPUTS = os.path.join("features", "testFiles", "inputs")
VIDEO = os.path.join(INPUTS, "Jupiter.mp4")
COLOR_DIR = os.path.join(INPUTS, "color")
FITS_DIR = os.path.join(INPUTS, "fits")
RAW_DIR = os.path.join(INPUTS, "raw")
DEFAULT_DNG = "/home/sankey/photos/2026-07-10_san_diego_pq_canyon/raw/IMG_20260710_182014_0.dng"

# Region of the 640x480 video frame around Jupiter (x, y, width, height); even so YUV 4:2:0 crops stay aligned
CROP = (98, 40, 420, 336)
RETAG_FRAMES = 6
SEQUENCE_FRAMES = 5

D50_XYZ = np.array([0.9642, 1.0, 0.8249])
GAMMA_22 = 563/256  # Gamma 2.2 as stored in a one-entry ICC curv tag
REC709_LUMINANCE = np.array([0.2126, 0.7152, 0.0722])

# Display P3 linear values of the saturated patches in the P3 files: (rows, columns, RGB)
P3_PATCHES = [(slice(4, 20), slice(4, 20), (0.0, 0.6, 0.0)),
              (slice(4, 20), slice(24, 40), (0.6, 0.0, 0.0))]

FITS_BLACK = 2000
FITS_WHITE = 60000
FITS_MONO_BLACK = 1000
FITS_SIZE = (160, 128)

# Pixel 8a DNG crops: origin (x, y) of each frame in the sensor data, and the crop size (width, height)
RAW_CROPS = [(1792, 1280), (1798, 1284)]
RAW_SIZE = (512, 384)
RAW_TAGS = (271,  # Make
            272,  # Model
            33421,  # CFARepeatPatternDim
            33422,  # CFAPattern
            50706,  # DNGVersion
            50707,  # DNGBackwardVersion
            50708,  # UniqueCameraModel
            50710,  # CFAPlaneColor
            50711,  # CFALayout
            50713,  # BlackLevelRepeatDim
            50714,  # BlackLevel
            50717,  # WhiteLevel
            50721,  # ColorMatrix1
            50722,  # ColorMatrix2
            50723,  # CameraCalibration1
            50724,  # CameraCalibration2
            50728,  # AsShotNeutral
            50730,  # BaselineExposure
            50778,  # CalibrationIlluminant1
            50779,  # CalibrationIlluminant2
            50964,  # ForwardMatrix1
            50965)  # ForwardMatrix2


def decodeFrames(count: int) -> list[np.ndarray]:
    """Decodes the first frames of Jupiter.mp4 to 16-bit RGB with the BT.709 matrix and full range.

    Args:
        count: Number of frames.

    Returns:
        The uint16 RGB frames, 640x480.
    """
    frames = []
    with av.open(VIDEO) as container:
        for frame in container.decode(video=0):
            rgb = frame.reformat(format="rgb48le",
                                 src_colorspace="ITU709",
                                 src_color_range="JPEG",
                                 dst_color_range="JPEG")
            frames.append(rgb.to_ndarray())
            if(len(frames) == count):
                break
    return frames


def cropAndHalve(img: np.ndarray) -> np.ndarray:
    """Crops a video frame around Jupiter and halves its size.

    Args:
        img: The 640x480 frame.

    Returns:
        The 210x168 float64 image, in the input's units.
    """
    x, y, w, h = CROP
    return cv2.resize(img[y:y + h, x:x + w].astype(np.float64), (w//2, h//2), interpolation=cv2.INTER_AREA)


def linearSource() -> np.ndarray:
    """Returns the reference data: the cropped first video frame, its 16-bit values taken as linear.

    Returns:
        The 210x168 RGB image, float64 on the 0-1 scale, rounded to 16-bit steps.
    """
    return np.around(cropAndHalve(decodeFrames(1)[0]))/65535


def to16(img: np.ndarray) -> np.ndarray:
    """Quantizes 0-1 data to 16 bits.

    Args:
        img: Image on the 0-1 scale.

    Returns:
        The uint16 image.
    """
    return np.around(np.clip(img, 0, 1)*65535).astype(np.uint16)


def to8(img: np.ndarray) -> np.ndarray:
    """Quantizes 0-1 data to 8 bits.

    Args:
        img: Image on the 0-1 scale.

    Returns:
        The uint8 image.
    """
    return np.around(np.clip(img, 0, 1)*255).astype(np.uint8)


def encodePNG(img: np.ndarray) -> bytes:
    """Encodes an untagged PNG with OpenCV.

    Args:
        img: RGB or gray image, uint8 or uint16.

    Returns:
        The PNG file bytes.
    """
    if(img.ndim == 3):
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    success, data = cv2.imencode(".png", img)
    return data.tobytes()


def writePNG(path: str, img: np.ndarray, chunks: list[tuple[bytes, bytes]]) -> None:
    """Writes a PNG with extra chunks right after IHDR.

    Args:
        path: Path of the file.
        img: RGB or gray image, uint8 or uint16.
        chunks: The chunks as (type, data).
    """
    existing = list(png.Reader(bytes=encodePNG(img)).chunks())
    with open(path, "wb") as f:
        png.write_chunks(f, existing[:1] + chunks + existing[1:])


def iccpChunk(name: str, icc: bytes) -> tuple[bytes, bytes]:
    """Returns a PNG iCCP chunk.

    Args:
        name: Profile name.
        icc: ICC profile bytes.

    Returns:
        The chunk as (type, data).
    """
    return (b"iCCP", name.encode("latin-1") + b"\0\0" + zlib.compress(icc))


def bradford(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Returns the Bradford chromatic adaptation matrix between two whites.

    Args:
        src: XYZ of the source white.
        dst: XYZ of the destination white.

    Returns:
        The 3x3 matrix.
    """
    return colour.adaptation.matrix_chromatic_adaptation_VonKries(XYZ_w=src, XYZ_wr=dst, transform="Bradford")


def s15Fixed16(values: np.ndarray) -> bytes:
    """Encodes numbers as ICC s15Fixed16Number.

    Args:
        values: The numbers.

    Returns:
        The big-endian encoded numbers.
    """
    return b"".join(struct.pack(">i", int(round(v*65536))) for v in np.ravel(values))


def xyzTag(XYZ: np.ndarray) -> bytes:
    """Returns an ICC XYZ tag.

    Args:
        XYZ: The XYZ values.

    Returns:
        The tag data.
    """
    return b"XYZ " + bytes(4) + s15Fixed16(XYZ)


def descTag(text: str) -> bytes:
    """Returns an ICC v2 desc tag.

    Args:
        text: The ASCII description.

    Returns:
        The tag data.
    """
    ascii = text.encode("ascii") + b"\0"
    return b"desc" + bytes(4) + struct.pack(">I", len(ascii)) + ascii + bytes(4 + 4 + 2 + 1 + 67)


def textTag(text: str) -> bytes:
    """Returns an ICC text tag.

    Args:
        text: The ASCII text.

    Returns:
        The tag data.
    """
    return b"text" + bytes(4) + text.encode("ascii") + b"\0"


def gammaCurve(gamma: float) -> bytes:
    """Returns an ICC curv tag holding a single gamma value.

    Args:
        gamma: The exponent, stored as u8Fixed8Number.

    Returns:
        The tag data.
    """
    return b"curv" + bytes(4) + struct.pack(">IH", 1, int(round(gamma*256)))


def iccFile(deviceClass: bytes, colorSpace: bytes, tags: list[tuple[bytes, bytes]]) -> bytes:
    """Assembles an ICC v2 profile from its tags.

    Args:
        deviceClass: Profile class signature, e.g. b"mntr".
        colorSpace: Data color space signature, e.g. b"RGB " or b"GRAY".
        tags: The tags as (signature, data).

    Returns:
        The ICC profile bytes.
    """
    offset = 128 + 4 + 12*len(tags)
    table = struct.pack(">I", len(tags))
    data = b""
    for signature, tag in tags:
        table += signature + struct.pack(">II", offset + len(data), len(tag))
        data += tag + bytes(-len(tag) % 4)
    header = (struct.pack(">I", offset + len(data)) + bytes(4) + struct.pack(">I", 0x02100000) + deviceClass +
              colorSpace + b"XYZ " + struct.pack(">6H", 2026, 1, 1, 0, 0, 0) + b"acsp" + bytes(24) +
              struct.pack(">I", 0) + s15Fixed16(D50_XYZ) + bytes(48))
    return header + table + data


def matrixProfile(colourspace: str, description: str, gamma: float) -> bytes:
    """Returns a matrix/TRC display profile for an RGB color space with a gamma tone curve.

    Args:
        colourspace: Name of the RGB color space in colour.RGB_COLOURSPACES, e.g. "Display P3".
        description: Profile description.
        gamma: Exponent of the tone curves.

    Returns:
        The ICC profile bytes, with colorants adapted to D50 and a chad tag.
    """
    space = colour.RGB_COLOURSPACES[colourspace]
    chad = bradford(src=colour.xy_to_XYZ(space.whitepoint), dst=D50_XYZ)
    colorants = chad @ space.matrix_RGB_to_XYZ
    curve = gammaCurve(gamma)
    return iccFile(deviceClass=b"mntr",
                   colorSpace=b"RGB ",
                   tags=[(b"desc", descTag(description)),
                         (b"cprt", textTag("No copyright, use freely")),
                         (b"wtpt", xyzTag(D50_XYZ)),
                         (b"chad", b"sf32" + bytes(4) + s15Fixed16(chad)),
                         (b"rXYZ", xyzTag(colorants[:, 0])),
                         (b"gXYZ", xyzTag(colorants[:, 1])),
                         (b"bXYZ", xyzTag(colorants[:, 2])),
                         (b"rTRC", curve),
                         (b"gTRC", curve),
                         (b"bTRC", curve)])


def grayProfile(gamma: float) -> bytes:
    """Returns a gray TRC display profile with a gamma tone curve.

    Args:
        gamma: Exponent of the kTRC curve.

    Returns:
        The ICC profile bytes.
    """
    return iccFile(deviceClass=b"mntr",
                   colorSpace=b"GRAY",
                   tags=[(b"desc", descTag("Gray gamma 2.2")),
                         (b"cprt", textTag("No copyright, use freely")),
                         (b"wtpt", xyzTag(D50_XYZ)),
                         (b"kTRC", gammaCurve(gamma))])


def lutSRGBProfile() -> bytes:
    """Returns a LUT based (mft2) input profile that describes sRGB, so results can be checked against the input values.

    Returns:
        The ICC profile bytes.
    """
    srgb = colour.RGB_COLOURSPACES["sRGB"]
    chad = bradford(src=colour.xy_to_XYZ(srgb.whitepoint), dst=D50_XYZ)
    colorants = chad @ srgb.matrix_RGB_to_XYZ
    curve = np.around(colour.models.eotf_sRGB(np.linspace(0, 1, 256))*65535).astype(">u2").tobytes()
    clut = b""
    for r in (0, 1):
        for g in (0, 1):
            for b in (0, 1):
                XYZ = colorants @ np.array([r, g, b])
                clut += np.around(np.clip(XYZ, 0, 1.99)*32768).astype(">u2").tobytes()
    identity = s15Fixed16(np.eye(3))
    mft2 = (b"mft2" + bytes(4) + struct.pack(">BBBB", 3, 3, 2, 0) + identity + struct.pack(">HH", 256, 2) +
            curve*3 + clut + struct.pack(">HH", 0, 65535)*3)
    return iccFile(deviceClass=b"scnr",
                   colorSpace=b"RGB ",
                   tags=[(b"desc", descTag("LUT sRGB")),
                         (b"cprt", textTag("No copyright, use freely")),
                         (b"wtpt", xyzTag(D50_XYZ)),
                         (b"A2B0", mft2)])


def displayP3Data(linear: np.ndarray) -> np.ndarray:
    """Converts linear sRGB data to linear Display P3 and paints the saturated P3_PATCHES.

    Args:
        linear: Linear sRGB image on the 0-1 scale.

    Returns:
        The linear Display P3 image.
    """
    p3 = colour.RGB_to_RGB(RGB=linear, input_colourspace="sRGB", output_colourspace="Display P3")
    for rows, columns, value in P3_PATCHES:
        p3[rows, columns] = value
    return p3


def writeImages(linear: np.ndarray) -> None:
    """Writes the color-tagged images, all derived from the linear source.

    Args:
        linear: Linear sRGB image on the 0-1 scale.
    """
    encoded = colour.models.eotf_inverse_sRGB(linear)
    chrmSRGB = struct.pack(">8I", 31270, 32900, 64000, 33000, 30000, 60000, 15000, 6000)
    pngs = [("linear.png", to16(linear), []),
            ("srgb_chunk.png", to16(encoded), [(b"sRGB", bytes([0]))]),
            ("gama45455.png", to16(encoded), [(b"gAMA", struct.pack(">I", 45455))]),
            ("gama100000.png", to16(linear), [(b"gAMA", struct.pack(">I", 100000))]),
            ("gama100000_chrm_srgb.png", to16(linear), [(b"gAMA", struct.pack(">I", 100000)), (b"cHRM", chrmSRGB)]),
            ("cicp.png", to16(encoded), [(b"cICP", bytes([1, 13, 0, 1]))])]
    for name, img, chunks in pngs:
        writePNG(path=os.path.join(COLOR_DIR, name), img=img, chunks=chunks)

    p3Profile = matrixProfile(colourspace="Display P3", description="Display P3 gamma 2.2", gamma=GAMMA_22)
    p3 = np.clip(displayP3Data(linear), 0, 1)**(1/GAMMA_22)
    writePNG(path=os.path.join(COLOR_DIR, "p3.png"), img=to16(p3), chunks=[iccpChunk(name="Display P3", icc=p3Profile)])
    Image.fromarray(to8(p3)).save(fp=os.path.join(COLOR_DIR, "p3.jpg"),
                                  quality=95,
                                  subsampling=0,
                                  icc_profile=p3Profile)

    exif = Image.Exif()
    exif.get_ifd(ExifTags.IFD.Exif)[ExifTags.Base.ColorSpace] = 1
    Image.fromarray(to8(encoded)).save(fp=os.path.join(COLOR_DIR, "exif_srgb.jpg"),
                                       quality=95,
                                       subsampling=0,
                                       exif=exif)

    gray = (linear @ REC709_LUMINANCE)**(1/GAMMA_22)
    writePNG(path=os.path.join(COLOR_DIR, "gray_icc.png"),
             img=to16(gray),
             chunks=[iccpChunk(name="Gray gamma 2.2", icc=grayProfile(GAMMA_22))])
    writePNG(path=os.path.join(COLOR_DIR, "lut_srgb.png"),
             img=to8(encoded),
             chunks=[iccpChunk(name="LUT sRGB", icc=lutSRGBProfile())])


def ffmpeg(*args: str) -> None:
    """Runs ffmpeg quietly, overwriting outputs.

    Args:
        *args: The ffmpeg arguments after the global options.
    """
    subprocess.run(["ffmpeg", "-y", "-v", "error", *args], check=True)


def retagVideo(name: str, primaries: int, transfer: int, matrix: int) -> None:
    """Copies the first frames of Jupiter.mp4 without re-encoding, with new color tags in the H.264 headers.

    The range flag stays full, as in the original.

    Args:
        name: File name of the copy.
        primaries: CICP colour primaries.
        transfer: CICP transfer characteristics.
        matrix: CICP matrix coefficients.
    """
    bsf = (f"h264_metadata=colour_primaries={primaries}:transfer_characteristics={transfer}:"
           f"matrix_coefficients={matrix}:video_full_range_flag=1")
    ffmpeg("-i", VIDEO,
           "-map", "0:v:0",
           "-frames:v", str(RETAG_FRAMES),
           "-c", "copy",
           "-bsf:v", bsf,
           "-map_metadata", "-1",
           "-fflags", "+bitexact",
           os.path.join(COLOR_DIR, name))


def writeRangeVideos() -> None:
    """Writes the first frames, cropped and halved, losslessly as full range and as limited range H.264.

    The full range video is yuvj420p; the limited range one is yuv420p converted from it, tagged limited.
    """
    x, y, w, h = CROP
    full = os.path.join(COLOR_DIR, "full_range.mp4")
    common = ["-map_metadata", "-1", "-fflags", "+bitexact", "-c:v", "libx264", "-qp", "0", "-preset", "veryslow"]
    ffmpeg("-i", VIDEO,
           "-frames:v", str(SEQUENCE_FRAMES),
           "-vf", f"crop={w}:{h}:{x}:{y},scale={w//2}:{h//2}:flags=area",
           "-pix_fmt", "yuvj420p",
           "-color_range", "pc",
           *common,
           full)
    ffmpeg("-i", full,
           "-vf", "scale=in_range=pc:out_range=tv,format=yuv420p",
           "-color_range", "tv",
           "-bsf:v", "h264_metadata=video_full_range_flag=0",
           *common,
           os.path.join(COLOR_DIR, "limited_range.mp4"))


def writeSequenceVideos() -> None:
    """Writes the cropped first frames as untagged 8-bit RGB and gray PNG sequences and the same frames as FFV1 AVI."""
    frames = [to8(cropAndHalve(frame)/65535) for frame in decodeFrames(SEQUENCE_FRAMES)]
    for kind, pixFmt in (("rgb", "gbrp"), ("gray", "gray")):
        folder = os.path.join(COLOR_DIR, f"{kind}_sequence")
        os.makedirs(folder, exist_ok=True)
        for i, frame in enumerate(frames):
            img = frame if kind == "rgb" else to8((frame/255) @ REC709_LUMINANCE)
            with open(os.path.join(folder, f"frame_{i:03d}.png"), "wb") as f:
                f.write(encodePNG(img))
        ffmpeg("-framerate", "10",
               "-i", os.path.join(folder, "frame_%03d.png"),
               "-c:v", "ffv1",
               "-pix_fmt", pixFmt,
               "-map_metadata", "-1",
               "-fflags", "+bitexact",
               os.path.join(COLOR_DIR, f"{kind}.avi"))


def writeMismatchedSequence() -> None:
    """Writes a two-frame sequence whose frames have different color tags: untagged, then with an sRGB chunk."""
    folder = os.path.join(COLOR_DIR, "mismatched_sequence")
    os.makedirs(folder, exist_ok=True)
    frames = [to8(cropAndHalve(frame)/65535) for frame in decodeFrames(2)]
    with open(os.path.join(folder, "frame_000.png"), "wb") as f:
        f.write(encodePNG(frames[0]))
    writePNG(path=os.path.join(folder, "frame_001.png"), img=frames[1], chunks=[(b"sRGB", bytes([0]))])


def writeVideos() -> None:
    """Writes the re-tagged, range and untagged RGB/gray videos."""
    retagVideo(name="bt709.mp4", primaries=1, transfer=1, matrix=1)
    retagVideo(name="bt601.mp4", primaries=6, transfer=6, matrix=6)
    retagVideo(name="srgb.mp4", primaries=1, transfer=13, matrix=1)
    retagVideo(name="bt709_unspecified_trc.mp4", primaries=1, transfer=2, matrix=1)
    writeRangeVideos()
    writeSequenceVideos()
    writeMismatchedSequence()


def writeFITS(linear: np.ndarray) -> None:
    """Writes a two frame 16-bit Bayer FITS sequence and a 16-bit mono FITS, with black offsets.

    The Bayer header states RGGB with YBAYROFF=1, so the mosaic starts with a GBRG row pair. Frame 2 is 60% as bright
    as frame 1 above black.

    Args:
        linear: Linear sRGB image on the 0-1 scale.
    """
    small = cv2.resize(linear, FITS_SIZE, interpolation=cv2.INTER_AREA)
    mosaic = np.empty(small.shape[:2])
    mosaic[0::2, 0::2] = small[0::2, 0::2, 1]
    mosaic[0::2, 1::2] = small[0::2, 1::2, 2]
    mosaic[1::2, 0::2] = small[1::2, 0::2, 0]
    mosaic[1::2, 1::2] = small[1::2, 1::2, 1]
    for i, scale in enumerate((1.0, 0.6)):
        data = np.around(FITS_BLACK + scale*mosaic*(FITS_WHITE - FITS_BLACK)).astype(np.uint16)
        header = fits.Header()
        header["BAYERPAT"] = "RGGB"
        header["XBAYROFF"] = 0
        header["YBAYROFF"] = 1
        header["SATURATE"] = FITS_WHITE
        fits.writeto(filename=os.path.join(FITS_DIR, f"bayer_{i + 1}.fits"), data=data, header=header, overwrite=True)
    mono = np.around(FITS_MONO_BLACK + (small @ REC709_LUMINANCE)*(65535 - FITS_MONO_BLACK)).astype(np.uint16)
    fits.writeto(filename=os.path.join(FITS_DIR, "mono.fits"), data=mono, overwrite=True)


def dngTags(path: str) -> list[tuple[int, int, int, object, bool]]:
    """Reads the RAW_TAGS of a DNG's main image as tifffile extratags.

    Args:
        path: Path of the source DNG.

    Returns:
        The tags as (code, TIFF data type, count, value, writeonce).
    """
    tags = []
    with tifffile.TiffFile(path) as tif:
        for code in RAW_TAGS:
            tag = tif.pages[0].tags.get(code)
            if(tag is None):
                continue
            value = tuple(tag.value) if isinstance(tag.value, bytes) else tag.value
            count = 0 if tag.dtype == tifffile.DATATYPE.ASCII else tag.count
            tags.append((code, int(tag.dtype), count, value, True))
    return tags


def writeRAW(source: str) -> None:
    """Writes small DNG crops of a Bayer DNG's sensor data with the source's color and level tags.

    Args:
        source: Path of the source DNG.
    """
    with rawpy.imread(source) as raw:
        cfa = raw.raw_image_visible.copy()
    tags = dngTags(source)
    w, h = RAW_SIZE
    for i, (x, y) in enumerate(RAW_CROPS):
        tifffile.imwrite(os.path.join(RAW_DIR, f"pixel_crop_{i + 1}.dng"),
                         data=np.ascontiguousarray(cfa[y:y + h, x:x + w]),
                         photometric=32803,
                         extratags=tags,
                         metadata=None,
                         software=None)


def main() -> None:
    """Writes every test input."""
    for folder in (COLOR_DIR, FITS_DIR, RAW_DIR):
        os.makedirs(folder, exist_ok=True)
    linear = linearSource()
    writeImages(linear)
    writeVideos()
    writeFITS(linear)
    source = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("ASTRASTACK_SOURCE_DNG", DEFAULT_DNG)
    if(os.path.exists(source)):
        writeRAW(source)
    else:
        print(f"Source DNG '{source}' not found, skipping the RAW inputs")


if __name__ == "__main__":
    main()
