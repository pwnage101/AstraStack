"""Writes and reads the generated test files shared by the pytest tests and the behave steps.

Nothing here imports ``UI`` or GTK.
"""
import os
import struct
import zlib
from fractions import Fraction

import cv2
import numpy as np
import OpenEXR
import tifffile
from PIL import Image

import ColorSpace
import Export
import Lens

# Nikon crop regions (mode, sensor width, sensor height, crop width, crop height, crop x, crop y)
NIKON_CROP_CENTERED = (6, 6064, 4040, 6064, 3416, 0, 312)
NIKON_CROP_OFF_CENTER = (6, 6064, 4040, 6064, 3416, 0, 0)


def colorTestImage(w: int = 32, h: int = 24) -> np.ndarray:
    """Returns an 8-bit RGB test image with saturated colors (not grayscale, so it keeps its color description).

    Args:
        w: Width of the image.
        h: Height of the image.

    Returns:
        The uint8 RGB image.
    """
    x = np.linspace(0, 1, w)[None,:]
    y = np.linspace(0, 1, h)[:,None]
    img = np.dstack([np.repeat(x, h, 0), np.repeat(y, w, 1), np.repeat(1 - x, h, 0)*np.repeat(y, w, 1)])
    return np.around(img*255).astype(np.uint8)


def displayP3Profile() -> bytes:
    """Returns an ICC profile with Display P3 primaries and a gamma 2.2 tone curve.

    Returns:
        The ICC profile bytes, described as "Display P3 test".
    """
    import colour
    p3 = colour.RGB_COLOURSPACES["Display P3"]
    return ColorSpace.iccProfile(matrix=colour.normalised_primary_matrix(p3.primaries, p3.whitepoint),
                                 white=p3.whitepoint,
                                 description="Display P3 test",
                                 gamma=2.2)


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
    header = (struct.pack(">I", offset + len(data)) + bytes(4) + struct.pack(">I", 0x02100000) +
              deviceClass + colorSpace + b"XYZ " + bytes(12) + b"acsp" + bytes(24) + struct.pack(">I", 0) +
              struct.pack(">3i", *[int(round(v*65536)) for v in ColorSpace.D50_XYZ]) + bytes(48))
    return header + table + data


def descTag(text: str) -> bytes:
    """Returns an ICC v2 desc tag.

    Args:
        text: The ASCII description.

    Returns:
        The tag data.
    """
    ascii = text.encode("ascii") + b"\0"
    return b"desc" + bytes(4) + struct.pack(">I", len(ascii)) + ascii + bytes(4 + 4 + 2 + 1 + 67)


def wtptTag() -> bytes:
    """Returns an ICC wtpt tag holding the D50 white point.

    Returns:
        The tag data.
    """
    return b"XYZ " + bytes(4) + struct.pack(">3i", *[int(round(v*65536)) for v in ColorSpace.D50_XYZ])


def lutSRGBProfile() -> bytes:
    """Returns a LUT based (mft2) input profile that describes sRGB, so results can be checked against the input values.

    Returns:
        The ICC profile bytes.
    """
    import colour
    chad = ColorSpace.bradford(colour.xy_to_XYZ(ColorSpace.D65), ColorSpace.D50_XYZ)
    colorants = chad @ ColorSpace.SRGB.matrixRGBToXYZ()
    curve = np.around(ColorSpace.SRGB.linearize(np.linspace(0, 1, 256))*65535).astype(">u2").tobytes()
    clut = b""
    for r in (0, 1):
        for gr in (0, 1):
            for b in (0, 1):
                clut += np.around(np.clip(colorants @ np.array([r, gr, b]), 0, 1.99)*32768).astype(">u2").tobytes()
    identity = struct.pack(">9i", *(65536*np.eye(3).astype(int).flatten()))
    mft2 = (b"mft2" + bytes(4) + struct.pack(">BBBB", 3, 3, 2, 0) + identity + struct.pack(">HH", 256, 2) +
            curve*3 + clut + struct.pack(">HH", 0, 65535)*3)
    return iccFile(deviceClass=b"scnr",
                   colorSpace=b"RGB ",
                   tags=[(b"desc", descTag("LUT sRGB")), (b"wtpt", wtptTag()), (b"A2B0", mft2)])


def grayProfile(gamma: float) -> bytes:
    """Returns a gray TRC display profile with a gamma tone curve.

    Args:
        gamma: Gamma of the kTRC curve.

    Returns:
        The ICC profile bytes.
    """
    curve = b"curv" + bytes(4) + struct.pack(">IH", 1, int(round(gamma*256)))
    return iccFile(deviceClass=b"mntr",
                   colorSpace=b"GRAY",
                   tags=[(b"desc", descTag("Gray test")), (b"wtpt", wtptTag()), (b"kTRC", curve)])


def writePQPNG(path: str) -> None:
    """Writes a 16-bit PNG with a cICP chunk: Rec.2020 primaries, PQ transfer, full range.

    Pixel (0,0) is PQ 0.58 (about 203 cd/m2, the reference white).

    Args:
        path: Path of the file.
    """
    img = colorTestImage().astype(np.uint16)*257
    img[0,0] = np.around(0.58*65535)
    success, png = cv2.imencode(".png", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    with open(path, "wb") as f:
        f.write(Export.insertPNGChunks(png=png.tobytes(), chunks=[(b"cICP", bytes([9, 16, 0, 1]))]))


def writeDNG(path: str) -> None:
    """Writes a small DNG: an RGGB mosaic with a neutral (as-shot) gray area and a colored block.

    Args:
        path: Path of the file.
    """
    h, w = 32, 48
    neutral = np.array([0.5, 1.0, 0.7]) # Camera response to the as-shot illuminant
    rgb = np.ones((h, w, 3))*0.25*neutral
    rgb[4:12, 4:12] = [0.4, 0.1, 0.05]
    cfa = np.zeros((h, w), np.uint16)
    cfa[0::2,0::2] = rgb[0::2,0::2,0]*4000
    cfa[0::2,1::2] = rgb[0::2,1::2,1]*4000
    cfa[1::2,0::2] = rgb[1::2,0::2,1]*4000
    cfa[1::2,1::2] = rgb[1::2,1::2,2]*4000
    def srational(values: list[float]) -> list[int]:
        result = []
        for v in values:
            f = Fraction(v).limit_denominator(10000)
            result += [f.numerator, f.denominator]
        return result
    colorMatrix = [0.7, -0.1, -0.05, -0.3, 1.2, 0.1, -0.05, 0.2, 0.6] # XYZ -> camera
    tags = [(50706, 'B', 4, (1, 4, 0, 0), True), # DNGVersion
            (50707, 'B', 4, (1, 1, 0, 0), True), # DNGBackwardVersion
            (50708, 's', 0, 'AstraStack Test', True), # UniqueCameraModel
            (33421, 'H', 2, (2, 2), True), # CFARepeatPatternDim
            (33422, 'B', 4, (0, 1, 1, 2), True), # CFAPattern RGGB
            (50721, '2i', 9, srational(colorMatrix), True), # ColorMatrix1
            (50778, 'H', 1, 21, True), # CalibrationIlluminant1 D65
            (50728, '2I', 3, srational(neutral), True), # AsShotNeutral
            (50717, 'I', 1, 4095, True), # WhiteLevel
            (50714, 'H', 1, 0, True)] # BlackLevel
    tifffile.imwrite(path, data=cfa, photometric=32803, extratags=tags, compression=None)


def writeRec2020Video(path: str) -> None:
    """Writes a 10 frame lossless H.264 video tagged Rec.2020 primaries and matrix with the BT.709 transfer.

    Args:
        path: Path of the file.
    """
    import av
    w, h = 64, 48
    with av.open(path, "w") as container:
        stream = container.add_stream("libx264", rate=10)
        stream.width, stream.height, stream.pix_fmt = w, h, "yuv420p"
        stream.codec_context.color_primaries = 9
        stream.codec_context.color_trc = 1
        stream.codec_context.colorspace = 9
        stream.codec_context.color_range = 1
        stream.options = {"qp": "0"}
        for i in range(10):
            yuv = np.full((h*3//2, w), 128, np.uint8)
            yuv[:h] = 180 # Neutral gray, Y = 180
            yuv[10 + i:20 + i, 10:20] = 80 # Moving colored block
            yuv[h + (10 + i)//4:h + (20 + i)//4, 5:10] = 200 # U
            frame = av.VideoFrame.from_ndarray(yuv, format="yuv420p")
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)


def gridTestImage(w: int, h: int, offset: int = 0) -> np.ndarray:
    """Returns a grid of bright lines on a dark background, so lens corrections visibly move pixels.

    Args:
        w: Width of the image.
        h: Height of the image.
        offset: Shift of the grid in pixels (modulo 8).

    Returns:
        The uint8 BGR image.
    """
    img = np.full((h, w, 3), 40, np.uint8)
    img[:, (offset % 8)::8] = (200, 220, 240)
    img[(offset % 8)::8, :] = (240, 200, 220)
    return img


def nctgAtom(cropRegion: tuple[int, ...]) -> bytes:
    """Returns a Nikon NCTG atom (big-endian tag list) with the camera, lens and crop region.

    Args:
        cropRegion: The crop region (mode, sensor width, sensor height, crop width, crop height, crop x, crop y).

    Returns:
        The atom bytes.
    """
    def tag(id: int, type: int, count: int, data: bytes) -> bytes:
        return struct.pack(">IHH", id, type, count) + data
    tags = b""
    for id, text in ((Lens.NCTG_MAKE, "NIKON CORPORATION"), (Lens.NCTG_MODEL, "NIKON Z 5"),
                     (Lens.NCTG_LENS_MAKE, "NIKON"), (Lens.NCTG_LENS_MODEL, "NIKKOR Z 24-70mm f/2.8 S")):
        data = text.encode() + b"\0"
        tags += tag(id=id, type=2, count=len(data), data=data)
    tags += tag(id=Lens.NCTG_FOCAL_LENGTH, type=5, count=1, data=struct.pack(">II", 240, 10))
    tags += tag(id=Lens.NCTG_FNUMBER, type=5, count=1, data=struct.pack(">II", 280, 100))
    tags += tag(id=Lens.NCTG_CROP_HISPEED, type=3, count=7, data=struct.pack(">7H", *cropRegion))
    return struct.pack(">I", 8 + len(tags)) + b"NCTG" + tags


def atom(type: bytes, data: bytes) -> bytes:
    """Returns a QuickTime atom.

    Args:
        type: The four character atom type.
        data: The atom's contents.

    Returns:
        The atom bytes.
    """
    return struct.pack(">I", 8 + len(data)) + type + data


def writeNikonVideo(path: str, cropRegion: tuple[int, ...]) -> None:
    """Writes a 10 frame MOV with a Nikon NCTG atom (moov/udta/NCDT/NCTG).

    The moov atom is last, so no offsets change when the udta atom is appended to it.

    Args:
        path: Path of the file.
        cropRegion: The crop region written in the NCTG atom.
    """
    import av
    w, h = 96, 64
    with av.open(path, "w", format="mov") as container:
        stream = container.add_stream("libx264", rate=10)
        stream.width, stream.height, stream.pix_fmt = w, h, "yuv420p"
        stream.options = {"qp": "0"}
        for i in range(10):
            image = cv2.cvtColor(gridTestImage(w=w, h=h, offset=i//4), cv2.COLOR_BGR2RGB)
            frame = av.VideoFrame.from_ndarray(image, format="rgb24")
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    with open(path, "rb") as f:
        data = f.read()
    position = 0
    while(position < len(data)):
        size, type = struct.unpack(">I4s", data[position:position+8])
        if(type == b"moov"):
            break
        position += size
    assert type == b"moov" and position + size == len(data), "moov is not the last atom"
    udta = atom(type=b"udta", data=atom(type=b"NCDT", data=nctgAtom(cropRegion)))
    moov = struct.pack(">I", size + len(udta)) + b"moov" + data[position+8:position+size] + udta
    with open(path, "wb") as f:
        f.write(data[:position] + moov)


def writeLensJPEG(path: str) -> None:
    """Writes a JPEG with the camera and lens in its EXIF.

    Args:
        path: Path of the file.
    """
    exif = Image.Exif()
    exif[0x010f] = "Canon"
    exif[0x0110] = "Canon EOS 5D Mark II"
    exifIFD = exif.get_ifd(0x8769)
    exifIFD[0xa434] = "Canon EF 24-105mm f/4L IS USM"
    exifIFD[0x920a] = 24.0
    exifIFD[0x829d] = 4.0
    Image.fromarray(cv2.cvtColor(gridTestImage(w=96, h=64), cv2.COLOR_BGR2RGB)).save(path, exif=exif, quality=95)


def writeFITS(path: str) -> None:
    """Writes a 16-bit grayscale FITS image with a gradient.

    Args:
        path: Path of the file.
    """
    from astropy.io import fits
    data = (np.add.outer(np.arange(24), np.arange(32))*1000).astype(np.uint16)
    fits.writeto(path, data, overwrite=True)


def writeTestFile(kind: str, path: str) -> None:
    """Writes a generated test file, creating its folder.

    Args:
        kind: Kind of file, e.g. "Display P3 PNG" or "Nikon video".
        path: Path of the file.

    Raises:
        ValueError: The kind is unknown.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if(kind == "Display P3 PNG"):
        Image.fromarray(colorTestImage()).save(path, icc_profile=displayP3Profile())
    elif(kind == "untagged PNG"):
        Image.fromarray(colorTestImage()).save(path)
    elif(kind == "LUT profile PNG"):
        Image.fromarray(colorTestImage()).save(path, icc_profile=lutSRGBProfile())
    elif(kind == "cICP PQ PNG"):
        writePQPNG(path)
    elif(kind == "DNG"):
        writeDNG(path)
    elif(kind == "Rec.2020 video"):
        writeRec2020Video(path)
    elif(kind == "Nikon video"):
        writeNikonVideo(path=path, cropRegion=NIKON_CROP_CENTERED)
    elif(kind == "off-center Nikon video"):
        writeNikonVideo(path=path, cropRegion=NIKON_CROP_OFF_CENTER)
    elif(kind == "lens EXIF JPEG"):
        writeLensJPEG(path)
    elif(kind == "FITS"):
        writeFITS(path)
    else:
        raise ValueError(f"Unknown test file kind '{kind}'")


def sameProfile(icc1: bytes | None, icc2: bytes | None) -> bool:
    """Returns whether the ICC profiles are the same, ignoring the creation date and profile ID in the header.

    Args:
        icc1: First profile, or None.
        icc2: Second profile, or None.

    Returns:
        True if both are the same profile (or both None).
    """
    def strip(icc: bytes | None) -> bytes | None:
        return icc[:24] + icc[36:84] + icc[100:] if icc is not None else None
    return strip(icc1) == strip(icc2)


def pngChunksBeforeIDAT(path: str) -> list[tuple[bytes, bytes]]:
    """Returns the PNG chunks before the first IDAT, checking every CRC.

    Args:
        path: Path of the PNG file.

    Returns:
        The chunks as (type, data).

    Raises:
        AssertionError: The file isn't a PNG or a CRC is wrong.
    """
    with open(path, "rb") as f:
        png = f.read()
    assert png[:8] == b"\x89PNG\r\n\x1a\n", f"'{path}' is not a PNG"
    chunks = []
    i = 8
    while(i < len(png)):
        length = struct.unpack(">I", png[i:i+4])[0]
        chunkType = png[i+4:i+8]
        data = png[i+8:i+8+length]
        crc = struct.unpack(">I", png[i+8+length:i+12+length])[0]
        assert crc == zlib.crc32(chunkType + data), f"Bad CRC in '{chunkType}' chunk of '{path}'"
        if(chunkType == b"IDAT"):
            break
        chunks.append((chunkType, data))
        i += 12 + length
    return chunks


def exportTags(path: str) -> dict[str, object]:
    """Returns the color tags and processing record embedded in an exported file.

    Args:
        path: Path of the file.

    Returns:
        A dict with "icc" (ICC profile bytes or None), "pngColor" (PNG color chunk types), "chromaticities" (EXR, or
        None), "iccAttribute" (bytes of the EXR iccProfile attribute, or None) and "record" (processing record by name).

    Raises:
        AssertionError: The file has no processing record, or its EXR iccProfile attribute has the wrong type hint.
    """
    file = os.path.basename(path)
    tags = {"icc": None, "pngColor": [], "chromaticities": None, "iccAttribute": None, "record": {}}
    def parseText(text: str) -> dict[str, str]:
        lines = text.split("\n")
        assert lines[0] == Export.RECORD_TITLE, f"'{file}' has no processing record"
        return dict(line.split(": ", 1) for line in lines[1:])
    with open(path, "rb") as f:
        signature = f.read(8)
    if(signature == b"\x89PNG\r\n\x1a\n"):
        for chunkType, data in pngChunksBeforeIDAT(path):
            if(chunkType in (b"sRGB", b"iCCP", b"cHRM", b"gAMA", b"cICP")):
                tags["pngColor"].append(chunkType)
            elif(chunkType == b"tEXt"):
                name, value = data.split(b"\0", 1)
                tags["record"][name.decode("latin-1")] = value.decode("latin-1")
    elif(signature[:2] == b"\xff\xd8"):
        with Image.open(path) as im:
            tags["icc"] = im.info.get("icc_profile")
            tags["record"] = parseText(im.info["comment"].decode("latin-1"))
    elif(signature[:4] in (b"II*\0", b"MM\0*")):
        with tifffile.TiffFile(path) as tif:
            page = tif.pages[0]
            icc = page.tags.get("InterColorProfile")
            tags["icc"] = None if icc is None else bytes(icc.value)
            tags["record"] = parseText(page.tags["ImageDescription"].value)
    else:
        with OpenEXR.File(path) as exr:
            header = exr.header()
            tags["chromaticities"] = header.get("chromaticities")
            if("iccProfile" in header):
                assert header["iccProfile"].type_hint == Export.ICC_TYPE_HINT, f"'{file}' iccProfile type hint"
                tags["iccAttribute"] = bytes(header["iccProfile"].data)
            tags["record"] = {name: header[name] for name in header if isinstance(header[name], str)}
    return tags


def readExport(path: str) -> np.ndarray:
    """Reads the pixels of an exported file.

    Args:
        path: Path of the PNG, JPEG, TIFF or EXR file.

    Returns:
        The RGB image in the file's dtype.
    """
    if(path.endswith(".exr")):
        with OpenEXR.File(path, separate_channels=True) as exr:
            channels = exr.channels()
            return np.dstack([channels[c].pixels for c in ("R", "G", "B")])
    elif(path.endswith(".tif")):
        return tifffile.imread(path)
    elif(path.endswith(".jpg")):
        with Image.open(path) as im:
            return np.asarray(im)
    return cv2.cvtColor(cv2.imread(path, cv2.IMREAD_UNCHANGED), cv2.COLOR_BGR2RGB)
