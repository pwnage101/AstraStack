from __future__ import annotations
from lazy import lazy
cv2 = lazy("cv2")
np = lazy("numpy")
tifffile = lazy("tifffile")
OpenEXR = lazy("OpenEXR")
Image = lazy("PIL.Image")
ImageCms = lazy("PIL.ImageCms")
import struct, zlib

import ColorSpace

FILE_PNG = "PNG"
FILE_JPEG = "JPEG"
FILE_TIFF = "TIFF"
FILE_EXR = "EXR"

FORMAT_8BIT = "8-bit"
FORMAT_16BIT = "16-bit"
FORMAT_16FLOAT = "16-bit float"
FORMAT_32FLOAT = "32-bit float"

# Valid pixel formats for each file type, the first is used when the current one isn't valid
PIXEL_FORMATS = {FILE_PNG:  [FORMAT_16BIT, FORMAT_8BIT],
                 FILE_JPEG: [FORMAT_8BIT],
                 FILE_TIFF: [FORMAT_16BIT, FORMAT_8BIT, FORMAT_16FLOAT, FORMAT_32FLOAT],
                 FILE_EXR:  [FORMAT_16FLOAT, FORMAT_32FLOAT]}

# Extensions for each file type, the first is used when the typed one doesn't match
EXTENSIONS = {FILE_PNG:  [".png"],
              FILE_JPEG: [".jpg", ".jpeg"],
              FILE_TIFF: [".tif", ".tiff"],
              FILE_EXR:  [".exr"]}

# Name of the processing record in text formats (TIFF ImageDescription, JPEG comment)
RECORD_TITLE = "AstraStack processing record"

# Type hint of the EXR attribute holding the input's ICC profile
ICC_TYPE_HINT = "application/vnd.iccprofile"

def srgbProfile() -> bytes:
    """Returns the sRGB ICC profile.

    Returns:
        The profile bytes of LittleCMS's built-in sRGB profile.
    """
    return ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes()

def isFloat(pixelFormat: str) -> bool:
    """Returns whether the pixel format is a float format.

    Args:
        pixelFormat: One of the FORMAT_* constants.

    Returns:
        True for 16-bit and 32-bit float.
    """
    return pixelFormat in (FORMAT_16FLOAT, FORMAT_32FLOAT)

def processingRecord(blackLevel: float,
                     exposure: float,
                     gains: tuple[float, float, float],
                     saturation: float,
                     stretch: str,
                     gamma: float) -> dict[str, str]:
    """Returns the processing record written into exported files.

    Args:
        blackLevel: Black level subtracted, as a fraction of full scale.
        exposure: Exposure in EV.
        gains: Red, green and blue gains, 1.0 is neutral.
        saturation: Saturation, 1.0 is neutral.
        stretch: Name of the stretch function.
        gamma: Gamma, 1.0 is neutral.

    Returns:
        The values as text by name, in pipeline order.
    """
    return {"BlackLevel": "%.6g%%" % (blackLevel*100),
            "Exposure": "%+.3f EV" % exposure,
            "RedGain": "%.6g%%" % (gains[0]*100),
            "GreenGain": "%.6g%%" % (gains[1]*100),
            "BlueGain": "%.6g%%" % (gains[2]*100),
            "Saturation": "%.6g%%" % (saturation*100),
            "Stretch": stretch,
            "Gamma": "%.6g" % gamma}

def recordText(record: dict[str, str]) -> str:
    """Returns the processing record as text, one "name: value" line per value after a title line.

    Args:
        record: The processing record from processingRecord.

    Returns:
        The ASCII text.
    """
    return "\n".join([RECORD_TITLE] + [name + ": " + value for name, value in record.items()])

def linearProfile(colorSpace: ColorSpace.ColorSpace) -> bytes:
    """Returns an ICC profile describing linear data in the color space's primaries.

    Args:
        colorSpace: Color description of the input.

    Returns:
        An ICC v2 matrix profile with the color space's primaries and white point and a gamma 1.0 tone curve.
    """
    name = colorSpace.name.encode("ascii", "replace").decode("ascii")
    return ColorSpace.iccProfile(matrix=colorSpace.matrixRGBToXYZ(),
                                 white=colorSpace.white,
                                 description="Linear " + name)

def convert(img: np.ndarray, pixelFormat: str, colorSpace: ColorSpace.ColorSpace | None) -> np.ndarray:
    """Converts the linear RGB image to the pixel format's data.

    Integer formats are display encoded like the preview (sRGB primaries, each channel clipped to 0-1, sRGB curve; only
    the sRGB curve without a color description). Float formats stay linear in the input's primaries, unclipped.

    Args:
        img: Linear RGB image in colorSpace's primaries.
        pixelFormat: One of the FORMAT_* constants.
        colorSpace: Color description of the input, None when there is none.

    Returns:
        The image as uint8, uint16 (0-65025), float16 or float32.
    """
    if(pixelFormat == FORMAT_8BIT):
        return np.around(ColorSpace.displayEncode(img, colorSpace)*255).astype('uint8')
    elif(pixelFormat == FORMAT_16BIT):
        return np.around(ColorSpace.displayEncode(img, colorSpace)*255*255).astype('uint16')
    elif(pixelFormat == FORMAT_16FLOAT):
        return img.astype('float16')
    elif(pixelFormat == FORMAT_32FLOAT):
        return img.astype('float32')

def insertPNGChunks(png: bytes, chunks: list[tuple[bytes, bytes]]) -> bytes:
    """Inserts chunks before the first IDAT chunk of the PNG data.

    Args:
        png: The PNG file data.
        chunks: The chunks to insert as (type, data).

    Returns:
        The PNG file data with the chunks.
    """
    data = b""
    for chunkType, chunkData in chunks:
        crc = struct.pack(">I", zlib.crc32(chunkType + chunkData))
        data += struct.pack(">I", len(chunkData)) + chunkType + chunkData + crc
    i = 8 # Skip the PNG signature
    while(png[i+4:i+8] != b"IDAT"):
        i += 12 + struct.unpack(">I", png[i:i+4])[0] # Length, type and CRC fields plus the data
    return png[:i] + data + png[i:]

def writePNG(fileName: str,
             img: np.ndarray,
             pixelFormat: str,
             colorSpace: ColorSpace.ColorSpace | None,
             record: dict[str, str]) -> None:
    """Writes a PNG with an sRGB chunk (none without a color description) and the record as tEXt chunks.

    Args:
        fileName: Path of the file.
        img: Linear RGB image in colorSpace's primaries.
        pixelFormat: One of the FORMAT_* constants valid for the file type.
        colorSpace: Color description of the input, None when there is none.
        record: The processing record from processingRecord.
    """
    data = convert(img=img, pixelFormat=pixelFormat, colorSpace=colorSpace)
    success, png = cv2.imencode(".png", cv2.cvtColor(data, cv2.COLOR_RGB2BGR))
    if(not success):
        raise Exception("PNG encoding failed")
    chunks = []
    if(colorSpace is not None):
        chunks.append((b"sRGB", b"\0")) # Rendering intent: perceptual
    for name, value in record.items():
        chunks.append((b"tEXt", name.encode("latin-1") + b"\0" + value.encode("latin-1")))
    png = insertPNGChunks(png.tobytes(), chunks)
    with open(fileName, "wb") as file:
        file.write(png)

def writeJPEG(fileName: str,
              img: np.ndarray,
              pixelFormat: str,
              colorSpace: ColorSpace.ColorSpace | None,
              record: dict[str, str]) -> None:
    """Writes a JPEG with the sRGB profile (none without a color description) and the record as a comment.

    Args:
        fileName: Path of the file.
        img: Linear RGB image in colorSpace's primaries.
        pixelFormat: One of the FORMAT_* constants valid for the file type.
        colorSpace: Color description of the input, None when there is none.
        record: The processing record from processingRecord.
    """
    data = convert(img=img, pixelFormat=pixelFormat, colorSpace=colorSpace)
    options = {"quality": 95, "comment": recordText(record)}
    if(colorSpace is not None):
        options["icc_profile"] = srgbProfile()
    Image.fromarray(data).save(fp=fileName, format="JPEG", **options)

def writeTIFF(fileName: str,
              img: np.ndarray,
              pixelFormat: str,
              colorSpace: ColorSpace.ColorSpace | None,
              record: dict[str, str]) -> None:
    """Writes a TIFF with the record as ImageDescription.

    Integer formats get the sRGB profile, float formats a linear profile with the input's primaries and white point,
    and there is no profile without a color description.

    Args:
        fileName: Path of the file.
        img: Linear RGB image in colorSpace's primaries.
        pixelFormat: One of the FORMAT_* constants valid for the file type.
        colorSpace: Color description of the input, None when there is none.
        record: The processing record from processingRecord.
    """
    if(colorSpace is None):
        profile = None
    elif(isFloat(pixelFormat)):
        profile = linearProfile(colorSpace)
    else:
        profile = srgbProfile()
    tifffile.imwrite(fileName,
                     data=convert(img=img, pixelFormat=pixelFormat, colorSpace=colorSpace),
                     photometric='rgb',
                     compression='zlib',
                     iccprofile=profile,
                     description=recordText(record),
                     metadata=None)

def writeEXR(fileName: str,
             img: np.ndarray,
             pixelFormat: str,
             colorSpace: ColorSpace.ColorSpace | None,
             record: dict[str, str]) -> None:
    """Writes a linear EXR with the record as string attributes.

    The input's primaries and white point go in chromaticities, and the input's ICC profile, when it had one, in an
    iccProfile attribute. Neither is written without a color description.

    Args:
        fileName: Path of the file.
        img: Linear RGB image in colorSpace's primaries.
        pixelFormat: One of the FORMAT_* constants valid for the file type.
        colorSpace: Color description of the input, None when there is none.
        record: The processing record from processingRecord.
    """
    img = convert(img=img, pixelFormat=pixelFormat, colorSpace=colorSpace)
    header = {"compression": OpenEXR.ZIP_COMPRESSION,
              "type": OpenEXR.scanlineimage}
    if(colorSpace is not None):
        (r, g, b), w = colorSpace.primaries, colorSpace.white
        header["chromaticities"] = (r[0], r[1], g[0], g[1], b[0], b[1], w[0], w[1])
        if(colorSpace.icc is not None):
            header["iccProfile"] = OpenEXR.Bytes(colorSpace.icc, ICC_TYPE_HINT)
    header.update(record)
    channels = {"R": np.ascontiguousarray(img[:,:,0]),
                "G": np.ascontiguousarray(img[:,:,1]),
                "B": np.ascontiguousarray(img[:,:,2])}
    with OpenEXR.File(header, channels) as file:
        file.write(fileName)

WRITERS = {FILE_PNG:  writePNG,
           FILE_JPEG: writeJPEG,
           FILE_TIFF: writeTIFF,
           FILE_EXR:  writeEXR}

def write(fileName: str,
          img: np.ndarray,
          fileType: str,
          pixelFormat: str,
          colorSpace: ColorSpace.ColorSpace | None,
          record: dict[str, str]) -> None:
    """Writes the image with the file type and pixel format, and the processing record as file attributes.

    Args:
        fileName: Path of the file.
        img: Linear RGB image in colorSpace's primaries.
        fileType: One of the FILE_* constants.
        pixelFormat: One of the FORMAT_* constants valid for the file type.
        colorSpace: Color description of the input, None when there is none.
        record: The processing record from processingRecord.
    """
    WRITERS[fileType](fileName=fileName,
                      img=img,
                      pixelFormat=pixelFormat,
                      colorSpace=colorSpace,
                      record=record)
