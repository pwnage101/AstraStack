"""Tests for the export writers and their embedded metadata in Export.py."""
import io
import os

import colour
import cv2
import numpy as np
import pytest
from PIL import Image, ImageCms

import ColorSpace
import Export
from Process import applyLevels
from tests.fixtureFiles import colorTestImage, displayP3Profile, exportTags, lutSRGBProfile, readExport, sameProfile

# Valid (file type, pixel format) combinations and the extension used for them
COMBINATIONS = [
    (fileType, pixelFormat, Export.EXTENSIONS[fileType][0])
    for fileType, pixelFormats in Export.PIXEL_FORMATS.items()
    for pixelFormat in pixelFormats
]

DTYPES = {
    Export.FORMAT_8BIT: np.uint8,
    Export.FORMAT_16BIT: np.uint16,
    Export.FORMAT_16FLOAT: np.float16,
    Export.FORMAT_32FLOAT: np.float32,
}

# Linear value of the sRGB code 128
CODE_128 = float(ColorSpace.SRGB.linearize(np.array([128/255]))[0])

NEUTRAL_RECORD = Export.processingRecord(
    blackLevel=0,
    exposure=0,
    gains=(1, 1, 1),
    saturation=1,
    stretch="None",
    gamma=1,
)


def combinationId(combination: tuple[str, str, str]) -> str:
    """Returns a test id for a (file type, pixel format, extension) combination.

    Args:
        combination: The combination.

    Returns:
        E.g. "TIFF 16-bit float".
    """
    return combination[0] + " " + combination[1]


def export(
    path: str,
    img: np.ndarray,
    fileType: str,
    pixelFormat: str,
    colorSpace: ColorSpace.ColorSpace | None,
    record: dict[str, str] = NEUTRAL_RECORD,
) -> dict[str, object]:
    """Writes the image and returns its embedded tags.

    Args:
        path: Path of the file.
        img: Linear RGB image.
        fileType: One of the Export.FILE_* constants.
        pixelFormat: One of the Export.FORMAT_* constants.
        colorSpace: Color description of the image, or None.
        record: The processing record.

    Returns:
        The tags as ``fixtureFiles.exportTags`` returns them.
    """
    Export.write(
        fileName=path,
        img=img,
        fileType=fileType,
        pixelFormat=pixelFormat,
        colorSpace=colorSpace,
        record=record,
    )
    return exportTags(path)


def assertSRGBTag(tags: dict[str, object], fileType: str) -> None:
    """Checks the sRGB tag of an integer export: only the sRGB chunk for PNG, the sRGB profile otherwise.

    Args:
        tags: The tags of the file.
        fileType: One of the Export.FILE_* constants.
    """
    if(fileType == Export.FILE_PNG):
        assert tags["pngColor"] == [b"sRGB"]
    else:
        assert sameProfile(tags["icc"], Export.srgbProfile())


def assertLinearTag(tags: dict[str, object], fileType: str, name: str) -> None:
    """Checks the linear tag of a float export: EXR chromaticities, or a gamma 1.0 matrix profile for TIFF.

    Args:
        tags: The tags of the file.
        fileType: Export.FILE_TIFF or Export.FILE_EXR.
        name: colour-science name of the expected primaries and white point.
    """
    expected = colour.RGB_COLOURSPACES[name]
    if(fileType == Export.FILE_EXR):
        chromaticities = np.array(tags["chromaticities"])
        assert np.allclose(chromaticities[:6], np.ravel(expected.primaries), atol=1e-4)
        assert np.allclose(chromaticities[6:], expected.whitepoint, atol=1e-4)
    else:
        icc = tags["icc"]
        assert icc is not None and ColorSpace.isMatrixProfile(icc)
        iccTags = ColorSpace.iccTags(icc)
        assert all(ColorSpace.iccCurve(iccTags[t]) == ("gamma", 1.0) for t in ("rTRC", "gTRC", "bTRC"))
        colorSpace = ColorSpace.fromICC(icc)
        assert np.allclose(colorSpace.primaries, expected.primaries, atol=1e-4)
        assert np.allclose(colorSpace.white, expected.whitepoint, atol=1e-4)


def assertNoColorTags(tags: dict[str, object]) -> None:
    """Checks that a file has no ICC profile, PNG color chunk or EXR chromaticities/iccProfile.

    Args:
        tags: The tags of the file.
    """
    assert tags["icc"] is None
    assert tags["pngColor"] == []
    assert tags["chromaticities"] is None
    assert tags["iccAttribute"] is None


@pytest.fixture
def testJPEG(testInputs: str) -> np.ndarray:
    """Returns the 8-bit RGB pixels of ``test.jpg``."""
    return cv2.cvtColor(cv2.imread(os.path.join(testInputs, "test.jpg")), cv2.COLOR_BGR2RGB)


def test_convert() -> None:
    """Integer formats are display encoded and clipped; float formats keep the linear values."""
    img = np.array([[[CODE_128, -0.5, 2.0]]], np.float32)
    assert Export.convert(img=img, pixelFormat=Export.FORMAT_8BIT, colorSpace=None)[0, 0].tolist() == [128, 0, 255]
    assert Export.convert(img=img, pixelFormat=Export.FORMAT_16BIT, colorSpace=None)[0, 0].tolist() == [
        128*255,
        0,
        65025,
    ]
    half = Export.convert(img=img, pixelFormat=Export.FORMAT_16FLOAT, colorSpace=None)
    assert half.dtype == np.float16
    assert half[0, 0] == pytest.approx(img[0, 0], rel=1e-3)
    assert np.array_equal(Export.convert(img=img, pixelFormat=Export.FORMAT_32FLOAT, colorSpace=None), img)


@pytest.mark.parametrize("combination", COMBINATIONS, ids=combinationId)
def test_export_round_trip(tmp_path: str, testJPEG: np.ndarray, combination: tuple[str, str, str]) -> None:
    """Every file type and pixel format writes the expected dtype, values and sRGB tag for an sRGB input."""
    fileType, pixelFormat, extension = combination
    path = os.path.join(tmp_path, "export" + extension)
    img = ColorSpace.SRGB.linearizeInteger(testJPEG)
    tags = export(path=path, img=img, fileType=fileType, pixelFormat=pixelFormat, colorSpace=ColorSpace.SRGB)
    data = readExport(path)
    assert data.dtype == DTYPES[pixelFormat]
    assert data.shape == testJPEG.shape
    if(fileType == Export.FILE_JPEG):
        assert np.abs(data.astype(np.float64) - testJPEG).mean() < 2
    elif(pixelFormat == Export.FORMAT_8BIT):
        assert np.array_equal(data, testJPEG)
    elif(pixelFormat == Export.FORMAT_16BIT):
        assert np.array_equal(data, testJPEG.astype(np.uint16)*255)
    elif(pixelFormat == Export.FORMAT_16FLOAT):
        assert np.allclose(data, img, rtol=2**-10, atol=2**-24)
    else:
        assert np.array_equal(data, img)
    if(Export.isFloat(pixelFormat)):
        assertLinearTag(tags=tags, fileType=fileType, name="sRGB")
    else:
        assertSRGBTag(tags=tags, fileType=fileType)
    assert tags["iccAttribute"] is None
    assert tags["record"] == NEUTRAL_RECORD


def test_linear_tiff_profile_is_readable(tmp_path: str) -> None:
    """The linear TIFF profile is readable by LittleCMS and converts linear 0.5 to sRGB 188."""
    path = os.path.join(tmp_path, "linear.tif")
    tags = export(
        path=path,
        img=np.zeros((2, 2, 3), np.float32),
        fileType=Export.FILE_TIFF,
        pixelFormat=Export.FORMAT_32FLOAT,
        colorSpace=ColorSpace.SRGB,
    )
    assert tags["icc"] == Export.linearProfile(ColorSpace.SRGB)
    profile = ImageCms.ImageCmsProfile(io.BytesIO(tags["icc"]))
    gray = Image.new("RGB", (1, 1), (128, 128, 128))
    converted = ImageCms.profileToProfile(gray, profile, ImageCms.createProfile("sRGB"), renderingIntent=1)
    assert converted.getpixel((0, 0)) == (188, 188, 188)


def test_processingRecord() -> None:
    """The record holds every tone and level value as text, in pipeline order."""
    record = Export.processingRecord(
        blackLevel=0.01,
        exposure=1.5,
        gains=(1.1, 0.95, 1.05),
        saturation=0.8,
        stretch="Square Root",
        gamma=1.2,
    )
    assert list(record.items()) == [
        ("BlackLevel", "1%"),
        ("Exposure", "+1.500 EV"),
        ("RedGain", "110%"),
        ("GreenGain", "95%"),
        ("BlueGain", "105%"),
        ("Saturation", "80%"),
        ("Stretch", "Square Root"),
        ("Gamma", "1.2"),
    ]
    assert Export.recordText(record).split("\n")[:2] == [Export.RECORD_TITLE, "BlackLevel: 1%"]


@pytest.mark.parametrize("combination", COMBINATIONS, ids=combinationId)
def test_processing_record_in_every_format(tmp_path: str, combination: tuple[str, str, str]) -> None:
    """The processing record is written as PNG tEXt, JPEG comment, TIFF ImageDescription or EXR attributes."""
    fileType, pixelFormat, extension = combination
    record = Export.processingRecord(
        blackLevel=0.01,
        exposure=-0.5,
        gains=(1, 1, 1),
        saturation=1,
        stretch="ArcSinH",
        gamma=1,
    )
    tags = export(
        path=os.path.join(tmp_path, "record" + extension),
        img=np.full((4, 4, 3), 0.5, np.float32),
        fileType=fileType,
        pixelFormat=pixelFormat,
        colorSpace=ColorSpace.SRGB,
        record=record,
    )
    assert tags["record"] == record


@pytest.mark.parametrize(
    "fileType, pixelFormat",
    [
        (Export.FILE_TIFF, Export.FORMAT_16FLOAT),
        (Export.FILE_TIFF, Export.FORMAT_32FLOAT),
        (Export.FILE_EXR, Export.FORMAT_16FLOAT),
        (Export.FILE_EXR, Export.FORMAT_32FLOAT),
    ],
)
def test_float_export_keeps_display_p3(tmp_path: str, fileType: str, pixelFormat: str) -> None:
    """Float exports of a Display P3 input are tagged linear P3 and keep the leveled values, unclipped."""
    colorSpace = ColorSpace.fromICC(displayP3Profile())
    img = applyLevels(img=colorSpace.linearizeInteger(colorTestImage()), blackLevel=0.01, exposure=1)
    path = os.path.join(tmp_path, "p3" + Export.EXTENSIONS[fileType][0])
    tags = export(path=path, img=img, fileType=fileType, pixelFormat=pixelFormat, colorSpace=colorSpace)
    assertLinearTag(tags=tags, fileType=fileType, name="Display P3")
    data = readExport(path)
    if(pixelFormat == Export.FORMAT_16FLOAT):
        assert np.allclose(data, img, rtol=2**-10, atol=2**-12)
    else:
        assert np.allclose(data, img, rtol=1e-5, atol=1e-5)
    assert data.max() > 1 and data.min() < 0


@pytest.mark.parametrize(
    "fileType, pixelFormat",
    [
        (Export.FILE_PNG, Export.FORMAT_8BIT),
        (Export.FILE_JPEG, Export.FORMAT_8BIT),
        (Export.FILE_TIFF, Export.FORMAT_16BIT),
    ],
)
def test_integer_export_of_display_p3_is_srgb(tmp_path: str, fileType: str, pixelFormat: str) -> None:
    """Integer exports of a Display P3 input are converted to sRGB and tagged sRGB."""
    colorSpace = ColorSpace.fromICC(displayP3Profile())
    img = colorSpace.linearizeInteger(colorTestImage())
    path = os.path.join(tmp_path, "p3" + Export.EXTENSIONS[fileType][0])
    tags = export(path=path, img=img, fileType=fileType, pixelFormat=pixelFormat, colorSpace=colorSpace)
    assertSRGBTag(tags=tags, fileType=fileType)
    if(fileType == Export.FILE_PNG):
        with Image.open(os.path.join(tmp_path, "p3.png")) as im:
            exported = np.asarray(im).astype(int)
        source = Image.fromarray(colorTestImage())
        profile = ImageCms.ImageCmsProfile(io.BytesIO(displayP3Profile()))
        expected = ImageCms.profileToProfile(source, profile, ImageCms.createProfile("sRGB"), renderingIntent=1)
        assert np.abs(exported - np.asarray(expected).astype(int)).max() <= 1


def test_exr_carries_the_input_icc_profile(tmp_path: str) -> None:
    """EXR exports carry the input's ICC profile bytes; inputs without one have no iccProfile attribute."""
    colorSpace = ColorSpace.fromICC(displayP3Profile())
    img = np.full((4, 4, 3), 0.5, np.float32)
    tags = export(
        path=os.path.join(tmp_path, "p3.exr"),
        img=img,
        fileType=Export.FILE_EXR,
        pixelFormat=Export.FORMAT_32FLOAT,
        colorSpace=colorSpace,
    )
    assert tags["iccAttribute"] == displayP3Profile()
    tags = export(
        path=os.path.join(tmp_path, "untagged.exr"),
        img=img,
        fileType=Export.FILE_EXR,
        pixelFormat=Export.FORMAT_32FLOAT,
        colorSpace=ColorSpace.SRGB,
    )
    assertLinearTag(tags=tags, fileType=Export.FILE_EXR, name="sRGB")
    assert tags["iccAttribute"] is None


def test_float_export_of_lut_profile_is_rec2020(tmp_path: str) -> None:
    """Float exports of a LUT profile input are linear Rec.2020, and the EXR carries the original profile."""
    colorSpace = ColorSpace.lutColorSpace(lutSRGBProfile())
    img = np.full((4, 4, 3), 0.5, np.float32)
    for fileType in (Export.FILE_TIFF, Export.FILE_EXR):
        tags = export(
            path=os.path.join(tmp_path, "lut" + Export.EXTENSIONS[fileType][0]),
            img=img,
            fileType=fileType,
            pixelFormat=Export.FORMAT_32FLOAT,
            colorSpace=colorSpace,
        )
        assertLinearTag(tags=tags, fileType=fileType, name="ITU-R BT.2020")
    assert tags["iccAttribute"] == lutSRGBProfile()


def test_grayscale_image_file_exports_are_tagged(tmp_path: str, testInputs: str) -> None:
    """A grayscale image file keeps its sRGB description, so its exports are tagged."""
    img, colorSpace = ColorSpace.describeImage(
        os.path.join(testInputs, "dso.jpg"),
        cv2.imread(os.path.join(testInputs, "dso.jpg")),
    )
    linear = colorSpace.linearizeInteger(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    tags = export(
        path=os.path.join(tmp_path, "dso.exr"),
        img=linear,
        fileType=Export.FILE_EXR,
        pixelFormat=Export.FORMAT_16FLOAT,
        colorSpace=colorSpace,
    )
    assertLinearTag(tags=tags, fileType=Export.FILE_EXR, name="sRGB")
    tags = export(
        path=os.path.join(tmp_path, "dso.png"),
        img=linear,
        fileType=Export.FILE_PNG,
        pixelFormat=Export.FORMAT_8BIT,
        colorSpace=colorSpace,
    )
    assertSRGBTag(tags=tags, fileType=Export.FILE_PNG)


@pytest.mark.parametrize("combination", COMBINATIONS, ids=combinationId)
def test_untagged_exports_have_no_color_tags(tmp_path: str, combination: tuple[str, str, str]) -> None:
    """Inputs without a color description (Bayer, grayscale video, FITS) are written without color tags."""
    fileType, pixelFormat, extension = combination
    path = os.path.join(tmp_path, "untagged" + extension)
    img = np.full((4, 4, 3), CODE_128, np.float32)
    tags = export(path=path, img=img, fileType=fileType, pixelFormat=pixelFormat, colorSpace=None)
    assertNoColorTags(tags)
    assert tags["record"] == NEUTRAL_RECORD
    if(pixelFormat == Export.FORMAT_8BIT and fileType != Export.FILE_JPEG):
        assert np.all(readExport(path) == 128)


def test_insertPNGChunks(tmp_path: str) -> None:
    """Chunks are inserted before the first IDAT with valid CRCs, and the image still decodes."""
    success, png = cv2.imencode(".png", colorTestImage())
    path = os.path.join(tmp_path, "chunks.png")
    with open(path, "wb") as f:
        f.write(Export.insertPNGChunks(png=png.tobytes(), chunks=[(b"tEXt", b"a\0b"), (b"sRGB", b"\0")]))
    tags = exportTags(path)
    assert tags["pngColor"] == [b"sRGB"]
    assert tags["record"] == {"a": "b"}
    assert np.array_equal(cv2.imread(path), colorTestImage())


def test_pixel_formats_and_extensions() -> None:
    """Each file type lists its valid pixel formats and extensions."""
    assert Export.PIXEL_FORMATS[Export.FILE_JPEG] == [Export.FORMAT_8BIT]
    assert not any(Export.isFloat(f) for f in Export.PIXEL_FORMATS[Export.FILE_PNG])
    assert all(Export.isFloat(f) for f in Export.PIXEL_FORMATS[Export.FILE_EXR])
    assert set(Export.WRITERS) == set(Export.EXTENSIONS) == set(Export.PIXEL_FORMATS)
