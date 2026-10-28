"""Tests for the color descriptions, transfer functions and ICC handling in ColorSpace.py."""
import os
from types import SimpleNamespace
from typing import Callable

import colour
import cv2
import numpy as np
import pytest
from PIL import Image, ImageCms

import ColorSpace
import Export
from conftest import FakeUI
from Globals import Object
from tests.fixtureFiles import colorTestImage, displayP3Profile, grayProfile, lutSRGBProfile
from Video import Video, loadFramesVideo

CODES = np.linspace(0, 1, 101)


def p3ColorSpace() -> ColorSpace.ColorSpace:
    """Returns the color description of the Display P3 test profile.

    Returns:
        The ColorSpace read from ``displayP3Profile()``.
    """
    return ColorSpace.fromICC(displayP3Profile())


def iccCurves(*curves: bytes) -> tuple:
    """Returns the tone curves of ICC curve tags.

    Args:
        *curves: The curv or para tags, one per channel.

    Returns:
        The curves as ``ColorSpace.iccCurve`` returns them.
    """
    return tuple(ColorSpace.iccCurve(curve) for curve in curves)


def curvTable(values: np.ndarray) -> bytes:
    """Returns an ICC curv tag with a lookup table.

    Args:
        values: Linear values (0-1) at evenly spaced codes.

    Returns:
        The tag data.
    """
    table = np.around(np.asarray(values)*65535).astype(">u2")
    return b"curv" + bytes(4) + len(table).to_bytes(4, "big") + table.tobytes()


def paraSRGB() -> bytes:
    """Returns an ICC para tag (function type 3) holding the sRGB curve.

    Returns:
        The tag data.
    """
    params = (2.4, 1/1.055, 0.055/1.055, 1/12.92, 0.04045)
    data = b"".join(int(round(p*65536)).to_bytes(4, "big", signed=True) for p in params)
    return b"para" + bytes(4) + (3).to_bytes(2, "big") + bytes(2) + data


ENCODINGS = {
    "sRGB": (ColorSpace.SRGB, colour.models.eotf_inverse_sRGB),
    "BT.709": (ColorSpace.named("ITU-R BT.709", ColorSpace.TRANSFER_BT709), colour.models.oetf_BT709),
    "BT.2020": (ColorSpace.named("ITU-R BT.2020", ColorSpace.TRANSFER_BT2020), colour.models.oetf_BT2020),
    "HLG": (ColorSpace.named("ITU-R BT.2020", ColorSpace.TRANSFER_HLG), colour.models.oetf_BT2100_HLG),
    "PQ": (
        ColorSpace.named("ITU-R BT.2020", ColorSpace.TRANSFER_PQ),
        lambda x: colour.models.eotf_inverse_ST2084(x*ColorSpace.PQ_REFERENCE_WHITE),
    ),
    "gamma": (ColorSpace.named("ITU-R BT.709", ColorSpace.TRANSFER_GAMMA, params=2.2), lambda x: x**(1/2.2)),
    "linear": (ColorSpace.named("ITU-R BT.709", ColorSpace.TRANSFER_LINEAR), lambda x: x),
}


@pytest.mark.filterwarnings("ignore:invalid value encountered in log")
@pytest.mark.parametrize("name", ENCODINGS)
def test_transfer_round_trip(name: str) -> None:
    """Linearizing values encoded with the transfer function's inverse gives them back."""
    colorSpace, encode = ENCODINGS[name]
    linear = CODES if name != "PQ" else CODES*2
    assert colorSpace.linearize(encode(linear)) == pytest.approx(linear, abs=1e-6)


def test_transfer_known_values() -> None:
    """Spot checks: sRGB 0.5 is 0.214 linear and PQ 0.58 is about the 203 cd/m2 reference white."""
    assert ColorSpace.SRGB.linearize(np.array([0.5]))[0] == pytest.approx(0.21404, abs=1e-5)
    pq = ColorSpace.named("ITU-R BT.2020", ColorSpace.TRANSFER_PQ)
    assert pq.linearize(np.array([0.58]))[0] == pytest.approx(1, abs=0.01)


@pytest.mark.parametrize(
    "curve, encode",
    [
        (b"curv" + bytes(4) + (0).to_bytes(4, "big"), lambda x: x),
        (b"curv" + bytes(4) + (1).to_bytes(4, "big") + (563).to_bytes(2, "big"), lambda x: x**(256/563)),
        (curvTable(colour.models.eotf_sRGB(np.linspace(0, 1, 1024))), colour.models.eotf_inverse_sRGB),
        (paraSRGB(), colour.models.eotf_inverse_sRGB),
    ],
    ids=["identity", "gamma", "table", "para"],
)
def test_icc_curve_round_trip(curve: bytes, encode: Callable[[np.ndarray], np.ndarray]) -> None:
    """Each ICC curve type linearizes values encoded with the curve's inverse."""
    colorSpace = ColorSpace.ColorSpace(
        name="ICC test",
        primaries=ColorSpace.SRGB_PRIMARIES,
        white=ColorSpace.D65,
        transfer=ColorSpace.TRANSFER_ICC,
        params=iccCurves(curve, curve, curve),
    )
    img = np.dstack([encode(CODES)[None, :]]*3)
    assert colorSpace.linearize(img)[0] == pytest.approx(np.dstack([CODES[None, :]]*3)[0], abs=2e-4)


@pytest.mark.parametrize("name", ENCODINGS)
@pytest.mark.parametrize("dtype", [np.uint8, np.uint16])
def test_linearizeInteger_matches_formula(name: str, dtype: type) -> None:
    """The lookup table gives the same values as the transfer function on every code."""
    colorSpace = ENCODINGS[name][0]
    maxValue = np.iinfo(dtype).max
    codes = np.arange(0, maxValue + 1, max(1, maxValue//1000), dtype=dtype)
    expected = colorSpace.linearize((codes/maxValue).astype(np.float32))
    assert colorSpace.linearizeInteger(codes) == pytest.approx(expected, abs=1e-6)


def test_linearizeInteger_icc_channel_order() -> None:
    """ICC profiles keep a curve per channel, applied in BGR order when asked."""
    gamma1 = b"curv" + bytes(4) + (0).to_bytes(4, "big")
    gamma2 = b"curv" + bytes(4) + (1).to_bytes(4, "big") + (512).to_bytes(2, "big")
    colorSpace = ColorSpace.ColorSpace(
        name="ICC test",
        primaries=ColorSpace.SRGB_PRIMARIES,
        white=ColorSpace.D65,
        transfer=ColorSpace.TRANSFER_ICC,
        params=iccCurves(gamma1, gamma2, gamma2),
    )
    img = np.full((1, 1, 3), 128, np.uint8)
    rgb = colorSpace.linearizeInteger(img)[0, 0]
    bgr = colorSpace.linearizeInteger(img, bgr=True)[0, 0]
    assert rgb == pytest.approx([128/255, (128/255)**2, (128/255)**2], abs=1e-6)
    assert bgr == pytest.approx(rgb[::-1], abs=1e-6)
    assert colorSpace.linearize(img/255, bgr=True)[0, 0] == pytest.approx(bgr, abs=1e-6)


def test_icc_write_and_read() -> None:
    """A generated matrix profile reads back with its primaries, white point, tone curve and description."""
    p3 = colour.RGB_COLOURSPACES["Display P3"]
    icc = displayP3Profile()
    assert ColorSpace.isMatrixProfile(icc)
    assert not ColorSpace.isGrayProfile(icc)
    colorSpace = ColorSpace.fromICC(icc)
    assert colorSpace.name == "Display P3 test"
    assert colorSpace.transfer == ColorSpace.TRANSFER_ICC
    assert colorSpace.icc == icc
    assert np.allclose(colorSpace.primaries, p3.primaries, atol=1e-4)
    assert np.allclose(colorSpace.white, p3.whitepoint, atol=1e-4)
    assert colorSpace.params == (("gamma", 563/256),)*3
    assert ColorSpace.fromICC(icc) is colorSpace


def test_icc_linear_profile() -> None:
    """A gamma 1.0 profile has identity tone curves."""
    icc = ColorSpace.iccProfile(
        matrix=ColorSpace.SRGB.matrixRGBToXYZ(),
        white=ColorSpace.D65,
        description="Linear test",
    )
    tags = ColorSpace.iccTags(icc)
    assert all(ColorSpace.iccCurve(tags[t]) == ("gamma", 1.0) for t in ("rTRC", "gTRC", "bTRC"))
    assert np.allclose(ColorSpace.fromICC(icc).primaries, ColorSpace.SRGB_PRIMARIES, atol=1e-4)


def test_icc_srgb_profile_is_srgb() -> None:
    """LittleCMS's sRGB profile (a parametric curve) is recognized as exact sRGB, keeping the ICC bytes."""
    icc = ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes()
    colorSpace = ColorSpace.fromICC(icc)
    assert colorSpace.name == "sRGB"
    assert colorSpace.transfer == ColorSpace.TRANSFER_SRGB
    assert colorSpace.icc == icc
    assert ColorSpace.same(colorSpace, ColorSpace.SRGB)


def test_icc_lut_profile_is_not_matrix() -> None:
    """A LUT based profile isn't a matrix profile."""
    assert not ColorSpace.isMatrixProfile(lutSRGBProfile())


def test_gray_icc() -> None:
    """A gray TRC profile becomes its tone curve on all channels, without keeping the ICC bytes."""
    icc = grayProfile(2.0)
    assert ColorSpace.isGrayProfile(icc)
    assert not ColorSpace.isMatrixProfile(icc)
    colorSpace = ColorSpace.fromGrayICC(icc)
    assert colorSpace.transfer == ColorSpace.TRANSFER_ICC
    assert colorSpace.icc is None
    assert colorSpace.params == (("gamma", 2.0),)*3
    assert colorSpace.linearizeInteger(np.full((1, 1, 3), 128, np.uint8))[0, 0] == pytest.approx([(128/255)**2]*3)


def test_gray_icc_image(tmp_path: str) -> None:
    """A grayscale JPEG with a gray profile is described by the profile's curve."""
    path = os.path.join(tmp_path, "gray.jpg")
    Image.fromarray(colorTestImage()[:, :, 0]).save(path, icc_profile=grayProfile(2.0))
    img, colorSpace = Video.decodeImage(path)
    assert colorSpace.name == "Gray ICC"
    assert colorSpace.params == (("gamma", 2.0),)*3


@pytest.mark.parametrize(
    "primaries, transfer, name, expected",
    [
        (1, 1, "ITU-R BT.709", (ColorSpace.TRANSFER_BT709, None)),
        (9, 16, "ITU-R BT.2020", (ColorSpace.TRANSFER_PQ, None)),
        (9, 18, "ITU-R BT.2020", (ColorSpace.TRANSFER_HLG, None)),
        (9, 14, "ITU-R BT.2020", (ColorSpace.TRANSFER_BT2020, None)),
        (12, 13, "Display P3", (ColorSpace.TRANSFER_SRGB, None)),
        (5, 5, "ITU-R BT.470 - 625", (ColorSpace.TRANSFER_GAMMA, 2.8)),
        (1, 8, "ITU-R BT.709", (ColorSpace.TRANSFER_LINEAR, None)),
        (2, 2, "ITU-R BT.709", (ColorSpace.TRANSFER_BT709, None)),
    ],
)
def test_fromCICP(primaries: int, transfer: int, name: str, expected: tuple) -> None:
    """CICP codes map to the named primaries and transfer function; unknown codes are Rec.709."""
    colorSpace = ColorSpace.fromCICP(primaries, transfer)
    assert colorSpace.name == name
    assert (colorSpace.transfer, colorSpace.params) == expected
    assert np.allclose(colorSpace.primaries, colour.RGB_COLOURSPACES[name].primaries)


def pngWithChunks(path: str, chunks: list[tuple[bytes, bytes]]) -> str:
    """Writes an 8-bit PNG with extra chunks before its image data.

    Args:
        path: Path of the file.
        chunks: The chunks as (type, data).

    Returns:
        The path.
    """
    success, png = cv2.imencode(".png", colorTestImage())
    with open(path, "wb") as f:
        f.write(Export.insertPNGChunks(png=png.tobytes(), chunks=chunks))
    return path


def test_pngChunks(tmp_path: str) -> None:
    """The chunks before IDAT are read in order; other files have none."""
    path = pngWithChunks(os.path.join(tmp_path, "a.png"), [(b"tEXt", b"a\0b"), (b"sRGB", b"\0")])
    assert ColorSpace.pngChunks(path)[-2:] == [(b"tEXt", b"a\0b"), (b"sRGB", b"\0")]
    jpeg = os.path.join(tmp_path, "a.jpg")
    Image.fromarray(colorTestImage()).save(jpeg)
    assert ColorSpace.pngChunks(jpeg) == []


@pytest.mark.parametrize(
    "chunks, expected",
    [
        ([], None),
        ([(b"sRGB", b"\0")], ("sRGB", ColorSpace.TRANSFER_SRGB, None)),
        ([(b"cICP", bytes([9, 16, 0, 1])), (b"sRGB", b"\0")], ("ITU-R BT.2020", ColorSpace.TRANSFER_PQ, None)),
        ([(b"gAMA", (45455).to_bytes(4, "big"))], ("PNG", ColorSpace.TRANSFER_GAMMA, 100000/45455)),
        ([(b"gAMA", (100000).to_bytes(4, "big"))], ("PNG", ColorSpace.TRANSFER_LINEAR, None)),
    ],
    ids=["none", "sRGB", "cICP", "gAMA", "linear"],
)
def test_fromPNGChunks(tmp_path: str, chunks: list[tuple[bytes, bytes]], expected: tuple | None) -> None:
    """cICP wins over sRGB, which wins over gAMA/cHRM; a PNG without color chunks has no description."""
    colorSpace = ColorSpace.fromPNGChunks(pngWithChunks(os.path.join(tmp_path, "a.png"), chunks))
    if(expected is None):
        assert colorSpace is None
    else:
        assert (colorSpace.name, colorSpace.transfer) == expected[:2]
        assert colorSpace.params == pytest.approx(expected[2])


def test_fromPNGChunks_cHRM(tmp_path: str) -> None:
    """cHRM gives the white point and primaries."""
    p3 = colour.RGB_COLOURSPACES["Display P3"]
    values = np.concatenate([p3.whitepoint, np.ravel(p3.primaries)])
    cHRM = b"".join(int(round(v*100000)).to_bytes(4, "big") for v in values)
    colorSpace = ColorSpace.fromPNGChunks(pngWithChunks(os.path.join(tmp_path, "a.png"), [(b"cHRM", cHRM)]))
    assert colorSpace.transfer == ColorSpace.TRANSFER_SRGB
    assert np.allclose(colorSpace.primaries, p3.primaries, atol=1e-5)
    assert np.allclose(colorSpace.white, p3.whitepoint, atol=1e-5)


def fakeRaw(**overrides: object) -> SimpleNamespace:
    """Returns a stand-in for ``rawpy.RawPy`` with the color fields ``ColorSpace.fromRAW`` reads.

    Args:
        **overrides: Fields to replace.

    Returns:
        The fake RAW, by default a camera with an XYZ matrix shot under a warm illuminant.
    """
    raw = SimpleNamespace(
        camera_whitebalance=[2.0, 1.0, 1.5, 0.0],
        daylight_whitebalance=[1.8, 1.0, 1.4, 0.0],
        rgb_xyz_matrix=np.array([[0.7, -0.1, -0.05], [-0.3, 1.2, 0.1], [-0.05, 0.2, 0.6], [0, 0, 0]]),
        color_matrix=np.zeros((3, 4)),
    )
    for name, value in overrides.items():
        setattr(raw, name, value)
    return raw


def test_fromRAW_white_maps_to_white_point() -> None:
    """Camera white maps to the white point with luminance 1, and the white point is the matrix's white."""
    colorSpace = ColorSpace.fromRAW(fakeRaw())
    M = np.array(colorSpace.matrix)
    assert colorSpace.transfer == ColorSpace.TRANSFER_LINEAR
    assert (M @ np.ones(3))[1] == pytest.approx(1)
    assert colour.XYZ_to_xy(M @ np.ones(3)) == pytest.approx(colorSpace.white)
    assert ColorSpace.whiteRGB(colorSpace) == pytest.approx([1, 1, 1], abs=1e-5)


def test_fromRAW_d65_white_balance() -> None:
    """With the white balance of a D65 neutral, camera white is D65."""
    xyzToCamera = fakeRaw().rgb_xyz_matrix[:3]
    cameraD65 = xyzToCamera @ colour.xy_to_XYZ(ColorSpace.D65)
    colorSpace = ColorSpace.fromRAW(fakeRaw(camera_whitebalance=list(1/cameraD65)))
    assert colorSpace.white == pytest.approx(ColorSpace.D65, abs=1e-6)


def test_fromRAW_color_matrix_only() -> None:
    """With only the camera -> sRGB matrix (e.g. DNG), the daylight white balance maps camera white to D65."""
    colorMatrix = np.hstack([np.eye(3), np.zeros((3, 1))])
    raw = fakeRaw(rgb_xyz_matrix=np.zeros((4, 3)), color_matrix=colorMatrix, camera_whitebalance=[1.8, 1.0, 1.4, 0.0])
    colorSpace = ColorSpace.fromRAW(raw)
    assert colorSpace.white == pytest.approx(ColorSpace.D65, abs=1e-6)
    assert np.allclose(colorSpace.primaries, ColorSpace.SRGB_PRIMARIES, atol=1e-6)


def test_fromRAW_without_matrix() -> None:
    """Without a camera matrix the data is described as linear with sRGB primaries."""
    colorSpace = ColorSpace.fromRAW(fakeRaw(rgb_xyz_matrix=np.zeros((4, 3))))
    assert (colorSpace.name, colorSpace.transfer, colorSpace.matrix) == ("Camera RGB", ColorSpace.TRANSFER_LINEAR, None)
    assert colorSpace.isSRGB()


def test_displayEncode() -> None:
    """Display encoding converts to sRGB primaries, clips each channel to 0-1 and applies the sRGB curve."""
    img = np.array([[[0.214041, 0.5, 1.5], [-0.2, 0, 1]]], np.float32)
    encoded = ColorSpace.displayEncode(img, ColorSpace.SRGB)
    assert encoded.dtype == np.float32
    assert encoded[0, 0] == pytest.approx([0.5, colour.models.eotf_inverse_sRGB(0.5), 1], abs=1e-5)
    assert encoded[0, 1] == pytest.approx([0, 0, 1], abs=1e-6)
    assert np.array_equal(ColorSpace.displayEncode(img, None), encoded)


def test_displayEncode_converts_primaries() -> None:
    """White stays white, and a saturated P3 red is out of the sRGB gamut and clipped."""
    p3 = p3ColorSpace()
    white = ColorSpace.whiteRGB(p3)[None, None, :]
    assert ColorSpace.displayEncode(white, p3)[0, 0] == pytest.approx([1, 1, 1], abs=1e-3)
    red = ColorSpace.displayEncode(np.array([[[1, 0, 0]]], np.float32), p3)[0, 0]
    assert red[0] == 1
    assert red[1] == 0
    assert ColorSpace.displayImage(np.array([[[0.5, 0.5, 0.5]]], np.float32), None).dtype == np.uint8


def test_luminanceWeights_and_whiteRGB() -> None:
    """Without a description, Rec.709 weights and (1, 1, 1); otherwise from the RGB -> XYZ matrix."""
    assert ColorSpace.luminanceWeights(None) == pytest.approx(ColorSpace.REC709_LUMINANCE)
    assert ColorSpace.luminanceWeights(ColorSpace.SRGB) == pytest.approx(ColorSpace.REC709_LUMINANCE, abs=1e-4)
    assert ColorSpace.whiteRGB(None) == pytest.approx([1, 1, 1])
    p3 = p3ColorSpace()
    white = ColorSpace.whiteRGB(p3)
    assert white.dot(ColorSpace.luminanceWeights(p3)) == pytest.approx(1)
    assert white == pytest.approx([1, 1, 1], abs=1e-3)


def test_same() -> None:
    """Descriptions are the same when primaries, white, transfer and parameters match."""
    assert ColorSpace.same(None, None)
    assert not ColorSpace.same(ColorSpace.SRGB, None)
    assert ColorSpace.same(ColorSpace.SRGB, ColorSpace.named("ITU-R BT.709", ColorSpace.TRANSFER_SRGB))
    assert not ColorSpace.same(ColorSpace.SRGB, ColorSpace.named("ITU-R BT.709", ColorSpace.TRANSFER_BT709))
    assert not ColorSpace.same(ColorSpace.SRGB, p3ColorSpace())


def test_pickling_drops_cache() -> None:
    """The cached colour objects aren't pickled (they're rebuilt in the pool processes)."""
    import pickle
    colorSpace = p3ColorSpace()
    colorSpace.matrixToSRGB()
    copy = pickle.loads(pickle.dumps(colorSpace))
    assert copy.cache == {}
    assert np.allclose(copy.matrixToSRGB(), colorSpace.matrixToSRGB())


def detectImage(file: str) -> Object:
    """Detects the color mode and color description of an image file the way opening an image does.

    Args:
        file: Path of the image.

    Returns:
        Globals with ``guessedColorMode``, ``colorSpace`` and ``colorSpaceFromImage`` set and no color mode chosen.
    """
    detected = Object()
    detected.colorMode = Video.COLOR_AUTO
    detected.guessedColorMode = Video().guessColorMode(file)
    img, detected.colorSpace = Video.decodeImage(file)
    detected.colorSpaceFromImage = True
    return detected


@pytest.mark.parametrize("name", ["test.jpg", "16bit.png", "dso.jpg"])
def test_image_files_are_srgb(fakeUI: FakeUI, testInputs: str, name: str) -> None:
    """Color and grayscale image files (untagged or with an sRGB profile) are sRGB."""
    detected = detectImage(os.path.join(testInputs, name))
    expectedMode = Video.COLOR_GRAYSCALE if name == "dso.jpg" else Video.COLOR_RGB
    colorSpace = detected.actualColorSpace()
    assert detected.actualColor() == expectedMode
    assert (colorSpace.name, colorSpace.transfer) == ("sRGB", ColorSpace.TRANSFER_SRGB)
    assert ColorSpace.same(colorSpace, ColorSpace.SRGB)


def test_display_p3_image(fakeUI: FakeUI, generated: Callable[[str, str], str]) -> None:
    """A PNG with a Display P3 matrix profile is described by it."""
    colorSpace = detectImage(generated("Display P3 PNG", "p3.png")).actualColorSpace()
    assert (colorSpace.name, colorSpace.transfer) == ("Display P3 test", ColorSpace.TRANSFER_ICC)
    assert np.allclose(colorSpace.primaries, colour.RGB_COLOURSPACES["Display P3"].primaries, atol=1e-3)


def test_lut_profile_image(fakeUI: FakeUI, generated: Callable[[str, str], str]) -> None:
    """A LUT based profile is converted to Rec.2020 (LUT_GAMMA encoded) and keeps its ICC bytes."""
    path = generated("LUT profile PNG", "lut.png")
    img, colorSpace = ColorSpace.describeImage(path, cv2.imread(path))
    assert (colorSpace.name, colorSpace.transfer) == ("ITU-R BT.2020", ColorSpace.TRANSFER_GAMMA)
    assert colorSpace.icc == lutSRGBProfile()
    rec2020 = colorSpace.linearizeInteger(img, bgr=True)
    srgb = ColorSpace.SRGB.linearizeInteger(cv2.imread(path))
    converted = (rec2020.reshape(-1, 3)[:, ::-1] @ colorSpace.matrixToSRGB().T).reshape(rec2020.shape)
    assert converted[:, :, ::-1] == pytest.approx(srgb, abs=0.03)


def test_pq_png(fakeUI: FakeUI, generated: Callable[[str, str], str]) -> None:
    """The cICP chunk describes the PNG; PQ 0.58 linearizes to about 1.0 (the reference white)."""
    path = generated("cICP PQ PNG", "pq.png")
    detected = detectImage(path)
    colorSpace = detected.actualColorSpace()
    assert (colorSpace.name, colorSpace.transfer) == ("ITU-R BT.2020", ColorSpace.TRANSFER_PQ)
    img = Video().getFrame(file=path, frame=path, colorMode=detected.actualColor())
    assert img[0, 0][::-1] == pytest.approx([0.9934]*3, abs=0.002)


def test_raw_image(generated: Callable[[str, str], str]) -> None:
    """A DNG is decoded linear in camera RGB with the white point of its as-shot neutral."""
    img, colorSpace = Video.decodeImage(generated("DNG", "raw.dng"))
    assert (colorSpace.name, colorSpace.transfer) == ("Camera RGB", ColorSpace.TRANSFER_LINEAR)
    assert colorSpace.white == pytest.approx((0.3262, 0.3498), abs=1e-3)


def test_video_color_tags(progress: object, generated: Callable[[str, str], str]) -> None:
    """A video's CICP tags describe it, and its frames are linearized with them."""
    path = generated("Rec.2020 video", "rec2020.mp4")
    video = Video()
    video.openVideo(path)
    times = video.videoFrameTimes()
    colorSpace = video.videoColorSpace
    assert (colorSpace.name, colorSpace.transfer) == ("ITU-R BT.2020", ColorSpace.TRANSFER_BT709)
    (frames, sharps, deltas) = loadFramesVideo(
        file=path,
        times=times,
        previous=None,
        colorMode=Video.COLOR_RGB,
        progress=progress,
    )
    assert len(frames) == 10
    img = Video().getFrame(file=path, frame=frames[0], colorMode=Video.COLOR_RGB)
    # swscale's YUV -> RGB is about 0.4% low
    assert img[40, 40][::-1] == pytest.approx([0.5618]*3, abs=0.015)


@pytest.mark.parametrize(
    "colorMode, fromImage, described",
    [
        (Video.COLOR_RGB, False, True),
        (Video.COLOR_GRAYSCALE, True, True),
        (Video.COLOR_GRAYSCALE, False, False),
        (Video.COLOR_RGGB, True, False),
        (Video.COLOR_BGGR_VNG, False, False),
    ],
    ids=["color", "grayscale image", "grayscale video", "Bayer image", "Bayer video"],
)
def test_actualColorSpace(colorMode: int, fromImage: bool, described: bool) -> None:
    """Only color inputs and grayscale image files keep their description; Bayer and grayscale video have none."""
    gl = Object()
    gl.colorMode = colorMode
    gl.guessedColorMode = Video.COLOR_RGB
    gl.colorSpace = ColorSpace.SRGB
    gl.colorSpaceFromImage = fromImage
    assert gl.actualColorSpace() is (ColorSpace.SRGB if described else None)


def test_fits_has_no_color_space(generated: Callable[[str, str], str]) -> None:
    """FITS images have no color description."""
    img, colorSpace = Video.decodeImage(generated("FITS", "gradient.fits"))
    assert colorSpace is None
    assert img.shape == (24, 32, 3)
