"""Tests for ColorSpace.py."""
import struct
import zlib
from types import SimpleNamespace

import colour
import imagecodecs
import numpy as np
import pytest

import ColorSpace
from inputs import INPUTS_DIR

D65_XYZ = np.array(ColorSpace.D65_XYZ)

# Measured with rawpy 0.25.1 (LibRaw 0.21.4)
PIXEL_8A_COLOR_MATRIX = np.array([[1.6143, -0.5798, -0.0345, 0.0],
                                  [-0.1447, 1.3474, -0.2027, 0.0],
                                  [0.0291, -0.8361, 1.8070, 0.0]])
A7III_RGB_XYZ_MATRIX = np.array([[0.7374, -0.2389, -0.0551],
                                 [-0.5435, 1.3162, 0.2519],
                                 [-0.1006, 0.1795, 0.6552],
                                 [0.0, 0.0, 0.0]])
A7III_RGB_CAM = np.array([[1.6354, -0.4220, -0.2134],
                          [-0.1311, 1.5427, -0.4116],
                          [0.0114, -0.4196, 1.4082]])


def p3Profile() -> bytes:
    """Returns the color inputs' Display P3 profile, a matrix/TRC profile with an sRGB para curve.

    Returns:
        The profile bytes.
    """
    return (INPUTS_DIR/"color"/"DisplayP3-v4.icc").read_bytes()


def fakeRaw(**overrides: object) -> SimpleNamespace:
    """Returns a stand-in for rawpy.RawPy with a Sony A7 III's color data.

    Args:
        **overrides: Attributes to replace.

    Returns:
        The fake RawPy.
    """
    raw = SimpleNamespace(color_matrix=np.zeros((3, 4)),
                          rgb_xyz_matrix=A7III_RGB_XYZ_MATRIX,
                          black_level_per_channel=[512, 510, 514, 512],
                          white_level=16383,
                          num_colors=3,
                          camera_whitebalance=[2.21, 1.0, 1.81, 0.0])
    for name, value in overrides.items():
        setattr(raw, name, value)
    return raw


def chrm(white: tuple, primaries: tuple) -> bytes:
    """Returns a PNG cHRM chunk's data.

    Args:
        white: CIE xy of the white point.
        primaries: CIE xy of the red, green and blue primaries.

    Returns:
        The chunk data.
    """
    values = list(white) + [v for xy in primaries for v in xy]
    return struct.pack(">8I", *[int(round(v*100000)) for v in values])


SRGB_CHRM = chrm(white=(0.3127, 0.3290), primaries=((0.64, 0.33), (0.30, 0.60), (0.15, 0.06)))


def test_transfer_known_values() -> None:
    """The transfer functions give their standard values."""
    assert ColorSpace.transferToLinear(x=np.array(0.5), transfer=("sRGB", None)) == pytest.approx(0.2140, abs=1e-4)
    assert ColorSpace.transferToLinear(x=np.array(0.5), transfer=("Gamma", 2.2)) == pytest.approx(0.5**2.2)
    assert ColorSpace.transferToLinear(x=np.array(0.081), transfer=("BT.709", None)) == pytest.approx(0.018, abs=1e-6)
    pqWhite = colour.models.eotf_inverse_ST2084(203.0)
    assert ColorSpace.transferToLinear(x=np.array(pqWhite),
                                       transfer=("PQ", None)) == pytest.approx(1.0)
    assert ColorSpace.transferToLinear(x=np.array(0.75), transfer=("HLG", None)) == pytest.approx(1.0, abs=1e-6)
    assert ColorSpace.transferToLinear(x=np.array(1.0), transfer=("HLG", None)) == pytest.approx(3.774, abs=1e-3)
    x = np.linspace(0, 1, 5)
    assert np.array_equal(ColorSpace.transferToLinear(x=x, transfer=None), x)


@pytest.mark.parametrize("dtype", [np.uint8, np.uint16])
@pytest.mark.parametrize("transfer", [None, ("sRGB", None), ("Gamma", 2.2)])
def test_linearizeInteger_matches_formula(dtype: type, transfer: tuple | None) -> None:
    """The lookup tables give the transfer function of each code value."""
    codes = np.arange(np.iinfo(dtype).max + 1, dtype=dtype)
    expected = ColorSpace.transferToLinear(x=codes/np.iinfo(dtype).max, transfer=transfer)
    assert np.allclose(ColorSpace.linearizeInteger(img=codes, transfer=transfer), expected, atol=1e-6)


def test_linearizeInteger_icc() -> None:
    """ICC curves apply per channel, to RGB or BGR images."""
    img = np.array([[[0, 128, 255]]], dtype=np.uint8)
    expected = colour.models.eotf_sRGB(img[0, 0]/255)
    rgb = ColorSpace.linearizeInteger(img=img, transfer=("ICC", p3Profile()))[0, 0]
    bgr = ColorSpace.linearizeInteger(img=img[..., ::-1], transfer=("ICC", p3Profile()), bgr=True)[0, 0]
    assert np.allclose(rgb, expected, atol=1e-4)
    assert np.allclose(bgr, expected[::-1], atol=1e-4)


@pytest.mark.parametrize("workingSpace",
                         [ColorSpace.SRGB,
                          ColorSpace.named("Display P3"),
                          ColorSpace.named("ITU-R BT.2020"),
                          ColorSpace.named("ITU-R BT.470 - 525"),
                          ColorSpace.named("DCI-P3"),
                          ColorSpace.fromICC(p3Profile())[1],
                          ColorSpace.fromPNGChunks(chunks=[(b"cHRM",
                                                            chrm(white=(0.3457, 0.3585),
                                                                 primaries=((0.64, 0.33), (0.30, 0.60), (0.15, 0.06))))],
                                                   channels=3)[1],
                          ColorSpace.fromRAW(fakeRaw())[1],
                          ColorSpace.fromRAW(fakeRaw(rgb_xyz_matrix=np.zeros((4, 3)),
                                                     color_matrix=PIXEL_8A_COLOR_MATRIX))[1]],
                         ids=["sRGB", "P3", "Rec.2020", "BT.470 M (white C)", "DCI-P3 (DCI white)", "P3 ICC (D50)",
                              "cHRM D50 white", "RAW rgb_xyz_matrix", "RAW color_matrix"])
def test_toXYZ_maps_white_to_d65(workingSpace: ColorSpace.WorkingSpace) -> None:
    """Every working space maps (1, 1, 1) to D65 white."""
    assert np.allclose(workingSpace.toXYZ @ np.ones(3), D65_XYZ, atol=1e-6)


def test_named_values() -> None:
    """Named working spaces have the expected matrices."""
    assert np.allclose(ColorSpace.named("Display P3").toXYZ,
                       [[0.4866, 0.2657, 0.1982], [0.2290, 0.6917, 0.0793], [0.0000, 0.0451, 1.0439]],
                       atol=1e-4)
    assert np.allclose(ColorSpace.named("ITU-R BT.2020").toXYZ,
                       [[0.6370, 0.1446, 0.1689], [0.2627, 0.6780, 0.0593], [0.0000, 0.0281, 1.0610]],
                       atol=1e-4)
    assert ColorSpace.named("ITU-R BT.709") is ColorSpace.SRGB


def test_icc_srgb_profile_is_srgb() -> None:
    """LittleCMS's sRGB profile reads as the sRGB curve and primaries."""
    params, workingSpace = ColorSpace.fromICC(ColorSpace.srgbProfile())
    assert params == ColorSpace.DecodeParams(transfer=("sRGB", None))
    assert workingSpace == ColorSpace.SRGB


def test_icc_p3_profile() -> None:
    """A D50-relative matrix/TRC profile reads as its D65 primaries, and its curves as the sRGB curve they match."""
    params, workingSpace = ColorSpace.fromICC(p3Profile())
    assert params.transfer == ("sRGB", None)
    assert np.allclose(workingSpace.toXYZ, ColorSpace.named("Display P3").toXYZ, atol=2e-4)


def test_icc_profile_kinds() -> None:
    """Profiles are told apart by their tags."""
    assert ColorSpace.isMatrixProfile(p3Profile())
    assert not ColorSpace.isGrayProfile(p3Profile())


def test_gray_icc() -> None:
    """A gray profile gives its curve and no primaries."""
    icc = imagecodecs.cms_profile("gray", gamma=563/256)
    params, workingSpace = ColorSpace.fromGrayICC(icc)
    assert params.transfer == ("ICC", icc)
    assert workingSpace.toXYZ is None
    x = np.linspace(0, 1, 5)
    assert np.allclose(ColorSpace.transferToLinear(x=x, transfer=params.transfer), x**(563/256), atol=1e-6)


def test_displayEncode_srgb_and_untagged_apply_only_the_curve() -> None:
    """sRGB primaries and no primaries both get the sRGB curve alone."""
    rng = np.random.default_rng(0)
    img = rng.random((8, 8, 3)).astype(np.float32)
    expected = colour.models.eotf_inverse_sRGB(img)
    assert ColorSpace.toSRGBMatrix(ColorSpace.SRGB) is None
    assert ColorSpace.toSRGBMatrix(ColorSpace.UNTAGGED) is None
    assert ColorSpace.toSRGBMatrix(None) is None
    assert np.allclose(ColorSpace.displayEncode(img=img, workingSpace=ColorSpace.SRGB), expected, atol=1e-6)
    assert np.allclose(ColorSpace.displayEncode(img=img, workingSpace=ColorSpace.UNTAGGED), expected, atol=1e-6)


def test_displayEncode_converts_primaries_and_clips() -> None:
    """Wider primaries are converted to sRGB; saturated colors clip, white stays white."""
    p3 = ColorSpace.named("Display P3")
    img = np.array([[[1.0, 0.0, 0.0], [1.0, 1.0, 1.0], [0.5, 0.5, 0.5]]], dtype=np.float32)
    encoded = ColorSpace.displayEncode(img=img, workingSpace=p3)
    assert encoded[0, 0, 0] == 1.0
    assert encoded[0, 0, 1] == 0.0
    assert np.allclose(encoded[0, 1], 1.0, atol=1e-5)
    assert np.allclose(encoded[0, 2], colour.models.eotf_inverse_sRGB(0.5), atol=1e-5)
    assert ColorSpace.displayImage(img=img, workingSpace=p3).dtype == np.uint8
    assert ColorSpace.displayImage(img=img, workingSpace=p3, bits=16).dtype == np.uint16


@pytest.mark.parametrize("code, expected",
                         [(1, ("BT.709", None)),
                          (6, ("BT.601", None)),
                          (14, ("BT.2020", None)),
                          (15, ("BT.2020", None)),
                          (13, ("sRGB", None)),
                          (18, ("HLG", None)),
                          (16, ("PQ", None)),
                          (8, None),
                          (2, None),
                          (None, None),
                          (0, None),
                          (99, None)])
def test_cicpTransfer(code: int | None, expected: tuple | None) -> None:
    """Every supported transfer tag decodes as tagged; linear, unspecified, missing or unknown give no curve."""
    assert ColorSpace.cicpTransfer(code) == expected


@pytest.mark.parametrize("code", [4, 5, 7, 9, 10, 11, 12, 17])
def test_cicpTransfer_unsupported_warns(code: int) -> None:
    """A recognized but unsupported transfer warns and is treated as untagged."""
    with pytest.warns(ColorSpace.ColorWarning):
        assert ColorSpace.cicpTransfer(code) is None


def test_video_tags_untagged_yuv() -> None:
    """Untagged YUV uses the BT.709 matrix, limited range, no curve and no primaries."""
    params, workingSpace = ColorSpace.fromVideoTags(primaries=2, transfer=2, matrix=2, colorRange=0, isYUV=True)
    assert params == ColorSpace.DecodeParams(yuvMatrix="BT.709", range="limited")
    assert workingSpace.toXYZ is None


def test_video_tags_jupiter() -> None:
    """Jupiter.mp4 (yuvj420p, only its range tagged, as full) uses BT.709 and full range."""
    params, workingSpace = ColorSpace.fromVideoTags(primaries=2,
                                                    transfer=2,
                                                    matrix=2,
                                                    colorRange=2,
                                                    isYUV=True,
                                                    fullRangeFormat=True)
    assert params == ColorSpace.DecodeParams(yuvMatrix="BT.709", range="full")
    assert workingSpace.toXYZ is None


@pytest.mark.parametrize("primaries, transfer, matrix, expectedParams, expectedSpace",
                         [(1, 1, 1, ColorSpace.DecodeParams(transfer=("BT.709", None), yuvMatrix="BT.709", range="limited"),
                           ColorSpace.SRGB),
                          (6, 6, 6, ColorSpace.DecodeParams(transfer=("BT.601", None), yuvMatrix="BT.601", range="limited"),
                           ColorSpace.named("SMPTE C")),
                          (1, 13, 1, ColorSpace.DecodeParams(transfer=("sRGB", None), yuvMatrix="BT.709", range="limited"),
                           ColorSpace.SRGB),
                          (1, 2, 1, ColorSpace.DecodeParams(yuvMatrix="BT.709", range="limited"), ColorSpace.SRGB),
                          (2, 1, 2, ColorSpace.DecodeParams(transfer=("BT.709", None), yuvMatrix="BT.709", range="limited"),
                           ColorSpace.UNTAGGED),
                          (9, 16, 9, ColorSpace.DecodeParams(transfer=("PQ", None),
                                                             yuvMatrix="BT.2020",
                                                             range="limited"),
                           ColorSpace.named("ITU-R BT.2020"))])
def test_video_tags_fall_back_independently(primaries: int,
                                            transfer: int,
                                            matrix: int,
                                            expectedParams: ColorSpace.DecodeParams,
                                            expectedSpace: ColorSpace.WorkingSpace) -> None:
    """Each tag is used as tagged, and each missing tag falls back on its own."""
    params, workingSpace = ColorSpace.fromVideoTags(primaries=primaries,
                                                    transfer=transfer,
                                                    matrix=matrix,
                                                    colorRange=None,
                                                    isYUV=True)
    assert params == expectedParams
    assert workingSpace == expectedSpace


def test_video_tags_rgb_and_gray() -> None:
    """Untagged RGB/gray streams have no matrix and are full range; tagged limited range is kept."""
    params, workingSpace = ColorSpace.fromVideoTags(primaries=None, transfer=None, matrix=0, colorRange=0, isYUV=False)
    assert params == ColorSpace.DecodeParams(range="full")
    assert workingSpace.toXYZ is None
    params, workingSpace = ColorSpace.fromVideoTags(primaries=None, transfer=None, matrix=0, colorRange=1, isYUV=False)
    assert params.range == "limited"


@pytest.mark.parametrize("chunks, expectedTransfer, expectedSpace",
                         [([(b"cICP", bytes([1, 13, 0, 1])), (b"iCCP", b"x\0\0"), (b"sRGB", b"\0")],
                           ("sRGB", None), ColorSpace.SRGB),
                          ([(b"cICP", bytes([9, 16, 0, 1])), (b"sRGB", b"\0")], ("PQ", None),
                           ColorSpace.named("ITU-R BT.2020")),
                          ([(b"sRGB", b"\0"), (b"gAMA", struct.pack(">I", 100000))], ("sRGB", None), ColorSpace.SRGB),
                          ([(b"gAMA", struct.pack(">I", 45455))], ("sRGB", None), ColorSpace.SRGB),
                          ([(b"gAMA", struct.pack(">I", 45455)), (b"cHRM", SRGB_CHRM)], ("sRGB", None),
                           ColorSpace.SRGB),
                          ([(b"gAMA", struct.pack(">I", 50000))], ("Gamma", 2.0), ColorSpace.UNTAGGED),
                          ([(b"gAMA", struct.pack(">I", 100000))], None, ColorSpace.UNTAGGED),
                          ([(b"gAMA", struct.pack(">I", 100000)), (b"cHRM", SRGB_CHRM)], None, ColorSpace.SRGB)],
                         ids=["cICP over iCCP and sRGB", "cICP PQ", "sRGB over gAMA", "gAMA 45455",
                              "gAMA 45455 with sRGB cHRM", "gAMA 50000", "gAMA 100000", "gAMA 100000 with sRGB cHRM"])
def test_png_chunk_precedence(chunks: list,
                              expectedTransfer: tuple | None,
                              expectedSpace: ColorSpace.WorkingSpace) -> None:
    """PNG chunks follow cICP > iCCP > sRGB > cHRM/gAMA."""
    params, workingSpace = ColorSpace.fromPNGChunks(chunks=chunks, channels=3)
    assert params.transfer == expectedTransfer
    assert workingSpace == expectedSpace


def test_png_iccp_over_srgb() -> None:
    """An iCCP chunk wins over an sRGB chunk."""
    iccp = b"P3\0\0" + zlib.compress(p3Profile())
    params, workingSpace = ColorSpace.fromPNGChunks(chunks=[(b"sRGB", b"\0"), (b"iCCP", iccp)], channels=3)
    assert np.allclose(workingSpace.toXYZ, ColorSpace.named("Display P3").toXYZ, atol=2e-4)


def test_png_untagged() -> None:
    """A PNG without color chunks is untagged."""
    assert ColorSpace.fromPNGChunks(chunks=[(b"IHDR", bytes(13)), (b"tEXt", b"a\0b")], channels=3) is None


def test_png_gama_100000_equals_untagged() -> None:
    """gAMA 100000 decodes exactly like an untagged PNG."""
    params, workingSpace = ColorSpace.fromPNGChunks(chunks=[(b"gAMA", struct.pack(">I", 100000))], channels=3)
    assert params == ColorSpace.DecodeParams()
    assert workingSpace == ColorSpace.UNTAGGED
    img = np.arange(0, 65536, 257, dtype=np.uint16).reshape(16, 16)
    img = np.dstack([img]*3)
    assert np.array_equal(ColorSpace.linearize(img=img, params=params),
                          ColorSpace.linearize(img=img, params=ColorSpace.DecodeParams()))


def test_linearize_sensor_data_keeps_black() -> None:
    """Sensor data is scaled by its white level and keeps its black level, which Process subtracts."""
    img = np.array([[[1000, 2000, 16383]]], dtype=np.uint16)
    params = ColorSpace.DecodeParams(blackLevel=(1000.0, 1008.0, 1016.0), whiteLevel=16383.0)
    linear = ColorSpace.linearize(img=img, params=params)
    assert linear.dtype == np.float32
    assert linear[0, 0] == pytest.approx([1000/16383, 2000/16383, 1.0])


@pytest.mark.parametrize("header, dtype, expectedWhite",
                         [({"DATAMAX": 60000}, np.uint16, 60000.0),
                          ({"SATURATE": 4095}, np.uint16, 4095.0),
                          ({"DATAMAX": 50000, "SATURATE": 60000}, np.uint16, 50000.0),
                          ({}, np.uint16, 65535.0),
                          ({}, np.uint8, 255.0),
                          ({}, np.int32, 2147483647.0)])
def test_fits_levels(header: dict, dtype: type, expectedWhite: float) -> None:
    """FITS white comes from DATAMAX or SATURATE when present, else the integer type's maximum; no black is stated."""
    params = ColorSpace.fromFITS(header=header, data=np.zeros((4, 4), dtype=dtype))
    assert params.blackLevel is None
    assert params.whiteLevel == expectedWhite
    assert params.bayerPattern is None


def test_fits_float_levels() -> None:
    """Float FITS without levels keywords: white 1.0 if the data fits in 0-1, else 65535."""
    assert ColorSpace.fromFITS(header={}, data=np.array([[0.0, 0.5, 1.0]])).whiteLevel == 1.0
    assert ColorSpace.fromFITS(header={}, data=np.array([[0.0, 2.0, 4000.0]])).whiteLevel == 65535.0


def test_fits_levels_dont_depend_on_brightness() -> None:
    """Two frames of different brightness get the same scale (no per-frame min/max)."""
    header = {"SATURATE": 60000}
    bright = np.array([[2000, 50000]], dtype=np.uint16)
    dim = np.array([[2000, 30000]], dtype=np.uint16)
    brightParams = ColorSpace.fromFITS(header=header, data=bright)
    dimParams = ColorSpace.fromFITS(header=header, data=dim)
    assert brightParams == dimParams
    assert ColorSpace.linearize(img=dim, params=dimParams)[0, 1] == pytest.approx(0.5)


@pytest.mark.parametrize("header, height, expected",
                         [({"BAYERPAT": "RGGB"}, 4, "RGGB"),
                          ({"BAYERPAT": "rggb "}, 4, "RGGB"),
                          ({"BAYERPAT": "RGGB", "YBAYROFF": 1}, 4, "GBRG"),
                          ({"BAYERPAT": "RGGB", "XBAYROFF": 1}, 4, "GRBG"),
                          ({"BAYERPAT": "RGGB", "XBAYROFF": 1, "YBAYROFF": 1}, 4, "BGGR"),
                          ({"BAYERPAT": "GRBG", "YBAYROFF": 2}, 4, "GRBG"),
                          ({"BAYERPAT": "RGGB", "ROWORDER": "BOTTOM-UP"}, 4, "GBRG"),
                          ({"BAYERPAT": "RGGB", "ROWORDER": "BOTTOM-UP"}, 5, "RGGB"),
                          ({"BAYERPAT": "RGGB", "ROWORDER": "TOP-DOWN"}, 4, "RGGB"),
                          ({"BAYERPAT": "CMYG"}, 4, None)])
def test_fits_bayer_pattern(header: dict, height: int, expected: str | None) -> None:
    """BAYERPAT is normalized to the stored data's first pixel with its offsets and row order."""
    params = ColorSpace.fromFITS(header=header, data=np.zeros((height, 4), dtype=np.uint16))
    assert params.bayerPattern == expected


def test_fromRAW_levels() -> None:
    """RAW levels come from black_level_per_channel (greens averaged) and white_level."""
    params, workingSpace = ColorSpace.fromRAW(fakeRaw())
    assert params == ColorSpace.DecodeParams(blackLevel=(512.0, 511.0, 514.0), whiteLevel=16383.0)


def test_fromRAW_rgb_xyz_matrix() -> None:
    """A per-model table matrix (A7 III) gives the expected rgbCam."""
    params, workingSpace = ColorSpace.fromRAW(fakeRaw())
    rgbCam = np.linalg.solve(ColorSpace.SRGB.toXYZ, workingSpace.toXYZ)
    assert np.allclose(rgbCam, A7III_RGB_CAM, atol=5e-4)
    assert np.allclose(rgbCam.sum(axis=1), 1)


def test_fromRAW_color_matrix() -> None:
    """A DNG's color_matrix (Pixel 8a) is used as rgbCam."""
    raw = fakeRaw(rgb_xyz_matrix=np.zeros((4, 3)), color_matrix=PIXEL_8A_COLOR_MATRIX)
    params, workingSpace = ColorSpace.fromRAW(raw)
    assert np.allclose(workingSpace.toXYZ, ColorSpace.SRGB.toXYZ @ PIXEL_8A_COLOR_MATRIX[:, :3])


def test_fromRAW_without_matrix() -> None:
    """A camera LibRaw doesn't know gets no primaries and a warning."""
    with pytest.warns(ColorSpace.ColorWarning):
        params, workingSpace = ColorSpace.fromRAW(fakeRaw(rgb_xyz_matrix=np.zeros((4, 3))))
    assert workingSpace.toXYZ is None
    assert params.whiteLevel == 16383.0


def test_working_space_equality() -> None:
    """Working spaces compare by matrix, within tolerance, and untagged only equals untagged."""
    assert ColorSpace.named("Display P3") == ColorSpace.named("Display P3")
    assert ColorSpace.named("Display P3") != ColorSpace.SRGB
    assert ColorSpace.UNTAGGED == ColorSpace.WorkingSpace(toXYZ=None)
    assert ColorSpace.UNTAGGED != ColorSpace.SRGB


def test_srgb_png_chunks() -> None:
    """Saved PNGs are tagged with cICP 1/13/0/1, an sRGB iCCP, sRGB 0 and gAMA 45455."""
    chunks = dict(ColorSpace.srgbPNGChunks())
    assert chunks[b"cICP"] == bytes([1, 13, 0, 1])
    assert chunks[b"sRGB"] == b"\0"
    assert chunks[b"gAMA"] == struct.pack(">I", 45455)
    assert chunks[b"iCCP"].startswith(b"sRGB\0\0")
    params, workingSpace = ColorSpace.fromPNGChunks(chunks=ColorSpace.srgbPNGChunks(), channels=3)
    assert params.transfer == ("sRGB", None)
    assert workingSpace == ColorSpace.SRGB


@pytest.mark.parametrize("bits, colorRange, black, white",
                         [(8, "limited", 16, 235),
                          (8, "full", 0, 255),
                          (10, "limited", 64, 940),
                          (10, "full", 0, 1023)])
def test_yuvToRGB_range(bits: int, colorRange: str, black: int, white: int) -> None:
    """Black and white with neutral chroma decode to 0 and 1, in limited and full range."""
    neutral = 2**(bits - 1) if colorRange == "full" else 128*2**(bits - 8)
    for matrix in ("BT.601", "BT.709", "BT.2020"):
        yuv = np.array([[[black, white]], [[neutral, neutral]], [[neutral, neutral]]])
        rgb = ColorSpace.yuvToRGB(yuv=yuv, bits=bits, matrix=matrix, colorRange=colorRange)
        assert np.allclose(rgb[0, 0], 0, atol=1e-6)
        assert np.allclose(rgb[0, 1], 1, atol=2e-3 if colorRange == "full" else 1e-6)


def test_yuvToRGB_matrix() -> None:
    """The matrix gives the primaries' luminance weights back: pure red's Y is Kr."""
    for matrix, (kr, kb) in ColorSpace.YUV_WEIGHTS.items():
        y = kr
        cb = (0 - y)/(2*(1 - kb))
        cr = (1 - y)/(2*(1 - kr))
        yuv = np.array([[[16 + 219*y]], [[128 + 224*cb]], [[128 + 224*cr]]])
        rgb = ColorSpace.yuvToRGB(yuv=yuv, bits=8, matrix=matrix, colorRange="limited")
        assert np.allclose(rgb[0, 0], [1, 0, 0], atol=1e-5)
