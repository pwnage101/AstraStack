"""Tests for ColorSpace.py."""
import colour
import numpy as np
import pytest

import ColorSpace

D65_XYZ = np.array(ColorSpace.D65_XYZ)

SRGB_CURVE = ColorSpace.SRGB_CURVE
BT709 = ColorSpace.BT709_CURVE
BT601 = ("H273", 6)
PQ = ("H273", 16)
HLG = ("H273", 18)


def test_transfer_known_values() -> None:
    """The transfer functions give their standard values."""
    assert ColorSpace.transferToLinear(x=np.array(0.5), transfer=SRGB_CURVE) == pytest.approx(0.2140, abs=1e-4)
    assert ColorSpace.transferToLinear(x=np.array(0.5), transfer=("Gamma", 2.2)) == pytest.approx(0.5**2.2)
    assert ColorSpace.transferToLinear(x=np.array(0.081), transfer=BT709) == pytest.approx(0.018, abs=5e-6)
    pqWhite = colour.models.eotf_inverse_ST2084(203.0)
    assert ColorSpace.transferToLinear(x=np.array(pqWhite), transfer=PQ) == pytest.approx(1.0)
    assert ColorSpace.transferToLinear(x=np.array(0.75), transfer=HLG) == pytest.approx(1.0, abs=1e-6)
    assert ColorSpace.transferToLinear(x=np.array(1.0), transfer=HLG) == pytest.approx(3.774, abs=1e-3)
    x = np.linspace(0, 1, 5)
    assert np.array_equal(ColorSpace.transferToLinear(x=x, transfer=None), x)


@pytest.mark.parametrize("code, x, expected", [(5, 0.5, 0.5**2.8), (4, 0.5, 0.5**2.2), (17, 1.0, 52.37/48)])
def test_transfer_legacy_codes(code: int, x: float, expected: float) -> None:
    """Every H.273 curve is decoded, by inverting its encoding curve."""
    assert ColorSpace.transferToLinear(x=np.array(x), transfer=("H273", code)) == pytest.approx(expected, rel=1e-3)


@pytest.mark.parametrize("dtype", [np.uint8, np.uint16])
@pytest.mark.parametrize("transfer", [None, SRGB_CURVE, ("Gamma", 2.2)])
def test_linearizeInteger_matches_formula(dtype: type, transfer: tuple | None) -> None:
    """The lookup tables give the transfer function of each code value."""
    codes = np.arange(np.iinfo(dtype).max + 1, dtype=dtype)
    expected = ColorSpace.transferToLinear(x=codes/np.iinfo(dtype).max, transfer=transfer)
    assert np.allclose(ColorSpace.linearizeInteger(img=codes, transfer=transfer), expected, atol=1e-6)


@pytest.mark.parametrize("workingSpace",
                         [ColorSpace.SRGB,
                          ColorSpace.cicpPrimaries(12),
                          ColorSpace.cicpPrimaries(9),
                          ColorSpace.cicpPrimaries(4),
                          ColorSpace.cicpPrimaries(11),
                          ColorSpace.fromPrimaries(primaries=((0.64, 0.33), (0.30, 0.60), (0.15, 0.06)),
                                                   white=(0.3457, 0.3585))],
                         ids=["sRGB", "P3", "Rec.2020", "BT.470 M (white C)", "DCI-P3 (DCI white)", "D50 white"])
def test_toXYZ_maps_white_to_d65(workingSpace: ColorSpace.WorkingSpace) -> None:
    """Every working space maps (1, 1, 1) to D65 white."""
    assert np.allclose(workingSpace.toXYZ @ np.ones(3), D65_XYZ, atol=1e-6)


def test_cicpPrimaries_values() -> None:
    """CICP primaries have the expected matrices."""
    assert np.allclose(ColorSpace.cicpPrimaries(12).toXYZ,
                       [[0.4866, 0.2657, 0.1982], [0.2290, 0.6917, 0.0793], [0.0000, 0.0451, 1.0439]],
                       atol=1e-4)
    assert np.allclose(ColorSpace.cicpPrimaries(9).toXYZ,
                       [[0.6370, 0.1446, 0.1689], [0.2627, 0.6780, 0.0593], [0.0000, 0.0281, 1.0610]],
                       atol=1e-4)
    assert ColorSpace.cicpPrimaries(1) == ColorSpace.SRGB


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
    p3 = ColorSpace.cicpPrimaries(12)
    img = np.array([[[1.0, 0.0, 0.0], [1.0, 1.0, 1.0], [0.5, 0.5, 0.5]]], dtype=np.float32)
    encoded = ColorSpace.displayEncode(img=img, workingSpace=p3)
    assert encoded[0, 0, 0] == 1.0
    assert encoded[0, 0, 1] == 0.0
    assert np.allclose(encoded[0, 1], 1.0, atol=1e-5)
    assert np.allclose(encoded[0, 2], colour.models.eotf_inverse_sRGB(0.5), atol=1e-5)
    assert ColorSpace.displayImage(img=img, workingSpace=p3).dtype == np.uint8
    assert ColorSpace.displayImage(img=img, workingSpace=p3, bits=16).dtype == np.uint16


@pytest.mark.parametrize("code", [0, 3, 19, 255])
def test_cicpTransfer_reserved_warns(code: int) -> None:
    """A reserved transfer warns and is treated as untagged."""
    with pytest.warns(ColorSpace.ColorWarning):
        assert ColorSpace.cicpTransfer(code) is None


def test_working_space_equality() -> None:
    """Working spaces compare by matrix, within tolerance, and untagged only equals untagged."""
    assert ColorSpace.cicpPrimaries(12) == ColorSpace.cicpPrimaries(12)
    assert ColorSpace.cicpPrimaries(12) != ColorSpace.SRGB
    assert ColorSpace.UNTAGGED == ColorSpace.WorkingSpace(toXYZ=None)
    assert ColorSpace.UNTAGGED != ColorSpace.SRGB


def test_video_tags_untagged_yuv() -> None:
    """Untagged YUV is BT.709 (curve and primaries)."""
    curve, workingSpace = ColorSpace.fromVideoTags(primaries=2, transfer=2, isYUV=True)
    assert curve == BT709
    assert workingSpace == ColorSpace.SRGB


@pytest.mark.parametrize(
    "primaries, transfer, isYUV, expectedCurve, expectedSpace",
    [
        (6, 6, True, BT601, ColorSpace.cicpPrimaries(6)),
        (1, 13, True, SRGB_CURVE, ColorSpace.SRGB),
        (6, 2, True, BT709, ColorSpace.cicpPrimaries(6)),
        (2, 8, True, None, ColorSpace.SRGB),
        (9, 16, True, PQ, ColorSpace.cicpPrimaries(9)),
        (None, None, False, SRGB_CURVE, ColorSpace.SRGB),
    ],
)
def test_video_tags_fall_back_independently(primaries: int,
                                            transfer: int,
                                            isYUV: bool,
                                            expectedCurve: tuple | None,
                                            expectedSpace: ColorSpace.WorkingSpace) -> None:
    """Each tag is used as tagged, and each missing tag falls back on its own."""
    curve, workingSpace = ColorSpace.fromVideoTags(primaries=primaries, transfer=transfer, isYUV=isYUV)
    assert curve == expectedCurve
    assert workingSpace == expectedSpace
