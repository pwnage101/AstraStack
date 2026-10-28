"""Tests for the pure functions in Process.py."""
import math

import numpy as np
import pytest

import ColorSpace
import Export
from Globals import g
from Process import (
    STRETCH_CURVES,
    Process,
    applyGamma,
    applyLevels,
    autoExposure,
    blackLevelFromSlider,
    displayLuminance,
    estimateBlackLevel,
    saturate,
    scalePixels,
    sliderFromBlackLevel,
    sqrtDecode,
    sqrtEncode,
    stretch,
)
from tests.fixtureFiles import colorTestImage, displayP3Profile

STRETCHES = {
    "Square Root": Process.STRETCH_SQRT,
    "Cube Root": Process.STRETCH_CBRT,
    "Logarithm": Process.STRETCH_LOG,
    "ArcSinH": Process.STRETCH_ASINH,
}


def test_applyLevels() -> None:
    """The black level is subtracted and the exposure applied as 2^EV, without clipping."""
    img = np.array([[[0.0, 0.5, 1.0]]], np.float32)
    leveled = applyLevels(img=img, blackLevel=0.25, exposure=1)
    assert leveled.dtype == np.float32
    assert leveled[0, 0] == pytest.approx([-0.5, 0.5, 1.5])
    assert applyLevels(img=img, blackLevel=0, exposure=0) == pytest.approx(img)


def test_estimateBlackLevel() -> None:
    """The black level is the minimum blurred lightness, limited to 0-1."""
    img = np.full((50, 60, 3), 0.2, np.float32)
    img[:, :, 0] = 0.4
    assert estimateBlackLevel(img) == pytest.approx(0.3)
    assert estimateBlackLevel(img - 1) == 0
    assert estimateBlackLevel(img + 2) == 1


def test_estimateBlackLevel_ignores_small_dark_spots() -> None:
    """A single dark pixel is blurred away, so the background sets the black level."""
    img = np.full((200, 200, 3), 0.1, np.float32)
    img[100, 100] = 0
    assert estimateBlackLevel(img) == pytest.approx(0.1, abs=0.005)


@pytest.mark.parametrize(
    "maxValue, blackLevel, expected",
    [
        (0.5, 0, 1),
        (1, 0, 0),
        (0.3, 0.05, 2),
        (4, 0, -2),
        (1e-6, 0, Process.EXPOSURE_MAX),
        (1e6, 0, Process.EXPOSURE_MIN),
        (0.1, 0.1, 0),
        (0.1, 0.2, 0),
    ],
)
def test_autoExposure(maxValue: float, blackLevel: float, expected: float) -> None:
    """The exposure puts the brightest value at 1.0 after black subtraction, within the slider range."""
    assert autoExposure(maxValue, blackLevel) == pytest.approx(expected)


@pytest.mark.parametrize(
    "value, blackLevel",
    [(-2, 0), (-3, 0), (-1, 0.001), (0, 0.01), (1, 0.1), (2, 1)],
)
def test_blackLevelFromSlider(value: float, blackLevel: float) -> None:
    """The slider is log10 of the percentage of full scale, and its far left end is 0."""
    assert blackLevelFromSlider(value) == pytest.approx(blackLevel)


@pytest.mark.parametrize("value", [-1.5, -1, 0, 0.5, 2])
def test_sliderFromBlackLevel_round_trip(value: float) -> None:
    """Converting a slider value to a black level and back gives the slider value."""
    assert sliderFromBlackLevel(blackLevelFromSlider(value)) == pytest.approx(value)


def test_sliderFromBlackLevel_limits() -> None:
    """Black levels outside the slider range are shown at its ends."""
    assert sliderFromBlackLevel(0) == Process.BLACK_LEVEL_MIN
    assert sliderFromBlackLevel(1e-6) == Process.BLACK_LEVEL_MIN
    assert sliderFromBlackLevel(5) == Process.BLACK_LEVEL_MAX


def bareProcess(**attributes: object) -> Process:
    """Returns a Process without running its constructor (which decomposes the image in the process pool).

    Args:
        **attributes: Attributes to set.

    Returns:
        The Process.
    """
    process = Process.__new__(Process)
    for name, value in attributes.items():
        setattr(process, name, value)
    return process


@pytest.mark.parametrize(
    "autoBlack, autoExp, expected",
    [
        (True, True, (0.1, 1)),
        (True, False, (0.1, -0.5)),
        (False, True, (0.01, -math.log2(0.59))),
        (False, False, (0.01, -0.5)),
    ],
)
def test_levels(
    monkeypatch: pytest.MonkeyPatch,
    autoBlack: bool,
    autoExp: bool,
    expected: tuple[float, float],
) -> None:
    """Each automatic toggle replaces its slider value with the estimate."""
    monkeypatch.setattr(g, "autoBlackLevel", autoBlack, raising=False)
    monkeypatch.setattr(g, "autoExposure", autoExp, raising=False)
    monkeypatch.setattr(g, "blackLevel", 0.01, raising=False)
    monkeypatch.setattr(g, "exposure", -0.5, raising=False)
    assert bareProcess(autoBlackLevel=0.1, maxValue=0.6).levels() == pytest.approx(expected)


@pytest.mark.parametrize(
    "img",
    [
        np.array([[[0.0, 0.5, 1.0], [-0.5, 0.25, 2.0]]], np.float32),
        np.random.default_rng(1).uniform(-0.2, 1.5, (8, 9, 3)).astype(np.float32),
    ],
    ids=["edges", "random"],
)
def test_sqrt_encoding_round_trip(img: np.ndarray) -> None:
    """The square root encoding of the wavelets round-trips, including values below 0 and above 1."""
    encoded = sqrtEncode(img)
    assert np.all(np.sign(encoded) == np.sign(img))
    assert sqrtDecode(encoded) == pytest.approx(img, abs=1e-6)


def test_scalePixels() -> None:
    """All channels of a pixel get the same ratio; NaN and infinite ratios give 0."""
    img = np.ones((1, 3, 3), np.float32)
    ratio = np.array([[2, np.nan, np.inf]], np.float32)
    assert scalePixels(img, ratio)[0].tolist() == [[2, 2, 2], [0, 0, 0], [0, 0, 0]]


def leveledTestImage(colorSpace: ColorSpace.ColorSpace = ColorSpace.SRGB) -> np.ndarray:
    """Returns the color test image linearized and leveled with the automatic black level and exposure.

    Args:
        colorSpace: Color description the 8-bit test image is linearized with.

    Returns:
        The float32 RGB image.
    """
    linear = colorSpace.linearizeInteger(colorTestImage())
    blackLevel = estimateBlackLevel(linear)
    return applyLevels(img=linear, blackLevel=blackLevel, exposure=autoExposure(float(linear.max()), blackLevel))


def assertKeepsRatios(before: np.ndarray, after: np.ndarray) -> None:
    """Checks that the R:G:B ratios of the pixels are unchanged and that the image changed.

    Args:
        before: Image before the tone step.
        after: Image after the tone step.
    """
    before = before.astype(np.float64)
    after = after.astype(np.float64)
    mask = (before.min(axis=2) > 0.01) & (before.mean(axis=2) > 0.05)
    assert mask.sum() > 100
    ratiosBefore = before[mask]/before[mask].sum(axis=1, keepdims=True)
    ratiosAfter = after[mask]/after[mask].sum(axis=1, keepdims=True)
    assert np.abs(ratiosAfter - ratiosBefore).max() < 1e-3
    assert np.abs(after - before).max() > 0.05


@pytest.mark.parametrize("name", STRETCHES)
def test_stretch_keeps_ratios(name: str) -> None:
    """Each stretch curve changes the intensity but keeps the R:G:B ratios."""
    img = leveledTestImage()
    assertKeepsRatios(before=img, after=stretch(img, STRETCHES[name]))


@pytest.mark.parametrize("name", STRETCHES)
def test_stretch_curves(name: str) -> None:
    """Curves are normalized so f(1) = 1, keep gray pixels gray and brighten the intensity below 1."""
    curve = STRETCH_CURVES[STRETCHES[name]]
    assert curve(np.float64(1)) == pytest.approx(1)
    gray = np.full((1, 1, 3), 0.25, np.float32)
    stretched = stretch(gray, STRETCHES[name])[0, 0]
    assert stretched == pytest.approx([curve(0.25)]*3, rel=1e-5)
    assert stretched[0] > 0.25


def test_stretch_limits() -> None:
    """Pixels with an intensity of 0 or less become 0, and pixels above 1.0 are scaled down as a whole."""
    img = np.array([[[0, 0, 0], [-0.1, 0.05, 0], [2, 1, 0]]], np.float32)
    stretched = stretch(img, Process.STRETCH_SQRT)
    assert stretched[0, 0].tolist() == [0, 0, 0]
    assert stretched[0, 1].tolist() == [0, 0, 0]
    assert stretched[0, 2] == pytest.approx([1, 0.5, 0])
    assert stretch(img, Process.STRETCH_NONE) is img


def test_gamma_keeps_ratios() -> None:
    """Gamma changes the intensity but keeps the R:G:B ratios."""
    img = leveledTestImage()
    assertKeepsRatios(before=img, after=applyGamma(img, 1.5))


def test_gamma_values() -> None:
    """The intensity I becomes I^(1/gamma); I <= 0 becomes 0, and gamma 1 leaves the image unchanged."""
    img = np.array([[[0.25, 0.25, 0.25], [-0.1, 0, 0]]], np.float32)
    assert applyGamma(img, 2)[0].tolist() == [[0.5, 0.5, 0.5], [0, 0, 0]]
    assert applyGamma(img, 1) is img


@pytest.mark.parametrize("profile", ["sRGB", "Display P3"])
def test_saturation_zero_is_luminance_gray(profile: str) -> None:
    """Saturation 0 makes every pixel its luminance gray Y*white, which displays and exports as gray."""
    colorSpace = ColorSpace.SRGB if profile == "sRGB" else ColorSpace.fromICC(displayP3Profile())
    img = leveledTestImage(colorSpace)
    luminance = ColorSpace.luminanceWeights(colorSpace)
    white = ColorSpace.whiteRGB(colorSpace)
    gray = saturate(img=img, saturation=0, luminance=luminance, white=white)
    assert gray == pytest.approx(img.dot(luminance)[:, :, None]*white, abs=1e-5)
    exported = Export.convert(img=gray, pixelFormat=Export.FORMAT_8BIT, colorSpace=colorSpace)
    assert np.array_equal(exported[:, :, 0], exported[:, :, 1])
    assert np.array_equal(exported[:, :, 1], exported[:, :, 2])


def test_saturation() -> None:
    """Saturation 1 leaves the image unchanged, other values mix toward or away from the luminance gray."""
    img = np.array([[[0.8, 0.2, 0.1]]], np.float32)
    luminance = ColorSpace.luminanceWeights(None)
    white = ColorSpace.whiteRGB(None)
    assert saturate(img=img, saturation=1, luminance=luminance, white=white) is img
    Y = float(img[0, 0].dot(luminance))
    half = saturate(img=img, saturation=0.5, luminance=luminance, white=white)
    assert half[0, 0] == pytest.approx((img[0, 0] + Y)/2)
    more = saturate(img=img, saturation=2, luminance=luminance, white=white)
    assert more[0, 0] == pytest.approx(2*img[0, 0] - Y)


def test_displayLuminance() -> None:
    """The dering mask's luminance is display encoded to 0-255 and clipped."""
    img = np.array([[[1, 1, 1], [0.214041, 0.214041, 0.214041], [2, 2, 2], [-1, -1, -1]]], np.float32)
    luminance = ColorSpace.luminanceWeights(None)
    assert displayLuminance(img, luminance)[0] == pytest.approx([255, 127.5, 255, 0], abs=0.01)


def test_processColor(monkeypatch: pytest.MonkeyPatch) -> None:
    """The color stage applies the gains, then Saturation, Stretch and Gamma, and records every value."""
    for name, value in (
        ("redAdjust", 110),
        ("greenAdjust", 95),
        ("blueAdjust", 105),
        ("saturation", 80),
        ("stretchFunction", Process.STRETCH_SQRT),
        ("gamma", 120),
    ):
        monkeypatch.setattr(g, name, value, raising=False)
    img = leveledTestImage()
    luminance = ColorSpace.luminanceWeights(ColorSpace.SRGB)
    white = ColorSpace.whiteRGB(ColorSpace.SRGB)
    process = bareProcess(deringedImage=img, luminance=luminance, white=white, blackLevel=0.01, exposure=1.5)
    process.processColor()
    expected = img*np.array([1.1, 0.95, 1.05], np.float32)
    expected = saturate(img=expected, saturation=0.8, luminance=luminance, white=white)
    expected = applyGamma(stretch(expected, Process.STRETCH_SQRT), 1.2)
    assert process.finalImage.dtype == np.float32
    assert process.finalImage == pytest.approx(expected, abs=1e-6)
    assert process.processingRecord == {
        "BlackLevel": "1%",
        "Exposure": "+1.500 EV",
        "RedGain": "110%",
        "GreenGain": "95%",
        "BlueGain": "105%",
        "Saturation": "80%",
        "Stretch": "Square Root",
        "Gamma": "1.2",
    }


def test_processColor_neutral_is_linear(monkeypatch: pytest.MonkeyPatch) -> None:
    """At the neutral settings the color stage keeps the data linear."""
    for name, value in (
        ("redAdjust", 100),
        ("greenAdjust", 100),
        ("blueAdjust", 100),
        ("saturation", 100),
        ("stretchFunction", Process.STRETCH_NONE),
        ("gamma", 100),
    ):
        monkeypatch.setattr(g, name, value, raising=False)
    img = leveledTestImage()
    process = bareProcess(
        deringedImage=img,
        luminance=ColorSpace.luminanceWeights(None),
        white=ColorSpace.whiteRGB(None),
        blackLevel=0,
        exposure=0,
    )
    process.processColor()
    assert np.array_equal(process.finalImage, img)
    assert process.processingRecord["Stretch"] == "None"
