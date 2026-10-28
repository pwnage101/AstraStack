"""Tests for the lens metadata reading and lensfun correction in Lens.py."""
import math
import os
import struct
from types import SimpleNamespace
from typing import Callable, Iterator

import numpy as np
import pytest

import Lens
from tests.fixtureFiles import NIKON_CROP_CENTERED, NIKON_CROP_OFF_CENTER, atom, gridTestImage, nctgAtom
from Video import Video


@pytest.fixture(autouse=True)
def emptyMaps() -> Iterator[None]:
    """Starts every test without cached correction maps."""
    Lens.MAPS.clear()
    yield
    Lens.MAPS.clear()


def canonCorrection(**overrides: object) -> Lens.LensCorrection:
    """Returns the correction settings for a Canon EOS 5D Mark II with an EF 24-105mm at 24mm f/4.

    Args:
        **overrides: Settings to replace.

    Returns:
        The settings, with distortion and vignetting on.
    """
    settings = Lens.LensCorrection()
    settings.cameraMaker = "Canon"
    settings.cameraModel = "Canon EOS 5D Mark II"
    settings.lensMaker = "Canon"
    settings.lensModel = "Canon EF 24-105mm f/4L IS USM"
    settings.focal = 24.0
    settings.aperture = 4.0
    settings.distortion = True
    settings.vignetting = True
    for name, value in overrides.items():
        setattr(settings, name, value)
    return settings


def gridFrame(w: int = 96, h: int = 64) -> np.ndarray:
    """Returns the grid test image as a linear float32 frame.

    Args:
        w: Width of the frame.
        h: Height of the frame.

    Returns:
        The BGR frame scaled to 0-1.
    """
    return gridTestImage(w=w, h=h).astype(np.float32)/255


def movWithAtoms(path: str, moovChildren: list[bytes]) -> str:
    """Writes a QuickTime file holding only an ftyp atom and a moov atom with the given children.

    Args:
        path: Path of the file.
        moovChildren: The atoms inside moov.

    Returns:
        The path.
    """
    with open(path, "wb") as f:
        f.write(atom(type=b"ftyp", data=b"qt  " + bytes(4)) + atom(type=b"moov", data=b"".join(moovChildren)))
    return path


def test_readNCTG_nikon_video(generated: Callable[[str, str], str]) -> None:
    """The camera, lens, focal length, aperture and crop region are read from a Nikon video's NCTG atom."""
    metadata = Lens.readMetadata(generated("Nikon video", "nikon.mov"))
    assert metadata == {
        "make": "NIKON CORPORATION",
        "model": "NIKON Z 5",
        "lensMake": "NIKON",
        "lens": "NIKKOR Z 24-70mm f/2.8 S",
        "focal": 24.0,
        "aperture": 2.8,
        "cropRegion": NIKON_CROP_CENTERED,
    }


def test_readNCTG_second_udta(tmp_path: str) -> None:
    """The NCTG atom is found in a later udta atom when an earlier one doesn't have it."""
    nctg = atom(type=b"NCDT", data=nctgAtom(NIKON_CROP_OFF_CENTER))
    path = movWithAtoms(
        os.path.join(tmp_path, "two.mov"),
        [atom(type=b"udta", data=atom(type=b"\xa9nam", data=b"title")), atom(type=b"udta", data=nctg)],
    )
    tags = Lens.readNCTG(path)
    assert tags[Lens.NCTG_LENS_MODEL] == "NIKKOR Z 24-70mm f/2.8 S"
    assert tags[Lens.NCTG_CROP_HISPEED] == NIKON_CROP_OFF_CENTER


def test_readNCTG_missing(tmp_path: str, testInputs: str) -> None:
    """Files without an NCTG atom (or that can't be read) give None."""
    path = movWithAtoms(os.path.join(tmp_path, "none.mov"), [atom(type=b"udta", data=b"")])
    assert Lens.readNCTG(path) is None
    assert Lens.readNCTG(os.path.join(tmp_path, "missing.mov")) is None
    assert Lens.readNCTG(os.path.join(testInputs, "test.jpg")) is None


def test_readNCTG_truncated_entry(tmp_path: str) -> None:
    """An entry whose data runs past the end of the atom ends the tag list."""
    data = struct.pack(">IHH", Lens.NCTG_MAKE, 2, 6) + b"NIKON\0" + struct.pack(">IHH", Lens.NCTG_MODEL, 2, 50) + b"Z"
    path = movWithAtoms(
        os.path.join(tmp_path, "short.mov"),
        [atom(type=b"udta", data=atom(type=b"NCDT", data=atom(type=b"NCTG", data=data)))],
    )
    assert Lens.readNCTG(path) == {Lens.NCTG_MAKE: "NIKON"}


@pytest.mark.parametrize(
    "data, type, count, expected",
    [
        (b"NIKON\0junk", 2, 10, "NIKON"),
        (struct.pack(">H", 7), 3, 1, 7),
        (struct.pack(">2H", 1, 2), 3, 2, (1, 2)),
        (struct.pack(">h", -3), 8, 1, -3),
        (struct.pack(">I", 70000), 4, 1, 70000),
        (struct.pack(">i", -70000), 9, 1, -70000),
        (struct.pack(">II", 240, 10), 5, 1, 24.0),
        (struct.pack(">ii", -1, 4), 10, 1, -0.25),
        (struct.pack(">II", 1, 0), 5, 1, 0.0),
        (b"\x01\x02", 7, 2, b"\x01\x02"),
    ],
)
def test_decodeValue(data: bytes, type: int, count: int, expected: object) -> None:
    """NCTG/EXIF values are decoded by type: strings, integers, rationals, raw bytes otherwise."""
    assert Lens.decodeValue(data, type, count) == expected


def test_readMetadata_exif(generated: Callable[[str, str], str]) -> None:
    """The camera and lens are read from a JPEG's EXIF."""
    metadata = Lens.readMetadata([generated("lens EXIF JPEG", "lens.jpg")])
    assert metadata == {
        "make": "Canon",
        "model": "Canon EOS 5D Mark II",
        "lens": "Canon EF 24-105mm f/4L IS USM",
        "focal": 24.0,
        "aperture": 4.0,
    }


def test_readMetadata_without_lens(testInputs: str) -> None:
    """An image without lens metadata gives no lens, focal length or aperture."""
    metadata = Lens.readMetadata(os.path.join(testInputs, "dso.jpg"))
    assert not {"lens", "focal", "aperture", "cropRegion"} & set(metadata)


def test_find_camera_and_lens() -> None:
    """Cameras and lenses are found by their EXIF/NCTG names; unknown ones give None."""
    camera = Lens.findCamera("NIKON CORPORATION", "NIKON Z 5")
    lens = Lens.findLens(camera, "NIKON", "NIKKOR Z 24-70mm f/2.8 S")
    assert Lens.displayName(camera) == "Nikon Z 5"
    assert lens.model == "NIKKOR Z 24-70mm f/2.8 S"
    assert Lens.findCamera("Foo", "Bar") is None
    assert Lens.findCamera(None, "NIKON Z 5") is None
    assert Lens.findLens(None, None, "nope") is None
    assert Lens.findLens(None, None, None) is None


def test_findLens_without_camera() -> None:
    """Without a camera every lens is searched, ignoring case."""
    lens = Lens.findLens(None, None, " canon ef 24-105mm f/4l is usm ")
    assert lens.model == "Canon EF 24-105mm f/4L IS USM"
    assert Lens.sameItem(lens, Lens.findLens(None, "Canon", "Canon EF 24-105mm f/4L IS USM"))
    assert Lens.sameItem(None, None)
    assert not Lens.sameItem(lens, None)


def test_imageCropFactor_nikon_centered() -> None:
    """A centered Nikon crop scales the camera's crop factor by the diagonal ratio."""
    camera = Lens.findCamera("NIKON CORPORATION", "NIKON Z 5")
    lens = Lens.findLens(camera, "NIKON", "NIKKOR Z 24-70mm f/2.8 S")
    crop, geometry = Lens.imageCropFactor(lens, camera, NIKON_CROP_CENTERED, 96, 64)
    assert crop == pytest.approx(1.0469, abs=1e-4)
    assert geometry


def test_imageCropFactor_nikon_off_center() -> None:
    """An off-center Nikon crop can't have its geometry corrected."""
    camera = Lens.findCamera("NIKON CORPORATION", "NIKON Z 5")
    lens = Lens.findLens(camera, "NIKON", "NIKKOR Z 24-70mm f/2.8 S")
    crop, geometry = Lens.imageCropFactor(lens, camera, NIKON_CROP_OFF_CENTER, 96, 64)
    assert crop == pytest.approx(1.0469, abs=1e-4)
    assert not geometry


@pytest.mark.parametrize(
    "camera, size, expected",
    [
        (SimpleNamespace(crop_factor=1.5, mount="Nikon F"), (96, 64), 1.5),
        (SimpleNamespace(crop_factor=1.5, mount="Nikon F"), (64, 96), 1.5),
        (SimpleNamespace(crop_factor=1.5, mount="Nikon F"), (1920, 1080), 1.5*math.hypot(1, 2/3)/math.hypot(1, 9/16)),
        (SimpleNamespace(crop_factor=2.0, mount="Micro 4/3 System"), (400, 300), 2.0),
        (SimpleNamespace(crop_factor=2.0, mount="Micro 4/3 System"), (300, 300), 2.0*math.hypot(1, 4/3)/math.sqrt(2)),
        (None, (96, 64), 1.6),
    ],
    ids=["3:2", "portrait", "16:9", "4:3 mount", "square", "unknown camera"],
)
def test_imageCropFactor_without_crop_region(
    camera: SimpleNamespace | None,
    size: tuple[int, int],
    expected: float,
) -> None:
    """Without a crop region the image spans the sensor's width (or height); an unknown camera uses the lens's."""
    lens = SimpleNamespace(crop_factor=1.6)
    crop, geometry = Lens.imageCropFactor(lens, camera, None, *size)
    assert crop == pytest.approx(expected)
    assert geometry


def test_LensCorrection_enabled_and_key() -> None:
    """Correction is enabled with a lens, focal length, aperture and at least one correction."""
    settings = canonCorrection()
    assert settings.enabled()
    assert settings.key()[3:6] == ("Canon EF 24-105mm f/4L IS USM", 24.0, 4.0)
    assert not canonCorrection(distortion=False, vignetting=False).enabled()
    assert not canonCorrection(focal=0.0).enabled()
    assert canonCorrection(lensModel=None).key() is None


def test_correct_unchanged_when_disabled() -> None:
    """Frames are returned as they are without settings or when nothing is enabled."""
    frame = gridFrame()
    assert Lens.correct(frame, None) is frame
    assert Lens.correct(frame, canonCorrection(distortion=False, vignetting=False)) is frame
    assert Lens.correct(frame, canonCorrection(lensModel="No such lens")) is frame


def test_correct_changes_pixels() -> None:
    """Distortion correction moves pixels except at the center, and vignetting correction brightens the corners."""
    frame = gridFrame()
    corrected = Lens.correct(frame, canonCorrection())
    h, w = frame.shape[:2]
    assert corrected.shape == frame.shape
    assert np.abs(corrected[h//2, w//2] - frame[h//2, w//2]).max() < 0.05
    assert np.abs(corrected - frame).max() > 0.05
    vignetting = Lens.correct(frame, canonCorrection(distortion=False))
    assert vignetting[:8, :8].mean() > frame[:8, :8].mean()
    assert vignetting[h//2, w//2] == pytest.approx(frame[h//2, w//2], abs=0.01)


def test_correct_tca_per_channel() -> None:
    """TCA correction uses one map per channel."""
    frame = gridFrame()
    corrected = Lens.correct(frame, canonCorrection(tca=True, vignetting=False))
    gain, coords = Lens.MAPS[next(iter(Lens.MAPS))]
    assert gain is None
    assert isinstance(coords, list) and len(coords) == 3
    assert corrected.shape == frame.shape


def test_maps_are_cached() -> None:
    """The maps are computed once per geometry and settings, and again when either changes."""
    before = Lens.mapsComputed
    settings = canonCorrection()
    for i in range(3):
        Lens.correct(gridFrame(), settings)
    assert Lens.mapsComputed - before == 1
    Lens.correct(gridFrame(w=64, h=48), settings)
    assert Lens.mapsComputed - before == 2
    Lens.correct(gridFrame(w=64, h=48), canonCorrection(focal=50.0))
    assert Lens.mapsComputed - before == 3


def test_nikon_video_frames_share_maps(generated: Callable[[str, str], str]) -> None:
    """Every frame of a Nikon video is corrected with the settings from its metadata and one set of maps."""
    path = generated("Nikon video", "nikon.mov")
    metadata = Lens.readMetadata(path)
    camera = Lens.findCamera(metadata["make"], metadata["model"])
    lens = Lens.findLens(camera, metadata["lensMake"], metadata["lens"])
    settings = Lens.LensCorrection()
    settings.cameraMaker, settings.cameraModel = camera.maker, camera.model
    settings.lensMaker, settings.lensModel = lens.maker, lens.model
    settings.focal, settings.aperture = metadata["focal"], metadata["aperture"]
    settings.cropRegion = metadata["cropRegion"]
    settings.distortion = settings.tca = settings.vignetting = True
    video = Video()
    video.openVideo(path)
    times = video.videoFrameTimes()
    assert len(times) == 10
    before = Lens.mapsComputed
    for time in times:
        corrected = video.getFrame(file=path, frame=time, colorMode=Video.COLOR_RGB, lens=settings)
        uncorrected = video.getFrame(file=path, frame=time, colorMode=Video.COLOR_RGB)
        assert corrected.shape == uncorrected.shape
        assert np.abs(corrected - uncorrected).max() > 0.05
    assert Lens.mapsComputed - before == 1
