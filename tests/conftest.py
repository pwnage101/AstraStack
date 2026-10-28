"""Shared fixtures for the pytest unit tests.

The tests never import ``UI`` or start GTK, so they run without a display.
"""
import os
import pathlib
from typing import Callable, Iterator

import pytest

from Globals import g
from Preferences import Preferences
from tests.fixtureFiles import writeTestFile

TEST_INPUTS = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "features",
    "testFiles",
    "inputs",
)


class FakeProgress:
    """Stands in for the ``ProgressCounter``/``ProgressBar`` passed to worker functions."""

    def __init__(self) -> None:
        """Initializes the call counters."""
        self.counts = 0
        self.extras = 0

    def count(self, i: int, total: int) -> None:
        """Records a progress update.

        Args:
            i: Index of the item just processed.
            total: Number of items in the chunk.
        """
        self.counts += 1

    def countExtra(self) -> None:
        """Records the end-of-chunk update."""
        self.extras += 1

    def setMessage(self, message: str, force: bool = False) -> None:
        """Ignores progress messages.

        Args:
            message: Message that would be shown.
            force: Whether the message would be shown immediately.
        """


class FakeUI:
    """Holds the only ``g.ui`` attribute the non-UI modules read: ``preferences``."""

    def __init__(self, preferences: Preferences) -> None:
        """Initializes the fake UI.

        Args:
            preferences: Preferences object exposed as ``preferences``.
        """
        self.preferences = preferences


@pytest.fixture
def testInputs() -> str:
    """Returns the path of ``features/testFiles/inputs`` (to be used read-only)."""
    return TEST_INPUTS


@pytest.fixture
def generated(tmp_path: pathlib.Path) -> Callable[[str, str], str]:
    """Returns a writer of generated test files (see ``fixtureFiles.writeTestFile``) in the test's temporary folder."""

    def write(kind: str, name: str) -> str:
        """Writes the file.

        Args:
            kind: Kind of file, e.g. "Display P3 PNG".
            name: File name, relative to the temporary folder.

        Returns:
            The path of the file.
        """
        path = os.path.join(tmp_path, name)
        writeTestFile(kind=kind, path=path)
        return path

    return write


@pytest.fixture
def progress() -> FakeProgress:
    """Returns a fake progress counter for worker functions."""
    return FakeProgress()


@pytest.fixture
def testingPreferences(monkeypatch: pytest.MonkeyPatch) -> Preferences:
    """Returns a ``Preferences`` that never touches the user's preferences file."""
    monkeypatch.setattr(g, "TESTING", True)
    return Preferences()


@pytest.fixture
def fakeUI(monkeypatch: pytest.MonkeyPatch, testingPreferences: Preferences) -> Iterator[FakeUI]:
    """Installs a ``FakeUI`` as ``g.ui`` for the duration of a test."""
    ui = FakeUI(testingPreferences)
    monkeypatch.setattr(g, "ui", ui)
    yield ui
