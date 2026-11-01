import glob
import os
import sys
from behave import *
from behave.runner import Context
import time
import struct
import zlib
import cv2
import numpy as np
from threading import Thread
from multiprocessing import Value
from gi import require_version
require_version('Gtk', '3.0')
from gi.repository import Gtk, GLib
sys.path.insert(0, os.getcwd())

from UI import UI
from Globals import g
from Video import Video
from FrameSource import ImageSequenceSource
import ColorSpace

tabMap = {"Load": UI.LOAD_TAB,
          "Align": UI.ALIGN_TAB,
          "Stack": UI.STACK_TAB,
          "Process": UI.PROCESS_TAB}

# Up to 1% of pixel values for two images being compared are allowed to be off-by-one.
MAX_VALUES_FRAC_OFF_BY_ONE = 0.01

# Seconds to wait for a widget to become visible and sensitive before failing.
WIDGET_TIMEOUT = 10
TEST_INPUTS_PATH = os.path.join(os.getcwd(), "features/testFiles/inputs")
TEST_REFERENCES_PATH = os.path.join(os.getcwd(), "features/testFiles/reference")

def delay(delay=0.1):
    time.sleep(delay)
    
def idle_add_delay(fn, until=None):
    """Runs a function on the GTK thread and waits for it to finish.

    Args:
        fn: The function to run.
        until: If given, also waits after fn until this returns True, or WIDGET_TIMEOUT seconds
            pass. It is checked on the GTK thread every 50ms.

    Returns:
        Whether until returned True, or True if it wasn't given.
    """
    deadline = time.monotonic() + WIDGET_TIMEOUT
    updateDone = Value('b', False)
    succeeded = Value('b', False)
    def finished():
        succeeded.value = until is None or until()
        updateDone.value = succeeded.value or time.monotonic() >= deadline
        return updateDone.value
    def update():
        fn()
        if(not finished()):
            # GLib calls this again every 50ms until it returns False.
            GLib.timeout_add(50, lambda: not finished())
        return False
    GLib.idle_add(update)
    while(not updateDone.value):
        delay(0.01)
    delay(0.01) # Always delay just a bit
    return succeeded.value

def widgetValidation(id: str) -> None:
    widget = g.ui.builder.get_object(id)
    deadline = time.monotonic() + WIDGET_TIMEOUT
    while ((widget is None or
            not widget.is_visible() or
            not widget.is_sensitive()) and 
           time.monotonic() < deadline):
        # Give it a chance to become valid before asserting
        delay(0.05)
    assert widget != None, f"The widget '{id}' does not exist"
    assert widget.is_visible(), f"The widget '{id}' is not visible"
    assert widget.is_sensitive(), f"The widget '{id}' is not sensitive"
    
@given(u'I wait "{ms}"')
def wait(context, ms):
    delay(int(ms)/1000)
    
@given(u'I wait until active tab is "{tab}"')
def waitUntil(context, tab):
    while(g.ui.builder.get_object("tabs").get_current_page() != tabMap[tab]):
        delay(0.01)

@given(u'I am on tab "{tab}"')
def changeTab(context, tab):
    widgetValidation("tabs")
    def update():
        g.ui.builder.get_object("tabs").set_current_page(tabMap[tab])
    idle_add_delay(update)

@given(u'I press "{button}"')
def pressButton(context, button):
    widgetValidation(button)
    def update():
        g.ui.builder.get_object(button).clicked()
    idle_add_delay(update)

@given(u'I async press "{button}"')
def asyncPressButton(context, button):
    widgetValidation(button)
    def run():
        def update():
            g.ui.builder.get_object(button).clicked()
        GLib.idle_add(update)
    thread = Thread(target=run, args=())
    thread.start()
    delay()
    
def waitForDialog(dialog: str) -> Gtk.FileChooserDialog:
    """Waits for a file chooser dialog to open in the app's home folder.

    The app sets the dialog's folder to its home folder just before opening it, but the dialog
    changes folders asynchronously: it first looks up the folder, and only then switches to it,
    clearing the selection. Until then, it is still in the folder it was last closed in, and
    selecting a file there would be undone once the switch lands.

    Args:
        dialog: ID of the file chooser dialog.

    Returns:
        The dialog.
    """
    window = g.ui.builder.get_object(dialog)
    assert idle_add_delay(lambda: None, until=window.is_visible), f"'{dialog}' did not open"
    delay() # Sometimes it crashes if going to the next step too quickly, so delay a bit
    homeFolder = os.path.expanduser(UI.HOME_DIR)
    def inHomeFolder():
        return window.get_current_folder() == homeFolder
    assert idle_add_delay(lambda: None, until=inHomeFolder), f"'{dialog}' did not open in {homeFolder}"
    return window

def confirmDialog(window: Gtk.FileChooserDialog) -> None:
    """Clicks OK in a file chooser dialog and waits for it to close.

    Args:
        window: The file chooser dialog.
    """
    def clickOk():
        window.get_widget_for_response(Gtk.ResponseType.OK).clicked()
    def closed():
        return not window.is_visible()
    assert idle_add_delay(clickOk, until=closed), "The dialog did not close"

@given(u'I choose input file "{file}" from "{dialog}"')
def chooseInputFile(context: Context, file: str, dialog: str) -> None:
    """Selects the input files in a file open dialog.

    This only supports selecting one file (file="file.avi") OR every file in a
    folder (file="path/to/dir/*"), but nothing in between (e.g. "*.png" in a
    directory containing other types) or else it raises AssertionError.
    """
    desiredPaths = sorted(glob.glob(os.path.join(TEST_INPUTS_PATH, file)))
    assert desiredPaths, f"No input file matches '{file}'"
    window = waitForDialog(dialog)
    actualPaths = []

    # Select the first (or only) file. Since it's in another folder, the dialog
    # switches to that folder, and selects the file once the folder has loaded.
    def selectFirst():
        window.select_filename(desiredPaths[0])
    def firstSelected():
        return desiredPaths[0] in window.get_filenames()
    assert idle_add_delay(selectFirst, until=firstSelected), f"Timed out selecting {desiredPaths[0]}"

    # If every file in a folder is desired, select the entire folder.
    def selectAll():
        if(len(desiredPaths) > 1):
            window.select_all()
        actualPaths[:] = sorted(window.get_filenames())
    idle_add_delay(selectAll)

    # If this assertion fails, the glob probably matched only a subset of the
    # files in the directory, which is unsupported.
    assert actualPaths == desiredPaths, f"Selected {actualPaths} instead of {desiredPaths}"

    # Finally, click OK and wait for the window to close.
    confirmDialog(window)

@given(u'I choose output file "{file}" from "{dialog}"')
def chooseOutputFile(context: Context, file: str, dialog: str) -> None:
    window = waitForDialog(dialog)
    def selectFile():
        window.select_filename(os.path.join(context.testOutputs, file))
        window.set_current_name(file)
    idle_add_delay(selectFile)
    confirmDialog(window)
    
@given(u'I set "{adjustment}" to "{value}"')
def setAdjustment(context, adjustment, value):
    def update():
        g.ui.builder.get_object(adjustment).set_value(float(value))
    idle_add_delay(update)
    
@given(u'I check "{checkbox}"')
def setCheck(context, checkbox):
    widgetValidation(checkbox)
    def update():
        g.ui.builder.get_object(checkbox).set_active(True)
    idle_add_delay(update)
    
@given(u'I uncheck "{checkbox}"')
def setUncheck(context, checkbox):
    widgetValidation(checkbox)
    def update():
        g.ui.builder.get_object(checkbox).set_active(False)
    idle_add_delay(update)
    
@given(u'I select "{value}" from "{combo}"')
def setCombo(context, value, combo):
    widgetValidation(combo)
    model = g.ui.builder.get_object(combo).get_model()
    iter = model.get_iter_first()
    id = None
    i = 0
    while True:
        val = model.get_value(iter, 0)
        if(val == value):
            id = i
        iter = model.iter_next(iter)
        i += 1
        if iter is None:
            break
    assert id is not None, f"Item '{value}' not found"
    def update():
        g.ui.builder.get_object(combo).set_active(id)
    idle_add_delay(update)
    
@given(u'I set point "{var}" to "{point}"')
def setPoint(context, var, point):
    point = point.split(",")
    point = [int(x) for x in point]
    assert (var == "driftP1" or var == "driftP2" or
            var == "areaOfInterestP1" or var == "areaOfInterestP2"), f"'{var}' is not a valid variable"
    def update1():
        if(var == "driftP1"):
            g.ui.clickDriftP1()
        elif(var == "driftP2"):
            g.ui.clickDriftP2()
        elif(var == "areaOfInterestP1"):
            g.ui.clickAreaOfInterest()
    def update2():
        g.ui.mousePosition = point
        if(var == "driftP1" or var == "driftP2"):
            g.ui.setDriftPoint()
        elif(var == "areaOfInterestP1"):
            g.ui.dragBegin()
        elif(var == "areaOfInterestP2"):
            g.ui.dragEnd()
        g.ui.updateImage()
    idle_add_delay(update1)
    idle_add_delay(update2)
    
@step(u'active tab should be "{tab}"')
def activeTabShouldBe(context: Context, tab: str) -> None:
    """Asserts which tab is active.

    Args:
        context: The behave context.
        tab: The tab's name.
    """
    assert g.ui.tabs.get_current_page() == tabMap[tab], f"The active tab is not '{tab}'"

@step(u'the error dialog says "{message}"')
def dismissErrorDialog(context: Context, message: str) -> None:
    """Waits for the error dialog, asserts its message and dismisses it.

    Args:
        context: The behave context.
        message: The expected message.
    """
    dialog = g.ui.builder.get_object("errorDialog")
    deadline = time.monotonic() + WIDGET_TIMEOUT
    while(not dialog.is_visible() and time.monotonic() < deadline):
        delay(0.01)
    assert dialog.is_visible(), "The error dialog was not shown"
    shown = []
    def update() -> None:
        shown.append(dialog.get_property("secondary-text"))
        dialog.response(Gtk.ResponseType.OK)
    idle_add_delay(update)
    deadline = time.monotonic() + WIDGET_TIMEOUT
    while(dialog.is_visible() and time.monotonic() < deadline):
        delay(0.01)
    assert not dialog.is_visible(), "The error dialog was not dismissed"
    assert shown[0] == message, f"The error dialog says '{shown[0]}'"

@then(u'"{var}" should equal "{val}"')
def shouldEqual(context, var, val):
    if(hasattr(g, var)):
        assert str(getattr(g, var)) == str(val), f"'{var}' does not equal '{val}'"
    elif(isinstance(g.ui.builder.get_object(var), Gtk.Adjustment)):
        assert str(g.ui.builder.get_object(var).get_value()) == str(val), f"'{var}' does not equal '{val}'"
    elif(isinstance(g.ui.builder.get_object(var), Gtk.Label)):
        widgetValidation(var)
        assert str(g.ui.builder.get_object(var).get_text()) == str(val), f"'{var}' does not equal '{val}'"
    else:
        assert False, f"'{var}' does not exist"
    
def assertImagesEqual(file1: str, file2: str) -> None:
    """Asserts that two image files match.

    Tolerate values off by 1 in up to ``MAX_VALUES_FRAC_OFF_BY_ONE`` of the
    values because rounding floats to integers can differ by 1 between CPUs.

    Args:
        file1: Path of the first image.
        file2: Path of the second image.
    """
    img1 = cv2.imread(file1, cv2.IMREAD_UNCHANGED)
    img2 = cv2.imread(file2, cv2.IMREAD_UNCHANGED)
    assert img1 is not None and img2 is not None, f"'{file1}' or '{file2}' could not be read"
    assert img1.shape == img2.shape, f"'{file1}' {img1.shape} and '{file2}' {img2.shape} differ in shape"
    assert img1.dtype == img2.dtype, f"'{file1}' {img1.dtype} and '{file2}' {img2.dtype} differ in dtype"
    diff = cv2.absdiff(img1, img2)
    maxDiff = diff.max()
    assert maxDiff <= 1, f"'{file1}' and '{file2}' differ by up to {maxDiff}"
    offByOneFrac = np.count_nonzero(diff) / diff.size
    assert offByOneFrac <= MAX_VALUES_FRAC_OFF_BY_ONE, f"'{file1}' and '{file2}' differ in {offByOneFrac:.1%} of values"

@then(u'output "{outputFile}" and input "{inputFile}" should be equal')
def compareOutputToInput(context: Context, outputFile: str, inputFile: str) -> None:
    """Asserts that an output file has the same pixels as a file in ``TEST_INPUTS_PATH``."""
    assertImagesEqual(
        os.path.join(context.testOutputs, outputFile),
        os.path.join(TEST_INPUTS_PATH, inputFile),
    )

@then(u'output "{outputFile}" and reference "{referenceFile}" should be equal')
def compareOutputToReference(context: Context, outputFile: str, referenceFile: str) -> None:
    """Asserts that an output file has the same pixels as a file in ``TEST_REFERENCES_PATH``."""
    assertImagesEqual(
        os.path.join(context.testOutputs, outputFile),
        os.path.join(TEST_REFERENCES_PATH, referenceFile),
    )

def assertLinearEqual(file1: str, file2: str, maxDiff: float) -> None:
    """Asserts that two image files decode to the same linear values, compared as display-encoded sRGB.

    Each file is read through its own color tags, so an untagged linear input and its tagged sRGB output can match.

    Args:
        file1: Path of the first image.
        file2: Path of the second image.
        maxDiff: Largest allowed difference of the sRGB-encoded values, in 8-bit steps.
    """
    encoded = []
    for file in (file1, file2):
        video = Video(ImageSequenceSource(file))
        linear = cv2.cvtColor(video.getFrame(index=0, colorMode=Video.COLOR_RGB), cv2.COLOR_BGR2RGB)
        encoded.append(ColorSpace.displayEncode(img=linear, workingSpace=video.workingSpace))
    assert encoded[0].shape == encoded[1].shape, f"'{file1}' and '{file2}' differ in shape"
    diff = np.abs(encoded[0] - encoded[1]).max()*255
    assert diff <= maxDiff, f"'{file1}' and '{file2}' differ by up to {diff:.3f} 8-bit steps"

@then(u'output "{outputFile}" and input "{inputFile}" should decode to the same linear values')
def compareLinearOutputToInput(context: Context, outputFile: str, inputFile: str) -> None:
    """Asserts that an output file decodes to the same linear values as a file in ``TEST_INPUTS_PATH``."""
    assertLinearEqual(
        file1=os.path.join(context.testOutputs, outputFile),
        file2=os.path.join(TEST_INPUTS_PATH, inputFile),
        maxDiff=1,
    )

@given(u'I choose saved file "{file}" from "{dialog}"')
def chooseSavedFile(context: Context, file: str, dialog: str) -> None:
    chooseFile(path=os.path.join(context.testOutputs, file), dialog=dialog)

@step(u'I cancel "{dialog}"')
def cancelDialog(context: Context, dialog: str) -> None:
    window = g.ui.builder.get_object(dialog)
    while(not window.is_visible()):
        delay(0.01)
    delay()
    def update():
        window.response(Gtk.ResponseType.CANCEL)
    idle_add_delay(update)
    while(window.is_visible()):
        delay(0.01)

@then(u'the error dialog should show "{text}"')
def errorDialogShows(context: Context, text: str) -> None:
    """Asserts that the error dialog is shown with a message containing the text, then closes it."""
    dialog = g.ui.builder.get_object("errorDialog")
    deadline = time.monotonic() + WIDGET_TIMEOUT
    while(not dialog.is_visible() and time.monotonic() < deadline):
        delay(0.05)
    assert dialog.is_visible(), "The error dialog is not shown"
    delay()
    message = []
    def update():
        message.append(dialog.get_property("secondary-text"))
        dialog.response(Gtk.ResponseType.OK)
    idle_add_delay(update)
    assert text in message[0], f"The error dialog shows '{message[0]}'"

@then(u'output "{file1}" and output "{file2}" should be equal within "{maxDiff}"')
def compareOutputs(context: Context, file1: str, file2: str, maxDiff: str) -> None:
    """Asserts that two output files have the same shape and pixels within a tolerance."""
    img1 = cv2.imread(os.path.join(context.testOutputs, file1), cv2.IMREAD_UNCHANGED)
    img2 = cv2.imread(os.path.join(context.testOutputs, file2), cv2.IMREAD_UNCHANGED)
    assert img1 is not None and img2 is not None, f"'{file1}' or '{file2}' could not be read"
    assert img1.shape == img2.shape, f"'{file1}' {img1.shape} and '{file2}' {img2.shape} differ in shape"
    diff = cv2.absdiff(img1, img2).max()
    assert diff <= float(maxDiff), f"'{file1}' and '{file2}' differ by up to {diff}"

@then(u'output "{file1}" and output "{file2}" should be equal within "{maxMean}" on average')
def compareOutputsOnAverage(context: Context, file1: str, file2: str, maxMean: str) -> None:
    """Asserts that two output files have the same shape and a mean absolute difference within a tolerance."""
    img1 = cv2.imread(os.path.join(context.testOutputs, file1), cv2.IMREAD_UNCHANGED)
    img2 = cv2.imread(os.path.join(context.testOutputs, file2), cv2.IMREAD_UNCHANGED)
    assert img1.shape == img2.shape, f"'{file1}' {img1.shape} and '{file2}' {img2.shape} differ in shape"
    mean = cv2.absdiff(img1, img2).mean()
    assert mean <= float(maxMean), f"'{file1}' and '{file2}' differ by {mean:.2f} on average"

@then(u'output "{file1}" and output "{file2}" should differ')
def outputsDiffer(context: Context, file1: str, file2: str) -> None:
    img1 = cv2.imread(os.path.join(context.testOutputs, file1), cv2.IMREAD_UNCHANGED)
    img2 = cv2.imread(os.path.join(context.testOutputs, file2), cv2.IMREAD_UNCHANGED)
    assert img1.shape != img2.shape or cv2.absdiff(img1, img2).max() > 8, f"'{file1}' and '{file2}' are equal"

@then(u'output "{file}" at "{x},{y}" should be about "{r},{g},{b}" within "{tolerance}"')
def outputPixel(context: Context, file: str, x: str, y: str, r: str, g: str, b: str, tolerance: str) -> None:
    """Asserts the 8-bit RGB value of an output pixel."""
    img = cv2.imread(os.path.join(context.testOutputs, file))
    pixel = img[int(y), int(x)][::-1].astype(int)
    expected = np.array([int(r), int(g), int(b)])
    assert np.abs(pixel - expected).max() <= int(tolerance), f"'{file}' at {x},{y} is {pixel.tolist()}"

@then(u'output "{file}" should not exist')
def outputMissing(context: Context, file: str) -> None:
    assert not os.path.exists(os.path.join(context.testOutputs, file)), f"'{file}' was written"

def isSRGBProfile(icc: bytes | None) -> bool:
    """Returns whether an ICC profile describes sRGB.

    Args:
        icc: The profile bytes, or None.

    Returns:
        True for a matrix/TRC profile with sRGB primaries and the sRGB curve.
    """
    if(not icc or not ColorSpace.isMatrixProfile(icc)):
        return False
    params, workingSpace = ColorSpace.fromICC(icc)
    return params.transfer == ("sRGB", None) and workingSpace is ColorSpace.SRGB

@then(u'output "{file}" should be tagged as sRGB')
def outputTaggedSRGB(context: Context, file: str) -> None:
    """Asserts that a saved PNG has exactly the sRGB chunks, or that a JPEG/TIFF carries an sRGB profile."""
    path = os.path.join(context.testOutputs, file)
    if(file.endswith(".png")):
        chunks = ColorSpace.pngChunks(path)
        colorChunks = [(kind, data) for kind, data in chunks if kind in (b"cICP", b"iCCP", b"sRGB", b"gAMA", b"cHRM")]
        kinds = [kind for kind, data in colorChunks]
        assert kinds == [b"cICP", b"iCCP", b"sRGB", b"gAMA"], f"'{file}' has {kinds}"
        chunks = dict(colorChunks)
        assert chunks[b"cICP"] == bytes([1, 13, 0, 1]), f"'{file}' has cICP {list(chunks[b'cICP'])}"
        assert chunks[b"sRGB"] == bytes([0]), f"'{file}' has sRGB {list(chunks[b'sRGB'])}"
        assert chunks[b"gAMA"] == struct.pack(">I", 45455), f"'{file}' has gAMA {chunks[b'gAMA']}"
        name, icc = chunks[b"iCCP"].split(b"\0", 1)
        assert name == b"sRGB", f"'{file}' has iCCP profile name {name}"
        assert isSRGBProfile(zlib.decompress(icc[1:])), f"'{file}' has an iCCP profile that isn't sRGB"
    else:
        assert isSRGBProfile(ColorSpace.readICC(path)), f"'{file}' doesn't carry an sRGB profile"

@then(u'output "{file}" should have "{bits}" bits')
def outputBits(context: Context, file: str, bits: str) -> None:
    img = cv2.imread(os.path.join(context.testOutputs, file), cv2.IMREAD_UNCHANGED)
    assert img.dtype == (np.uint16 if bits == "16" else np.uint8), f"'{file}' is {img.dtype}"

@then(u'the loaded file should be tagged as sRGB')
def loadedSRGB(context: Context) -> None:
    assert g.decodeParams.transfer == ("sRGB", None), f"The transfer is {g.decodeParams.transfer}"
    assert g.workingSpace == ColorSpace.SRGB, "The working space isn't sRGB"

def regionMean(context: Context, file: str, region: str) -> np.ndarray:
    """Returns the mean linear sRGB value of a region of an output file.

    Args:
        context: The behave context.
        file: The output file.
        region: "x1,y1,x2,y2".

    Returns:
        The mean linear RGB value.
    """
    x1, y1, x2, y2 = [int(v) for v in region.split(",")]
    video = Video(ImageSequenceSource(os.path.join(context.testOutputs, file)))
    linear = cv2.cvtColor(video.getFrame(index=0, colorMode=Video.COLOR_RGB), cv2.COLOR_BGR2RGB)
    return linear[y1:y2, x1:x2].reshape(-1, 3).mean(axis=0)

@then(u'output "{file}" in region "{region}" should have the camera ratios "{r},{g},{b}" of "{rawFile}" within "{tolerance}"')
def outputCameraRatios(context: Context,
                       file: str,
                       region: str,
                       r: str,
                       g: str,
                       b: str,
                       rawFile: str,
                       tolerance: str) -> None:
    """Asserts the ratios to green of a region of an output, converted back from sRGB to the RAW file's camera RGB."""
    import rawpy
    with rawpy.imread(os.path.join(TEST_INPUTS_PATH, rawFile)) as raw:
        rgbCam = ColorSpace.rawRGBCam(raw)
    camera = np.linalg.solve(rgbCam, regionMean(context=context, file=file, region=region))
    ratios = camera/camera[1]
    expected = np.array([float(r), float(g), float(b)])
    assert np.abs(ratios/expected - 1).max() <= float(tolerance), f"The camera ratios are {ratios.round(3).tolist()}"

@then(u'output "{file}" in region "{region}" should be neutral within "{tolerance}"')
def outputNeutral(context: Context, file: str, region: str, tolerance: str) -> None:
    mean = regionMean(context=context, file=file, region=region)
    ratios = mean/mean[1]
    assert np.abs(ratios - 1).max() <= float(tolerance), f"The ratios to green are {ratios.round(3).tolist()}"
