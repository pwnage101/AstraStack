import os
import sys
from behave import *
from behave.runner import Context
import time
import io
import cv2
import numpy as np
import tifffile
import OpenEXR
from PIL import Image, ImageCms
from threading import Thread
from multiprocessing import Value
from gi import require_version
require_version('Gtk', '3.0')
from gi.repository import Gtk, Gdk, GdkPixbuf, GLib
sys.path.insert(0, os.getcwd())

from UI import UI
from Globals import g
import Export
import ColorSpace
import Lens
from tests.fixtureFiles import exportTags, pngChunksBeforeIDAT, sameProfile, writeTestFile

tabMap = {"Load": UI.LOAD_TAB,
          "Align": UI.ALIGN_TAB,
          "Stack": UI.STACK_TAB,
          "Process": UI.PROCESS_TAB,
          "Export": UI.EXPORT_TAB}

WIDGET_TIMEOUT = 10
TEST_INPUTS_PATH = os.path.join(os.getcwd(), "features/testFiles/inputs")
TEST_REFERENCES_PATH = os.path.join(os.getcwd(), "features/testFiles/reference")

def delay(delay=0.1):
    time.sleep(delay)
    
def idle_add_delay(fn):
    updateDone = Value('b', False)
    def update(updateDone):
        fn()
        updateDone.value = True
    GLib.idle_add(update, updateDone)
    while(not updateDone.value):
        delay(0.01)
    delay(0.01) # Always delay just a bit
    
def widgetValidation(id: str) -> None:
    widget = g.ui.builder.get_object(id)
    deadline = time.monotonic() + WIDGET_TIMEOUT
    while ((widget is None or
            not widget.is_visible() or
            not widget.is_sensitive()) and 
           time.monotonic() < deadline):
        # Give it a chance to become valid before asserting (loading finishes after switching tabs)
        delay(0.05)
    assert widget != None, f"The widget '{id}' does not exist"
    assert widget.is_visible(), f"The widget '{id}' is not visible"
    assert widget.is_sensitive(), f"The widget '{id}' is not sensitive"
    
@given(u'I wait "{ms}"')
def wait(context, ms):
    delay(int(ms)/1000)
    
@given(u'I wait until active tab is "{tab}"')
def waitUntil(context, tab):
    while(True):
        ui = g.ui # Briefly None while cloneGlobals copies the globals in a processing thread
        if(ui is not None and ui.builder.get_object("tabs").get_current_page() == tabMap[tab]):
            break
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
    
def chooseFile(path: str, dialog: str) -> None:
    """Selects a file in a file chooser dialog and confirms it.

    Args:
        path: Absolute path of the file to select. A "*" file name selects every file in its folder.
        dialog: ID of the file chooser dialog.
    """
    window = g.ui.builder.get_object(dialog)
    while(not window.is_visible()):
        delay(0.01)
    delay() # Sometimes it crashes if going to the next step too quickly, so delay a bit
    def update1():
        window.select_filename(path)
    def update2():
        if("*" in path):
            window.select_all()
        if(window.get_action() == Gtk.FileChooserAction.SAVE):
            window.set_current_name(os.path.basename(path))
    def update3():
        window.get_widget_for_response(Gtk.ResponseType.OK).clicked()
    while(window.is_visible()):
        # The chooser loads folders asynchronously, so OK is ignored if pressed before the file is selected; retry
        idle_add_delay(update1)
        idle_add_delay(update2)
        idle_add_delay(update3)
        i = 0
        while(window.is_visible() and i < 300):
            delay(0.01)
            i += 1

@given(u'I choose input file "{file}" from "{dialog}"')
def chooseInputFile(context: Context, file: str, dialog: str) -> None:
    chooseFile(path=os.path.join(TEST_INPUTS_PATH, file), dialog=dialog)

@given(u'I choose output file "{file}" from "{dialog}"')
def chooseOutputFile(context: Context, file: str, dialog: str) -> None:
    chooseFile(path=os.path.join(context.testOutputs, file), dialog=dialog)
    
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
    
def assertImagesEqual(outputPath: str, expectedPath: str) -> None:
    """Asserts that two image files have identical pixels.

    Args:
        outputPath: Path of the output file.
        expectedPath: Path of the expected file.
    """
    img1 = cv2.imread(outputPath, cv2.IMREAD_UNCHANGED)
    img2 = cv2.imread(expectedPath, cv2.IMREAD_UNCHANGED)
    assert np.array_equal(img1, img2), f"'{outputPath}' and '{expectedPath}' are not equal"

@then(u'output "{outputFile}" and input "{inputFile}" should be equal')
def compareOutputToInput(context: Context, outputFile: str, inputFile: str) -> None:
    """Asserts that an output file has the same pixels as a file in ``TEST_INPUTS_PATH``."""
    assertImagesEqual(os.path.join(context.testOutputs, outputFile), os.path.join(TEST_INPUTS_PATH, inputFile))

@then(u'output "{outputFile}" and reference "{referenceFile}" should be equal')
def compareOutputToReference(context: Context, outputFile: str, referenceFile: str) -> None:
    """Asserts that an output file has the same pixels as a file in ``TEST_REFERENCES_PATH``."""
    assertImagesEqual(os.path.join(context.testOutputs, outputFile), os.path.join(TEST_REFERENCES_PATH, referenceFile))
    
@then(u'the stacked image should have values above 1')
def stackNotClipped(context):
    img = g.ui.stack.stackedImage
    assert img.dtype == np.float32, f"Stacked image is {img.dtype}"
    assert img.max() > 1, f"Stacked image maximum is {img.max()}"
    
@then(u'"{widget}" should be sensitive')
def shouldBeSensitive(context, widget):
    assert g.ui.builder.get_object(widget).is_sensitive(), f"'{widget}' is not sensitive"
    
@then(u'"{widget}" should not be sensitive')
def shouldNotBeSensitive(context, widget):
    assert not g.ui.builder.get_object(widget).is_sensitive(), f"'{widget}' is sensitive"
    
@then(u'output "{file}" should be grayscale')
def shouldBeGrayscale(context, file):
    img = cv2.imread(os.path.join(context.testOutputs, file), cv2.IMREAD_UNCHANGED)
    assert img is not None, f"'{file}' could not be read"
    assert (np.array_equal(img[:,:,0], img[:,:,1]) and
            np.array_equal(img[:,:,1], img[:,:,2])), f"'{file}' is not grayscale"
    
@then(u'"{widget}" should be active')
def shouldBeActive(context, widget):
    assert g.ui.builder.get_object(widget).get_active(), f"'{widget}' is not active"
    
@then(u'output "{file}" should exist')
def shouldExist(context, file):
    assert os.path.isfile(os.path.join(context.testOutputs, file)), f"'{file}' does not exist"
    
@then(u'output "{file}" should not exist')
def shouldNotExist(context, file):
    assert not os.path.isfile(os.path.join(context.testOutputs, file)), f"'{file}' exists"
    
@then(u'output "{file}" should be a "{fileType}" "{pixelFormat}" export of input "{source}"')
def checkExport(context, file, fileType, pixelFormat, source):
    path = os.path.join(context.testOutputs, file)
    expected = cv2.cvtColor(cv2.imread(os.path.join(TEST_INPUTS_PATH, source)), cv2.COLOR_BGR2RGB).astype(np.float64)
    dtype = {Export.FORMAT_8BIT: np.uint8,
             Export.FORMAT_16BIT: np.uint16,
             Export.FORMAT_16FLOAT: np.float16,
             Export.FORMAT_32FLOAT: np.float32}[pixelFormat]
    if(pixelFormat == Export.FORMAT_16BIT):
        expected = expected*255
    elif(pixelFormat == Export.FORMAT_16FLOAT or pixelFormat == Export.FORMAT_32FLOAT):
        expected = ColorSpace.SRGB.linearize(expected/255)
    srgb = ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes()
    
    if(fileType == Export.FILE_PNG):
        img = cv2.cvtColor(cv2.imread(path, cv2.IMREAD_UNCHANGED), cv2.COLOR_BGR2RGB)
        chunks = pngChunksBeforeIDAT(path)
        assert (b"sRGB", b"\0") in chunks, f"'{file}' has no sRGB chunk"
    elif(fileType == Export.FILE_JPEG):
        with Image.open(path) as im:
            assert im.format == "JPEG", f"'{file}' is not a JPEG"
            img = np.asarray(im)
            assert sameProfile(im.info.get("icc_profile"), srgb), f"'{file}' does not have the sRGB profile"
    elif(fileType == Export.FILE_TIFF):
        with tifffile.TiffFile(path) as tif:
            page = tif.pages[0]
            img = page.asarray()
            assert page.photometric == tifffile.PHOTOMETRIC.RGB, f"'{file}' is not RGB"
            icc = page.tags["InterColorProfile"].value
            if(pixelFormat == Export.FORMAT_16FLOAT or pixelFormat == Export.FORMAT_32FLOAT):
                assert page.tags["SampleFormat"].value[0] == tifffile.SAMPLEFORMAT.IEEEFP, f"'{file}' is not float"
                assert icc == Export.linearProfile(ColorSpace.SRGB), f"'{file}' does not have the linear sRGB profile"
                # The profile must be readable by LittleCMS and convert linear 0.5 to sRGB 188
                profile = ImageCms.ImageCmsProfile(io.BytesIO(icc))
                gray = Image.new("RGB", (1, 1), (128, 128, 128))
                converted = ImageCms.profileToProfile(gray, profile, ImageCms.createProfile("sRGB"), renderingIntent=1)
                assert converted.getpixel((0, 0)) == (188, 188, 188), f"'{file}' profile is not linear sRGB"
            else:
                assert sameProfile(icc, srgb), f"'{file}' does not have the sRGB profile"
    elif(fileType == Export.FILE_EXR):
        with OpenEXR.File(path, separate_channels=True) as exr:
            channels = exr.channels()
            assert sorted(channels.keys()) == ["B", "G", "R"], f"'{file}' channels are {list(channels.keys())}"
            img = np.dstack([channels[c].pixels for c in ("R", "G", "B")])
            chromaticities = exr.header()["chromaticities"]
            assert np.allclose(chromaticities, (0.64, 0.33, 0.30, 0.60, 0.15, 0.06, 0.3127, 0.3290), atol=1e-6), f"'{file}' chromaticities are {chromaticities}"
    
    assert img.dtype == dtype, f"'{file}' is {img.dtype}, not {np.dtype(dtype)}"
    assert img.shape == expected.shape, f"'{file}' is {img.shape}, not {expected.shape}"
    if(fileType == Export.FILE_JPEG):
        # Lossy, so only check that it's close
        diff = np.abs(img - expected).mean()
        assert diff < 2, f"'{file}' differs from '{source}' by {diff} on average"
    elif(pixelFormat == Export.FORMAT_16FLOAT):
        assert np.allclose(img, expected, rtol=2**-10, atol=2**-24), f"'{file}' values do not round-trip"
    elif(pixelFormat == Export.FORMAT_32FLOAT):
        assert np.allclose(img, expected, rtol=1e-6, atol=1e-7), f"'{file}' values do not round-trip"
    else:
        assert np.array_equal(img, expected), f"'{file}' values do not round-trip"
    
@given(u'I generate the {kind} output file "{file}"')
def generateTestFile(context: Context, kind: str, file: str) -> None:
    """Writes a generated test file (see fixtureFiles.writeTestFile).

    Args:
        context: The behave context.
        kind: Kind of file, e.g. "Display P3 PNG".
        file: Path of the file, relative to the test outputs folder.
    """
    writeTestFile(kind=kind, path=os.path.join(context.testOutputs, file))
        
@then(u'the color space should be "{name}" with transfer "{transfer}"')
def checkColorSpace(context, name, transfer):
    colorSpace = g.actualColorSpace()
    assert colorSpace is not None, "There is no color space"
    assert colorSpace.name == name, f"Color space is '{colorSpace.name}', not '{name}'"
    assert colorSpace.transfer == transfer, f"Transfer is '{colorSpace.transfer}', not '{transfer}'"
    
@then(u'the color space primaries should be "{name}"')
def checkPrimaries(context, name):
    import colour
    cs = colour.RGB_COLOURSPACES[name]
    colorSpace = g.actualColorSpace()
    assert np.allclose(colorSpace.primaries, cs.primaries, atol=1e-3), f"Primaries are {colorSpace.primaries}"
    assert np.allclose(colorSpace.white, cs.whitepoint, atol=1e-3), f"White point is {colorSpace.white}"
    
@then(u'the color space white point should be "{x},{y}"')
def checkWhitePoint(context, x, y):
    colorSpace = g.actualColorSpace()
    assert np.allclose(colorSpace.white, (float(x), float(y)), atol=1e-3), f"White point is {colorSpace.white}"
    
@then(u'output "{file}" should match output "{source}" converted to sRGB within "{tolerance}"')
def checkConvertedToSRGB(context, file, source, tolerance):
    img = cv2.cvtColor(cv2.imread(os.path.join(context.testOutputs, file)), cv2.COLOR_BGR2RGB).astype(int)
    with Image.open(os.path.join(context.testOutputs, source)) as im:
        profile = ImageCms.ImageCmsProfile(io.BytesIO(im.info["icc_profile"]))
        expected = np.asarray(ImageCms.profileToProfile(im, profile, ImageCms.createProfile("sRGB"), renderingIntent=1)).astype(int)
    diff = np.abs(img - expected).max()
    assert diff <= int(tolerance), f"'{file}' differs from the sRGB conversion of '{source}' by {diff}"
    
@then(u'output "{file}" should match output "{source}" within "{tolerance}"')
def checkMatch(context, file, source, tolerance):
    img = cv2.imread(os.path.join(context.testOutputs, file)).astype(int)
    expected = cv2.imread(os.path.join(context.testOutputs, source)).astype(int)
    diff = np.abs(img - expected).max()
    assert diff <= int(tolerance), f"'{file}' differs from '{source}' by {diff}"
    
@then(u'output "{file}" should be neutral at "{x},{y}"')
def checkNeutral(context, file, x, y):
    value = cv2.imread(os.path.join(context.testOutputs, file))[int(y), int(x)].astype(int)
    assert value.max() - value.min() <= 1 and value.max() > 0, f"'{file}' is {value[::-1]} at {x},{y}"
    
@then(u'the error "{message}" should be shown')
def checkError(context, message):
    dialog = g.ui.builder.get_object("errorDialog")
    i = 0
    while(not dialog.is_visible() and i < 500):
        delay(0.01)
        i += 1
    assert dialog.is_visible(), "No error was shown"
    text = dialog.get_property("secondary-text")
    def update():
        dialog.response(Gtk.ResponseType.OK)
    idle_add_delay(update)
    assert text == message, f"The error is '{text}'"
    
@given(u'I wait until processing is finished')
def waitProcessing(context):
    delay()
    while((g.ui.processThread is not None and g.ui.processThread.is_alive()) or
          g.ui.processSpinner.get_property("active")):
        delay(0.01)
    idle_add_delay(lambda: None)

@then(u'the lens correction should be "{camera}", "{lens}", "{focal}", "{aperture}"')
def checkLensCorrection(context, camera, lens, focal, aperture):
    values = (g.lens.cameraModel, g.lens.lensModel, g.lens.focal, g.lens.aperture)
    assert values == (camera, lens, float(focal), float(aperture)), f"Lens correction is {values}"
    def label(combo):
        iter = combo.get_active_iter()
        return combo.get_model()[iter][0] if iter is not None else None
    assert label(g.ui.lensCamera) == camera, f"Camera dropdown shows '{label(g.ui.lensCamera)}'"
    assert label(g.ui.lensLens) == lens, f"Lens dropdown shows '{label(g.ui.lensLens)}'"
    
@then(u'there should be no lens correction')
def checkNoLensCorrection(context):
    assert not g.lens.enabled(), f"Lens correction is enabled ({g.lens.key()})"
    
@given(u'I select camera "{camera}" and lens "{lens}"')
def selectLens(context, camera, lens):
    def update():
        g.ui.populateLensCameras()
        cameraItem = next(c for c in g.ui.lensCameras if Lens.displayName(c) == camera)
        g.ui.selectLensItem(g.ui.lensCamera, g.ui.lensCameras, cameraItem)
        lensItem = next(l for l in g.ui.lensLenses if l.model == lens)
        g.ui.selectLensItem(g.ui.lensLens, g.ui.lensLenses, lensItem)
    idle_add_delay(update)
    
@then(u'the lens correction should change the image')
def checkCorrectionChanges(context):
    from Video import Video
    assert g.lens.enabled(), "Lens correction is not enabled"
    frame = g.file if len(g.ui.video.frames) == 0 else g.ui.video.frames[0]
    corrected = Video().getFrame(g.file, frame, g.actualColor(), lens=g.lens)
    uncorrected = Video().getFrame(g.file, frame, g.actualColor())
    h, w = corrected.shape[:2]
    # The center stays put, the rest moves (and the corners are brightened by the vignetting correction)
    assert np.abs(corrected[h//2, w//2] - uncorrected[h//2, w//2]).max() < 0.05, "The center moved"
    assert np.abs(corrected - uncorrected).max() > 0.05, "The image did not change"
    if(g.lens.vignetting):
        assert corrected[:8,:8].mean() > uncorrected[:8,:8].mean(), "The corners were not brightened"

@then(u'"{label}" should contain "{text}"')
def shouldContain(context, label, text):
    value = g.ui.builder.get_object(label).get_text()
    assert text in value, f"'{label}' is '{value}'"
    
@given(u'I set "{checkbox}" active without changing tab')
def setActive(context, checkbox):
    def update():
        g.ui.builder.get_object(checkbox).set_active(True)
    idle_add_delay(update)

# Checks the widget's own sensitivity, for widgets on tabs that aren't sensitive yet
@then(u'"{widget}" should be sensitive by itself')
def shouldBeSensitiveItself(context, widget):
    assert g.ui.builder.get_object(widget).get_sensitive(), f"'{widget}' is not sensitive"
    
@then(u'"{widget}" should not be sensitive by itself')
def shouldNotBeSensitiveItself(context, widget):
    assert not g.ui.builder.get_object(widget).get_sensitive(), f"'{widget}' is sensitive"


def sliderLevels() -> tuple[float, float]:
    """Returns the Black Level and Exposure slider values.

    Returns:
        The Black Level slider value (log10 of the percentage) and the Exposure slider value (EV).
    """
    return (g.ui.builder.get_object("blackLevelAdjust").get_value(),
            g.ui.builder.get_object("exposureAdjust").get_value())

@given(u'I set neutral levels')
def setNeutralLevels(context: Context) -> None:
    """Turns off Auto Black Level and Auto Exposure and sets both to neutral (0% and 0 EV).

    Args:
        context: The behave context.
    """
    setUncheck(context=context, checkbox="autoBlackLevel")
    setUncheck(context=context, checkbox="autoExposure")
    setAdjustment(context=context, adjustment="blackLevelAdjust", value="-2")
    setAdjustment(context=context, adjustment="exposureAdjust", value="0")
    
@then(u'"{widget}" should be visible')
def shouldBeVisible(context: Context, widget: str) -> None:
    """Checks that the widget's visible property is set.

    Args:
        context: The behave context.
        widget: Id of the widget.

    Raises:
        AssertionError: The widget isn't visible.
    """
    assert g.ui.builder.get_object(widget).get_visible(), f"'{widget}' is not visible"
    
@then(u'"{widget}" should not be visible')
def shouldNotBeVisible(context: Context, widget: str) -> None:
    """Checks that the widget's visible property is not set.

    Args:
        context: The behave context.
        widget: Id of the widget.

    Raises:
        AssertionError: The widget is visible.
    """
    assert not g.ui.builder.get_object(widget).get_visible(), f"'{widget}' is visible"
    
@then(u'the image should have black level "{blackLevel}" and exposure "{exposure}"')
def checkLevels(context: Context, blackLevel: str, exposure: str) -> None:
    """Checks the black level and exposure used and that they were applied to the input.

    Args:
        context: The behave context.
        blackLevel: Expected black level (fraction of full scale).
        exposure: Expected exposure (EV).

    Raises:
        AssertionError: Different levels were used, or they weren't applied.
    """
    process = g.ui.process
    levels = (process.blackLevel, process.exposure)
    assert levels == (float(blackLevel), float(exposure)), f"Levels are {levels}"
    expected = (process.inputImage.astype(np.float64) - float(blackLevel))*2**float(exposure)
    assert np.abs(process.stackedImage - expected).max() < 1e-5, "The levels were not applied"
    
@then(u'the levels should be automatic')
def checkAutoLevels(context: Context) -> None:
    """Checks that the levels were computed from the current image and are shown on the sliders.

    Args:
        context: The behave context.

    Raises:
        AssertionError: The levels or slider values aren't the automatic ones.
    """
    from Process import estimateBlackLevel, sliderFromBlackLevel
    process = g.ui.process
    blackLevel = estimateBlackLevel(process.inputImage)
    assert process.blackLevel == blackLevel, f"Black level is {process.blackLevel}, not {blackLevel}"
    # The brightest channel value is at full scale
    assert abs(process.stackedImage.max() - 1) < 1e-5, f"Maximum is {process.stackedImage.max()}"
    peak = 2**process.exposure*(process.inputImage.max() - blackLevel)
    assert abs(peak - 1) < 1e-5, f"Exposure is {process.exposure}"
    (blackLevelSlider, exposureSlider) = sliderLevels()
    expectedSlider = sliderFromBlackLevel(blackLevel)
    assert abs(blackLevelSlider - expectedSlider) < 1e-9, f"Black Level slider is {blackLevelSlider}"
    assert abs(exposureSlider - process.exposure) < 1e-9, f"Exposure slider is {exposureSlider}"
    
@given(u'I remember the levels')
def rememberLevels(context: Context) -> None:
    """Stores the Black Level and Exposure slider values in the context.

    Args:
        context: The behave context.
    """
    context.levels = sliderLevels()
    
@then(u'the levels should be different from the remembered ones')
def checkLevelsChanged(context: Context) -> None:
    """Checks that both sliders changed since they were remembered.

    Args:
        context: The behave context.

    Raises:
        AssertionError: A slider value didn't change.
    """
    levels = sliderLevels()
    assert levels[0] != context.levels[0] and levels[1] != context.levels[1], f"Levels {levels} were not recomputed"
    
@then(u'the levels should be the remembered ones')
def checkLevelsSame(context: Context) -> None:
    """Checks that both sliders still have the remembered values.

    Args:
        context: The behave context.

    Raises:
        AssertionError: A slider value changed.
    """
    levels = sliderLevels()
    assert levels == context.levels, f"Levels {levels} are not {context.levels}"

    
@given(u'I export output file "{file}" as "{fileType}" "{pixelFormat}"')
def exportAs(context: Context, file: str, fileType: str, pixelFormat: str) -> None:
    """Saves the processed image from the Export tab with the file type and pixel format.

    Args:
        context: The behave context.
        file: Path of the file, relative to the test outputs folder.
        fileType: One of the Export.FILE_* names.
        pixelFormat: One of the Export.FORMAT_* names.
    """
    radios = {Export.FORMAT_8BIT: "pixelFormat8",
              Export.FORMAT_16BIT: "pixelFormat16",
              Export.FORMAT_16FLOAT: "pixelFormat16Float",
              Export.FORMAT_32FLOAT: "pixelFormat32Float"}
    changeTab(context=context, tab="Export")
    setCombo(context=context, value=fileType, combo="fileType")
    setCheck(context=context, checkbox=radios[pixelFormat])
    asyncPressButton(context=context, button="saveButton")
    chooseOutputFile(context=context, file=file, dialog="saveDialog")
    assert os.path.exists(os.path.join(context.testOutputs, file)), f"'{file}' was not written"

@then(u'output "{file}" should be tagged linear "{primaries}"')
def checkLinearTag(context: Context, file: str, primaries: str) -> None:
    """Checks that a float export is tagged with the primaries and white point and a linear tone curve.

    EXR: chromaticities. TIFF: an ICC matrix profile whose rXYZ/gXYZ/bXYZ give the primaries, with gamma 1.0 TRCs.

    Args:
        context: The behave context.
        file: Path of the file, relative to the test outputs folder.
        primaries: Name of the colour-science color space with the expected primaries and white point.

    Raises:
        AssertionError: The tags don't match.
    """
    import colour
    expected = colour.RGB_COLOURSPACES[primaries]
    tags = exportTags(os.path.join(context.testOutputs, file))
    if(file.endswith(".exr")):
        chromaticities = np.array(tags["chromaticities"])
        message = f"'{file}' chromaticities are {chromaticities}"
        # Within the precision of ICC numbers, for inputs described by an ICC profile
        assert np.allclose(chromaticities[:6], np.ravel(expected.primaries), atol=1e-4), message
        assert np.allclose(chromaticities[6:], expected.whitepoint, atol=1e-4), message
    else:
        icc = tags["icc"]
        assert icc is not None and ColorSpace.isMatrixProfile(icc), f"'{file}' has no matrix ICC profile"
        iccTags = ColorSpace.iccTags(icc)
        assert all(ColorSpace.iccCurve(iccTags[t]) == ("gamma", 1.0) for t in ("rTRC", "gTRC", "bTRC")), "Not linear"
        colorSpace = ColorSpace.fromICC(icc)
        assert np.allclose(colorSpace.primaries, expected.primaries, atol=1e-4), f"Primaries are {colorSpace.primaries}"
        assert np.allclose(colorSpace.white, expected.whitepoint, atol=1e-4), f"White point is {colorSpace.white}"

@then(u'output "{file}" should carry the ICC profile of output "{source}"')
def checkICCAttribute(context: Context, file: str, source: str) -> None:
    """Checks that an EXR export carries the source's ICC profile bytes in its iccProfile attribute.

    Args:
        context: The behave context.
        file: Path of the EXR file, relative to the test outputs folder.
        source: Path of the file with an ICC profile that was loaded, relative to the test outputs folder.

    Raises:
        AssertionError: The attribute is missing or differs.
    """
    with Image.open(os.path.join(context.testOutputs, source)) as im:
        icc = im.info["icc_profile"]
    attribute = exportTags(os.path.join(context.testOutputs, file))["iccAttribute"]
    assert attribute == icc, f"'{file}' does not carry the ICC profile of '{source}'"

@then(u'output "{file}" should not carry an ICC profile')
def checkNoICCAttribute(context: Context, file: str) -> None:
    """Checks that an EXR export has no iccProfile attribute.

    Args:
        context: The behave context.
        file: Path of the EXR file, relative to the test outputs folder.

    Raises:
        AssertionError: The attribute is present.
    """
    attribute = exportTags(os.path.join(context.testOutputs, file))["iccAttribute"]
    assert attribute is None, f"'{file}' has an iccProfile attribute"
