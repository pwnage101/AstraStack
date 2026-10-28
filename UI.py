from __future__ import annotations
from lazy import lazy
cv2 = lazy("cv2")
np = lazy("numpy")
pystackreg = lazy("pystackreg")
webbrowser = lazy("webbrowser")
psutil = lazy("psutil")
urllib = lazy("urllib")
json = lazy("json")
import subprocess, os, sys
import gc
import math
import math
import time
from packaging import version
from threading import Thread
from multiprocessing import active_children, get_start_method
from concurrent.futures import ProcessPoolExecutor

from Preferences import Preferences
from Video import Video
from Align import Align, centerOfMass, dilateImg, normalizeImg
from Stack import Stack, transform
from Process import Process, blackLevelFromSlider, sliderFromBlackLevel
import Export
import ColorSpace
import Lens
from Globals import *
from deconvolution import *

from gi import require_version
require_version('Gtk', '3.0')
from gi.repository import Gtk, Gdk, GdkPixbuf, GLib

class UI:
    
    LOAD_TAB = 0
    ALIGN_TAB = 1
    STACK_TAB = 2
    PROCESS_TAB = 3
    EXPORT_TAB = 4
    
    TITLE = "AstraStack"
    VERSION = "2.5.1"
    SNAP_DIR = ""
    HOME_DIR = "~"
    
    def __init__(self) -> None:
        self.preferences = Preferences()
        self.pids = []
        self.newVersionUrl = ""
        self.video = None
        self.align = None
        self.stack = None
        self.process = None
        self.mousePosition = None
        self.clickedDriftP1 = False
        self.clickedDriftP2 = False
        self.clickedAreaOfInterest = False
        
        self.builder = Gtk.Builder()
        self.builder.add_from_file(UI.SNAP_DIR + "ui/ui.glade")
        
        if(self.preferences.get('darkTheme')):
            self.builder.get_object("theme2").set_active(True)
            self.changeTheme()
        
        self.window = self.builder.get_object("mainWindow")
        self.saveDialog = self.builder.get_object("saveDialog")
        self.openDialog = self.builder.get_object("openDialog")
        self.preferencesDialog = self.builder.get_object("preferencesDialog")
        self.tabs = self.builder.get_object("tabs")
        self.cpus = self.builder.get_object("cpus")
        self.colorMode = self.builder.get_object("colorMode")
        self.progressBox = self.builder.get_object("progressBox")
        self.progress = self.builder.get_object("progress")
        self.processSpinner = self.builder.get_object("processSpinner")
        self.frame = self.builder.get_object("frame")
        self.overlay = self.builder.get_object("overlay")
        self.qualityImage = self.builder.get_object("qualityImage")
        self.psfImage = self.builder.get_object("psfImage")
        self.histImage = self.builder.get_object("histImage")
        self.frameSlider = self.builder.get_object("frameSlider")
        self.frameScale = self.builder.get_object("frameScale")
        self.startFrame = self.builder.get_object("startFrame")
        self.endFrame = self.builder.get_object("endFrame")
        self.normalize = self.builder.get_object("normalize")
        self.dilate = self.builder.get_object("dilate")
        self.alignChannels = self.builder.get_object("alignChannels")
        self.autoCrop = self.builder.get_object("autoCrop")
        self.driftType = self.builder.get_object("driftType")
        self.transformation = self.builder.get_object("transformation")
        self.fileType = self.builder.get_object("fileType")
        self.pixelFormats = {Export.FORMAT_8BIT: self.builder.get_object("pixelFormat8"),
                             Export.FORMAT_16BIT: self.builder.get_object("pixelFormat16"),
                             Export.FORMAT_16FLOAT: self.builder.get_object("pixelFormat16Float"),
                             Export.FORMAT_32FLOAT: self.builder.get_object("pixelFormat32Float")}
        self.drizzleFactor = self.builder.get_object("drizzleFactor")
        self.drizzleInterpolation = self.builder.get_object("drizzleInterpolation")
        self.frameSort = self.builder.get_object("frameSort")
        self.limit = self.builder.get_object("limit")
        self.limitPercent = self.builder.get_object("limitPercent")
        self.averageRadio = self.builder.get_object("averageRadio")
        self.medianRadio = self.builder.get_object("medianRadio")
        self.showMask = self.builder.get_object("showMask")
        self.blackLevel = self.builder.get_object("blackLevelAdjust")
        self.exposure = self.builder.get_object("exposureAdjust")
        self.autoBlackLevel = self.builder.get_object("autoBlackLevel")
        self.autoExposure = self.builder.get_object("autoExposure")
        self.nonLinearNotice = self.builder.get_object("nonLinearNotice")
        self.updatingLevels = False
        self.lensCamera = self.builder.get_object("lensCamera")
        self.lensLens = self.builder.get_object("lensLens")
        self.lensFocal = self.builder.get_object("lensFocalAdjust")
        self.lensAperture = self.builder.get_object("lensApertureAdjust")
        self.lensDistortion = self.builder.get_object("lensDistortion")
        self.lensTCA = self.builder.get_object("lensTCA")
        self.lensVignetting = self.builder.get_object("lensVignetting")
        self.lensStatus = self.builder.get_object("lensStatus")
        self.reference = None
        
        # Lensfun cameras and lenses by their index in the dropdowns
        self.lensCameras = []
        self.lensLenses = []
        self.lensListCamera = None
        self.lensMetadata = {}
        self.lensUpdating = True
        self.imageSize = None
        self.lensCamera.get_model().append(None, ["None", -1])
        self.lensLens.get_model().append(None, ["None", -1])
        self.lensCamera.set_active(0)
        self.lensLens.set_active(0)
        self.lensUpdating = False
        g.lens = Lens.LensCorrection()
        
        self.openDialog.set_preview_widget(Gtk.Image())
        self.saveDialog.set_preview_widget(Gtk.Image())
        
        self.builder.get_object("alignTab").set_sensitive(False)
        self.builder.get_object("stackTab").set_sensitive(False)
        self.builder.get_object("processTab").set_sensitive(False)
        self.builder.get_object("exportTab").set_sensitive(False)
        
        self.builder.get_object("deconvolveCircularDiameterWidget").add_mark(1, Gtk.PositionType.TOP, None)
        self.builder.get_object("deconvolveCircularAmountWidget").add_mark(25, Gtk.PositionType.TOP, None)
        self.builder.get_object("deconvolveGaussianDiameterWidget").add_mark(1, Gtk.PositionType.TOP, None)
        self.builder.get_object("deconvolveGaussianAmountWidget").add_mark(25, Gtk.PositionType.TOP, None)
        self.builder.get_object("deconvolveLinearDiameterWidget").add_mark(1, Gtk.PositionType.TOP, None)
        self.builder.get_object("deconvolveLinearAmountWidget").add_mark(25, Gtk.PositionType.TOP, None)
        self.builder.get_object("deconvolveLinearAngleWidget").add_mark(0, Gtk.PositionType.TOP, None)
        self.builder.get_object("deconvolveCustomAmountWidget").add_mark(25, Gtk.PositionType.TOP, None)
        self.builder.get_object("deringAdaptive").add_mark(0, Gtk.PositionType.TOP, None)
        self.builder.get_object("deringDark").add_mark(0, Gtk.PositionType.TOP, None)
        self.builder.get_object("deringBright").add_mark(0, Gtk.PositionType.TOP, None)
        self.builder.get_object("deringSize").add_mark(0, Gtk.PositionType.TOP, None)
        self.builder.get_object("deringBlend").add_mark(1, Gtk.PositionType.TOP, None)
        self.builder.get_object("blackLevel").add_mark(0, Gtk.PositionType.TOP, None)
        self.builder.get_object("gamma").add_mark(100, Gtk.PositionType.TOP, None)
        self.builder.get_object("exposure").add_mark(0, Gtk.PositionType.TOP, None)
        self.builder.get_object("red").add_mark(100, Gtk.PositionType.TOP, None)
        self.builder.get_object("green").add_mark(100, Gtk.PositionType.TOP, None)
        self.builder.get_object("blue").add_mark(100, Gtk.PositionType.TOP, None)
        self.builder.get_object("saturation").add_mark(100, Gtk.PositionType.TOP, None)

        self.disableScroll()
        
        self.cpus.set_upper(min(61, os.cpu_count())) # 61 is the maximum that Windows allows
        self.cpus.set_value(min(61, math.ceil(self.preferences.get("cpus", os.cpu_count()/2))))
        g.pool = None

        self.processThread = None
        
        self.builder.connect_signals(self)
        
        # Needed so it can be temporarily removed
        self.limitPercentSignal = self.limitPercent.connect("value-changed", self.setLimitPercent)
        
        # Default Open/Save Dialog buttons
        self.openDialog.add_button("Cancel", Gtk.ResponseType.CANCEL)
        self.openDialog.add_button("Open", Gtk.ResponseType.OK)
        self.saveDialog.add_button("Cancel", Gtk.ResponseType.CANCEL)
        self.saveDialog.add_button("Save", Gtk.ResponseType.OK)
        
        g.driftP1 = (0, 0)
        g.driftP2 = (0, 0)
        
        g.areaOfInterestP1 = (0, 0)
        g.areaOfInterestP2 = (0, 0)
        
        g.guessedColorMode = Video.COLOR_RGB
        g.colorSpace = ColorSpace.SRGB
        g.colorSpaceFromImage = True
        
        g.limit = 0
        
        g.sharpen = [None]*5
        g.radius  = [None]*5
        g.denoise = [None]*5
        g.level   = [None]*5
        
        g.stretchFunction = Process.STRETCH_NONE
        g.gamma = 100
        self.readLevels()

        self.window.show_all()
        self.loadOpenCV()
        self.setThreads()
        self.checkNewVersion()
        self.setProgress()
        self.setColorMode()
        self.setNormalize()
        self.setDilate()
        self.setAlignChannels()
        self.setDriftType()
        self.setTransformation()
        self.setDrizzleFactor()
        self.setDrizzleInterpolation()
        self.setFrameSortMethod()
        self.setAutoCrop()
        self.setFileType()
        self.setAutoLevels()
        self.frameScale.set_sensitive(False)
        
        g.file = None
        g.reference = "0"
        g.deconvolveCustomFile = None
        
    # Cancels scroll event for widget
    def propagateScroll(self, widget, event):
        Gtk.propagate_event(widget.get_parent(), event)
        
    # Disables the scroll event from some fields
    def disableScroll(self):
        mask = Gdk.EventMask.BUTTON_MOTION_MASK | Gdk.EventMask.BUTTON_PRESS_MASK | Gdk.EventMask.BUTTON_RELEASE_MASK | Gdk.EventMask.KEY_RELEASE_MASK | Gdk.EventMask.TOUCH_MASK
        self.builder.get_object("denoiseWidget1").set_events(mask)
        self.builder.get_object("denoiseWidget2").set_events(mask)
        self.builder.get_object("denoiseWidget3").set_events(mask)
        self.builder.get_object("denoiseWidget4").set_events(mask)
        self.builder.get_object("denoiseWidget5").set_events(mask)
        
        self.builder.get_object("radiusWidget1").set_events(mask)
        self.builder.get_object("radiusWidget2").set_events(mask)
        self.builder.get_object("radiusWidget3").set_events(mask)
        self.builder.get_object("radiusWidget4").set_events(mask)
        self.builder.get_object("radiusWidget5").set_events(mask)
    
    # Shows the error dialog with the given title and message
    def showErrorDialog(self, message):
        dialog = self.builder.get_object("errorDialog")
        dialog.format_secondary_text(message)
        response = dialog.run()
        dialog.hide()
        return response
        
    # Shows the warning dialog with the given title and message
    def showWarningDialog(self, message):
        dialog = self.builder.get_object("warningDialog")
        dialog.format_secondary_text(message)
        response = dialog.run()
        dialog.hide()
        return response
        
    # Opens the About dialog
    def showAbout(self, *args):
        dialog = self.builder.get_object("about")
        dialog.set_program_name(UI.TITLE)
        dialog.set_version(UI.VERSION)
        response = dialog.run()
        dialog.hide()
        
    # Resets the env to a 'clean' version not tainted by pyinstaller
    def cleanseEnv(self):
        env = dict(os.environ)  # make a copy of the environment
        lp_key = 'LD_LIBRARY_PATH'  # for GNU/Linux and *BSD.
        lp_orig = env.get(lp_key + '_ORIG')
        if lp_orig is not None:
            env[lp_key] = lp_orig  # restore the original, unmodified value
        else:
            # This happens when LD_LIBRARY_PATH was not set.
            # Remove the env var as a last resort:
            env.pop(lp_key, None)
        # 'Clean' some other variables so they aren't using astrastack paths
        env.pop("GTK_PATH", None)
        env.pop("GTK_DATA_PREFIX", None)
        env.pop("XDG_DATA_DIRS", None)
        env.pop("GIO_MODULE_DIR", None)
        env.pop("PANGO_LIBDIR", None)
        env.pop("GDK_PIXBUF_MODULE_FILE", None)
        env.pop("GI_TYPELIB_PATH", None)
        return env
    
    # When a link in the about dialog is clicked
    def clickAboutLink(self, dialog, url):
        if os.name == 'posix': # For Linux
            subprocess.Popen(('xdg-open', url), env=self.cleanseEnv())
            return True
        else:
            webbrowser.open(url)
            return True
        return False
        
    # Opens the user manual in the default pdf application
    def showManual(self, *args):
        print(sys.platform)
        if sys.platform.startswith('darwin'):
            subprocess.call(('open', "manual/Manual.pdf"))
        elif os.name == 'nt': # For Windows
            os.startfile("manual\\Manual.pdf")
        elif os.name == 'posix': # For Linux
            # Now open the file
            subprocess.Popen(('xdg-open', UI.SNAP_DIR + "manual/Manual.pdf"), env=self.cleanseEnv())
        
    # Disable inputs
    def disableUI(self):
        self.builder.get_object("tabs").set_sensitive(False)
        self.builder.get_object("toolbar").set_sensitive(False)
        
    # Enable inputs
    def enableUI(self):
        self.builder.get_object("tabs").set_sensitive(True)
        self.builder.get_object("toolbar").set_sensitive(True)
        
    # The following is needed to forcibly refresh the value spacing of the slider
    def fixFrameSliderBug(self):
        self.frameScale.set_value_pos(Gtk.PositionType.RIGHT)
        self.frameScale.set_value_pos(Gtk.PositionType.LEFT)
        
    # Loads opencv asynchronously
    def loadOpenCV(self):
        def load():
            time.sleep(0.5)
            import cv2
            return
        thread = Thread(target=load, args=())
        thread.start()
        
    # Sets the number of threads to use
    def setThreads(self, *args):
        def initPool(method=None):
            if(method == "spawn"):
                GLib.idle_add(self.disableUI)
            g.nThreads = int(self.cpus.get_value())
            self.preferences.set("cpus", g.nThreads)
            if(g.pool is not None):
                g.pool.shutdown()
            # Unreachable PyAV objects (e.g. from the file chooser preview) are freed here, not by the garbage collector of a
            # forked process, where freeing a codec waits for threads that don't exist there.  Objects that exist when the
            # processes are forked are frozen so the processes never collect them
            gc.collect()
            gc.freeze()
            g.pool = ProcessPoolExecutor(g.nThreads)
            
            # This seems like the most reliable way to get the pid of pool processes
            self.pids = []
            before = list(map(lambda p : p.pid, active_children()))
            g.pool.submit(dummy, ()).result()
            gc.unfreeze()
            after = list(map(lambda p : p.pid, active_children()))
            for pid in after:
                if(pid not in before):
                    self.pids.append(pid)
            if(method == "spawn"):
                GLib.idle_add(self.enableUI)
        
        # Behave a bit differently depending on platform
        if(get_start_method() == "spawn"):
            thread = Thread(target=initPool, args=(get_start_method(),))
            thread.start()
        else:
            initPool()
            
    # Change the theme of the application
    def changeTheme(self, *args):
        theme1 = self.builder.get_object("theme1")
        theme2 = self.builder.get_object("theme2")
        theme3 = self.builder.get_object("theme3")
        settings = Gtk.Settings.get_default()
        if(theme1.get_active()):
            settings.set_property("gtk-application-prefer-dark-theme", False)
            self.preferences.set("darkTheme", False)
        elif(theme2.get_active()):
            settings.set_property("gtk-application-prefer-dark-theme", True)
            self.preferences.set("darkTheme", True)
        
    # Checks github to see if there is a new version available
    def checkNewVersion(self):
        def callUrl():
            import ssl
            ssl._create_default_https_context = ssl._create_unverified_context
            try:
                if(not g.TESTING):
                    contents = urllib.request.urlopen("https://api.github.com/repos/Finalfantasykid/AstraStack/releases/latest").read()
                    obj = json.loads(contents)
                    if(version.parse(obj['name']) > version.parse(UI.VERSION)):
                        self.newVersionUrl = "https://astrastack.ca/?page=download"
                        button.show()
            except:
                return
        button = self.builder.get_object("newVersion")
        button.hide()
        if('SNAP' not in os.environ):
            thread = Thread(target=callUrl, args=())
            thread.start()
        
    # Opens the GitHub releases page in a browser
    def clickNewVersion(self, *args):
        if os.name == 'posix': # For Linux
            # Now open the file
            subprocess.Popen(('xdg-open', self.newVersionUrl), env=self.cleanseEnv())
        else:
            webbrowser.open(self.newVersionUrl)
    
    # Checks to see if there will be enough memory to process the image
    def checkMemory(self, w=0, h=0):
        if(Process.estimateMemoryUsage(w, h) > psutil.virtual_memory().available):
            response = self.showWarningDialog("Your system may not have enough memory to process this file, are you sure you want to continue?")
            return (response == Gtk.ResponseType.YES)
        return True
    
    # Returns a new GdkPixbuf from the given RGB image (linear float images are display encoded from colorSpace, 8-bit images are used as is)
    def createPixbuf(self, img, colorSpace=None):
        if(img.dtype != np.uint8):
            img = ColorSpace.displayImage(img, colorSpace)
        z = img.tobytes()
        Z = GLib.Bytes.new(z)
        height, width = img.shape[:2]
        return GdkPixbuf.Pixbuf.new_from_bytes(Z, GdkPixbuf.Colorspace.RGB, False, 8, width, height, width*3)
    
    # Shows preview image in file chooser dialog
    def updatePreview(self, dialog):
        path = dialog.get_preview_filename()
        pixbuf = None
        if(path != None and os.path.isfile(path)):
            try:
                # First try as image
                video = Video()
                img = cv2.cvtColor(video.getFrame(path, path, g.colorMode, dtype=np.uint8, thumbnail=True), cv2.COLOR_BGR2RGB)
                pixbuf = self.createPixbuf(img, video.colorSpace)
            except Exception:
                try:
                    # Now try as video
                    video = Video()
                    img = cv2.cvtColor(video.getFrame(path, 0, g.colorMode, dtype=np.uint8, thumbnail=True), cv2.COLOR_BGR2RGB)
                    pixbuf = self.createPixbuf(img, video.colorSpace)
                except Exception:
                    pass
        if(pixbuf is not None):
            # Valid pixbuf
            maxwidth, maxheight = 250, 250
            width, height = pixbuf.get_width(), pixbuf.get_height()
            scale= min(maxwidth/width, maxheight/height)
            if(scale<1):
                width, height= int(width*scale), int(height*scale)
                pixbuf= pixbuf.scale_simple(width, height, GdkPixbuf.InterpType.BILINEAR)

            dialog.get_preview_widget().set_size_request(width + 10, height + 10)
            dialog.get_preview_widget().set_from_pixbuf(pixbuf.copy())
            dialog.set_preview_widget_active(True)
        else:
            dialog.set_preview_widget_active(False)
    
    # Opens the file chooser to open load a file
    def openVideo(self, *args):
        self.openDialog.set_current_folder(os.path.expanduser(UI.HOME_DIR))
        self.openDialog.set_select_multiple(False)
        self.openDialog.set_filter(self.builder.get_object("videoFilter"))
        response = self.openDialog.run()
        self.openDialog.hide()
        if(response == Gtk.ResponseType.OK):
            try:
                # For some reason respawning the pool after loading drastically speeds up execution 
                # maybe because the preview image 'preloads' the video and then the processes are re-forked?  May only work on Linux...
                if os.name == 'posix':
                    self.setThreads()
                
                g.file = self.openDialog.get_filename()
                self.video = Video()
                self.video.checkMemory()
                thread = Thread(target=self.video.run, args=())
                thread.start()
                self.disableUI()
            except MemoryError as error:
                self.enableUI()
            
    # Opens the file chooser to open load a file
    def openImageSequence(self, *args):
        self.openDialog.set_current_folder(os.path.expanduser(UI.HOME_DIR))
        self.openDialog.set_select_multiple(True)
        self.openDialog.set_filter(self.builder.get_object("imageFilter"))
        response = self.openDialog.run()
        self.openDialog.hide()
        if(response == Gtk.ResponseType.OK):
            try:
                g.file = self.openDialog.get_filenames()
                self.video = Video()
                self.video.checkMemory()
                thread = Thread(target=self.video.run, args=())
                thread.start()
                self.disableUI()
            except MemoryError as error:
                pass
            except: # Open Failed
                self.showErrorDialog("There was an error opening the image, make sure it is a valid image.")
            self.enableUI()
            
    # Opens the file chooser to open load a file
    def openImage(self, *args: object) -> None:
        self.openDialog.set_current_folder(os.path.expanduser(UI.HOME_DIR))
        self.openDialog.set_select_multiple(False)
        self.openDialog.set_filter(self.builder.get_object("imageFilter"))
        response = self.openDialog.run()
        self.openDialog.hide()
        if(response == Gtk.ResponseType.OK):
            self.disableUI()
            g.file = self.openDialog.get_filename()
            try:
                self.video = Video()
                g.guessedColorMode = self.video.guessColorMode(g.file)
                img, g.colorSpace = Video.decodeImage(g.file)
                g.colorSpaceFromImage = True
                h, w = img.shape[:2]
                if(not self.checkMemory(w, h)):
                    raise MemoryError()
                self.imageSize = (w, h)
                self.prefillLens()
                
                self.window.set_title(os.path.split(g.file)[1] + " - " + UI.TITLE)
                self.saveDialog.set_current_name("")
                
                self.updateAutoColorModeText()
                
                self.stack = None
                self.process = Process(g.file, True)
                self.showLevels()
                self.builder.get_object("alignTab").set_sensitive(False)
                self.builder.get_object("stackTab").set_sensitive(False)
                self.builder.get_object("processTab").set_sensitive(True)
                self.builder.get_object("exportTab").set_sensitive(True)
                self.tabs.set_current_page(UI.PROCESS_TAB)
                
                img = cv2.cvtColor(self.video.getFrame(g.file, g.file, g.colorMode, lens=g.lens), cv2.COLOR_BGR2RGB)
                pixbuf = self.createPixbuf(img, g.actualColorSpace())
                self.frame.set_from_pixbuf(pixbuf.copy())
            except MemoryError as error:
                pass
            except Exception as error: # Open Failed
                print(error)
                self.showErrorDialog("There was an error opening the image, make sure it is a valid image.")
            self.enableUI()
            
    # Returns the file name with its extension replaced if it doesn't match the file type
    def fixExtension(self, fileName, fileType):
        root, ext = os.path.splitext(fileName)
        if(ext.lower() in Export.EXTENSIONS[fileType]):
            return fileName
        return root + Export.EXTENSIONS[fileType][0]
    
    # Opens the file chooser to save the final image
    def saveFileDialog(self, *args: object) -> None:
        from pathlib import Path
        fileType = self.fileType.get_active_text()
        pixelFormat = self.getPixelFormat()
        self.saveDialog.set_current_folder(os.path.expanduser(UI.HOME_DIR))
        if(self.saveDialog.get_current_name() == ""):
            # Set default file to save if empty
            if(isinstance(g.file, list)):
                sList = g.file
                self.saveDialog.set_current_name(Path(sList[0]).stem + "_" + Path(sList[-1]).stem)
            else:
                self.saveDialog.set_current_name(Path(g.file).stem)
        self.saveDialog.set_current_name(self.fixExtension(self.saveDialog.get_current_name(), fileType))
        response = self.saveDialog.run()
        if(response == Gtk.ResponseType.OK):
            fileName = self.fixExtension(self.saveDialog.get_filename(), fileType)
            if(fileName != self.saveDialog.get_filename() and os.path.exists(fileName)):
                # The dialog only confirmed overwriting the typed file name
                response = self.showWarningDialog(os.path.basename(fileName) + " already exists, do you want to replace it?")
                if(response != Gtk.ResponseType.YES):
                    self.saveDialog.hide()
                    return
            try:
                while(self.processThread != None and self.processThread.is_alive()):
                    time.sleep(0.1) # Make sure the file isn't still being processed
                Export.write(fileName=fileName,
                             img=self.process.finalImage,
                             fileType=fileType,
                             pixelFormat=pixelFormat,
                             colorSpace=self.process.colorSpace,
                             record=self.process.processingRecord)
            except Exception as error: # Save Failed
                print(error)
                self.showErrorDialog("There was an error saving the image.")
        self.saveDialog.hide()
        
    # Returns the selected pixel format
    def getPixelFormat(self):
        for pixelFormat, radio in self.pixelFormats.items():
            if(radio.get_active()):
                return pixelFormat
        
    # Sets which pixel formats can be chosen for the file type
    def setFileType(self, *args: object) -> None:
        valid = Export.PIXEL_FORMATS[self.fileType.get_active_text()]
        if(self.getPixelFormat() not in valid):
            self.pixelFormats[valid[0]].set_active(True)
        for pixelFormat, radio in self.pixelFormats.items():
            radio.set_sensitive(pixelFormat in valid)
        self.updateExportNotice()
            
    # Switches to the Export tab
    def showExportTab(self, *args):
        self.tabs.set_current_page(UI.EXPORT_TAB)
        
    # Called when the video is finished loading
    def finishedVideo(self):
        def update():
            self.tabs.next_page()
           
            self.frameScale.set_sensitive(True)
           
            self.startFrame.set_lower(0)
            self.startFrame.set_upper(len(self.video.frames)-1)
            self.startFrame.set_value(0)
            
            self.endFrame.set_upper(len(self.video.frames)-1)
            self.endFrame.set_lower(0)
            self.endFrame.set_value(len(self.video.frames)-1)
            
            g.driftP1 = (0, 0)
            g.driftP2 = (0, 0)
            g.areaOfInterestP1 = (0, 0)
            g.areaOfInterestP2 = (0, 0)
            g.reference = self.video.sharpest
            self.frameSlider.set_value(self.video.sharpest)

            self.updateAutoColorModeText()
            self.imageSize = (self.video.width, self.video.height)
            self.prefillLens()
            self.setReference()
            self.setStartFrame()
            self.setEndFrame()
            self.setDriftPoint()
            self.enableUI()
            if(isinstance(g.file, list)):
                sList = g.file
                if(len(sList) != len(self.video.frames)):
                    self.showErrorDialog("Not all images were added to the stack.  Make sure they are all the same dimensions.")
                if(self.video.colorSpaceMismatch):
                    self.showErrorDialog("Not all images have the same color space.  The color space of the first image is used for all of them.")
                self.window.set_title(os.path.split(sList[0])[1] + " ... " + os.path.split(sList[-1])[1] +  " - " + UI.TITLE)
            else:
                self.window.set_title(os.path.split(g.file)[1] + " - " + UI.TITLE)
            self.saveDialog.set_current_name("")

            self.builder.get_object("alignTab").set_sensitive(True)
            self.builder.get_object("stackTab").set_sensitive(False)
            self.builder.get_object("processTab").set_sensitive(False)
            self.builder.get_object("exportTab").set_sensitive(False)
            self.stack = None
        GLib.idle_add(update)
        
    # Opens the file chooser to open load a psf file (g.deconvolveCustomFile)
    def openPSF(self, *args):
        self.openDialog.set_current_folder(os.path.expanduser(UI.HOME_DIR))
        self.openDialog.set_select_multiple(False)
        self.openDialog.set_filter(self.builder.get_object("imageFilter"))
        response = self.openDialog.run()
        self.openDialog.hide()
        if(response == Gtk.ResponseType.OK):
            try:
                img = Video.readImage(self.openDialog.get_filename(), cv2.IMREAD_GRAYSCALE)
                if(max(img.shape) > 100):
                    raise Exception
                if(len(np.unique(img)) <= 1):
                    raise Exception
                img = np.float32(img)/255
                cv2.normalize(img, img, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
                g.deconvolveCustomFile = img
            except: # Open Failed
                self.showErrorDialog("There was an error opening the PSF, make sure it is a valid image.  The PSF should be no larger than 100x100.")
            self.processImage(self.builder.get_object("deconvolveCustomFile"))
       
    # Updates the Auto color mode text     
    def updateAutoColorModeText(self):
        colorMode = self.colorMode.get_active()
        self.colorMode.remove(0)
        if(g.guessedColorMode == Video.COLOR_RGB):
            self.colorMode.prepend_text("Auto (RGB)")
        elif(g.guessedColorMode == Video.COLOR_GRAYSCALE):
            self.colorMode.prepend_text("Auto (Grayscale)")
        elif(g.guessedColorMode == Video.COLOR_RGGB):
            self.colorMode.prepend_text("Auto (RGGB)")
        elif(g.guessedColorMode == Video.COLOR_GRBG):
            self.colorMode.prepend_text("Auto (GRBG)")
        elif(g.guessedColorMode == Video.COLOR_GBRG):
            self.colorMode.prepend_text("Auto (GBRG)")
        elif(g.guessedColorMode == Video.COLOR_BGGR):
            self.colorMode.prepend_text("Auto (BGGR)")
        elif(g.guessedColorMode == Video.COLOR_RGGB_VNG):
            self.colorMode.prepend_text("Auto (RGGB VNG)")
        elif(g.guessedColorMode == Video.COLOR_GRBG_VNG):
            self.colorMode.prepend_text("Auto (GRBG VNG)")
        elif(g.guessedColorMode == Video.COLOR_GBRG_VNG):
            self.colorMode.prepend_text("Auto (GBRG VNG)")
        elif(g.guessedColorMode == Video.COLOR_BGGR_VNG):
            self.colorMode.prepend_text("Auto (BGGR VNG)")
        if(colorMode == Video.COLOR_AUTO):
            self.colorMode.set_active(0)
        
    # Called when the tab is changed.  Updates parts of the UI based on the tab
    def changeTab(self, notebook, page, page_num, user_data=None):
        if((page_num == UI.LOAD_TAB or page_num == UI.ALIGN_TAB) and self.video is not None):
            self.setStartFrame()
            self.setEndFrame()
            if(len(self.video.frames) > 0):
                self.frameScale.show()
            self.updateImage(None, page_num)
        elif(page_num == UI.STACK_TAB and self.stack is not None):
            self.frameSlider.set_lower(0)
            self.frameSlider.set_upper(max(0,len(self.stack.tmats)-1))
            self.frameSlider.set_value(0)
            if(len(self.video.frames) > 0):
                self.frameScale.show()
            self.updateImage(None, page_num)
        elif(page_num == UI.PROCESS_TAB and (self.process is not None or
                                             (self.video is not None and len(self.video.frames) == 0))):
            # A single image's Process is rebuilt by processImage after a color mode change
            self.frameScale.hide()
            self.processImage()
        elif(page_num == UI.EXPORT_TAB and (self.process is not None or
                                            (self.video is not None and len(self.video.frames) == 0))):
            # Shows the Process result
            self.frameScale.hide()
            if(self.process is None):
                self.processImage()
            elif(self.processThread is None or not self.processThread.is_alive()):
                self.finishedProcessing()
        self.fixFrameSliderBug()
    
    # Changes the image frame to the frameSlider position    
    def updateImage(self, adjustment=None, page_num=None):
        if(self.video is None):
            return
        if(page_num is None):
            page_num = self.tabs.get_current_page()
        if(page_num == UI.LOAD_TAB or page_num == UI.ALIGN_TAB):
            videoIndex = int(self.frameSlider.get_value())
            if(len(self.video.frames) == 0):
                # Single Image
                img = ColorSpace.displayImage(cv2.cvtColor(self.video.getFrame(g.file, g.file, g.actualColor(), lens=g.lens), cv2.COLOR_BGR2RGB), g.actualColorSpace())
            else:
                # Video/Sequence
                img = ColorSpace.displayImage(cv2.cvtColor(self.video.getFrame(g.file, self.video.frames[videoIndex], g.actualColor(), lens=g.lens), cv2.COLOR_BGR2RGB), g.actualColorSpace())
                
            height, width = img.shape[:2]
            
            # Dilate
            if(g.dilate):
                img = dilateImg(img)
            # Normalize
            if(g.normalize):
                img = normalizeImg(img)
            # Center of Gravity
            if(g.driftType == Align.DRIFT_GRAVITY):
                (cx, cy) = centerOfMass(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY))
                C = np.float32([[1, 0, int(width/2) - int(cx)], 
                                [0, 1, int(height/2) - int(cy)]])

                img = cv2.warpAffine(img, C, (width, height), flags=cv2.INTER_NEAREST)
            # Frame Deltas
            if(g.driftType == Align.DRIFT_DELTA):
                C = np.identity(3, dtype=np.float64)
                for i, M in enumerate(self.video.deltas):
                    if((i > videoIndex and i <= int(g.reference)) or
                       (i > int(g.reference) and i <= videoIndex)):
                        C = C.dot(M)
                C = np.float32([[1, 0, C[0][2]], 
                                [0, 1, C[1][2]]])
                if(int(g.reference) > videoIndex):
                    C[0][2] = -C[0][2]
                    C[1][2] = -C[1][2]
                img = cv2.warpAffine(img, C, (width, height), flags=cv2.INTER_NEAREST)

            pixbuf = self.createPixbuf(img)
            self.frame.set_from_pixbuf(pixbuf.copy())
        elif(page_num == UI.STACK_TAB and self.stack is not None):
            if(int(self.frameSlider.get_value()) >= len(self.stack.tmats)):
                return
            tmat = self.stack.tmats[int(self.frameSlider.get_value())]
            videoIndex = tmat[0]
            M = tmat[1]
            img = self.video.getFrame(g.file, videoIndex, g.actualColor(), lens=g.lens)
            if(g.autoCrop):
                ref = self.stack.refBG
            else:
                ref = None
            img = transform(img, ref, M, self.align.minX, self.align.maxX, self.align.minY, self.align.maxY, g)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            pixbuf = self.createPixbuf(img, g.actualColorSpace())
            self.frame.set_from_pixbuf(pixbuf.copy())
            self.updateQualityImage()
            
    # Draws a rectangle where the area of interest is
    def drawOverlay(self, widget, cr):
        width = widget.get_allocated_width()
        height = widget.get_allocated_height()
        
        def drawPoint(cr, x, y):
            cr.new_sub_path()
            cr.set_line_width(2)
            cr.set_source_rgb(1, 1, 1)

            cr.arc(x, y, 2, 0, 2*math.pi)
            cr.stroke_preserve()
            
            cr.set_source_rgb(1, 0, 0)
            cr.fill()
            
        def drawRect(cr, x1, y1, x2, y2):
            cr.rectangle(0, 0, x1, y1)
            cr.rectangle(0, y1, x1, (y2-y1))
            cr.rectangle(0, y1, x1, height*2)
            cr.rectangle(x1, y2, (x2-x1), height*2)
            cr.rectangle(x2, y2, width*2, height*2)
            cr.rectangle(x2, y1, width*2, (y2-y1))
            cr.rectangle(x2, 0, width*2, y1)
            cr.rectangle(x1, 0, (x2-x1), y1)
            
            cr.set_source_rgba(0, 0, 0, 0.25)
            cr.fill()
            
            cr.set_line_width(1)
            cr.set_source_rgb(1, 0, 0)
            
            cr.rectangle(x1, y1, (x2-x1), (y2-y1))
            cr.stroke()
            
        if(self.tabs.get_current_page() == UI.ALIGN_TAB):
            current = self.frameSlider.get_value()
            
            # Area of Interest
            px1 = min(g.areaOfInterestP1[0], g.areaOfInterestP2[0])
            py1 = min(g.areaOfInterestP1[1], g.areaOfInterestP2[1])
            
            px2 = max(g.areaOfInterestP1[0], g.areaOfInterestP2[0])
            py2 = max(g.areaOfInterestP1[1], g.areaOfInterestP2[1])
            
            # Drift Points
            dx1 = g.driftP1[0]
            dy1 = g.driftP1[1]
            
            dx2 = g.driftP2[0]
            dy2 = g.driftP2[1]
            
            dx = 0
            dy = 0
            
            if(dx1 != 0 and dy1 != 0 and
               dx2 != 0 and dy2 != 0):
                dx = dx2 - dx1
                dy = dy2 - dy1
            
            if(px1 != 0 and py1 != 0 and
               px2 != 0 and py2 != 0):
                # Draw Area of Interest Rectangle
                drawRect(cr, px1 + (dx/max(1, g.endFrame - g.startFrame))*(current-g.startFrame), 
                             py1 + (dy/max(1, g.endFrame - g.startFrame))*(current-g.startFrame), 
                             px2 + (dx/max(1, g.endFrame - g.startFrame))*(current-g.startFrame), 
                             py2 + (dy/max(1, g.endFrame - g.startFrame))*(current-g.startFrame))
                
            if(dx1 != 0 and dy1 != 0 and current == g.startFrame):
                # Draw point on first frame
                drawPoint(cr, dx1, dy1)
                
            if(dx1 != 0 and dy1 != 0 and current != g.startFrame and
               dx2 != 0 and dy2 != 0 and current != g.endFrame):
                # Draw interpolated point
                drawPoint(cr, dx1 + (dx/(g.endFrame - g.startFrame))*(current-g.startFrame), 
                              dy1 + (dy/(g.endFrame - g.startFrame))*(current-g.startFrame))
                
            if(dx2 != 0 and dy2 != 0 and current == g.endFrame):
                # Draw point on last frame
                drawPoint(cr, dx2, dy2)
        
    # Updates the Quality Graph of the aligned images    
    def updateQualityImage(self):
        data = []
        if(g.frameSortMethod == Stack.SORT_DIFF):
            data = self.stack.tmats[:,2]
        elif(g.frameSortMethod == Stack.SORT_QUALITY):
            data = self.stack.tmats[:,3]
        elif(g.frameSortMethod == Stack.SORT_BOTH):
            data = self.stack.tmats[:,2] + self.stack.tmats[:,3]
        lenData = len(data)
        
        if(lenData > 1):
            self.qualityImage.show()
            allocation = self.builder.get_object("stackTab").get_allocation()
            
            width = max(250, allocation.width-1)
            padding = 5
            qualityImage = np.full((100+padding*2,width+padding*2, 3), 225, np.uint8)
            bins = np.arange(lenData).reshape(lenData, 1)

            # Draw Axis
            cv2.line(qualityImage, (padding,24+padding), (width+padding,24+padding), (200,200,200))
            cv2.line(qualityImage, (padding,49+padding), (width+padding,49+padding), (200,200,200))
            cv2.line(qualityImage, (padding,74+padding), (width+padding,74+padding), (200,200,200))
            framePercent = (self.frameSlider.get_value())/(lenData-1)
            cv2.line(qualityImage, (math.floor((width-1)*framePercent) + padding, padding), (math.floor((width-1)*framePercent)+padding, 100+padding), (163,163,163), lineType=cv2.LINE_AA)
            
            # Create data points
            data = np.float32(data)
            smoothen = math.ceil(lenData*0.01)
            data = np.pad(data, (smoothen//2, smoothen-smoothen//2), 'reflect')
            data = np.cumsum(data[smoothen:] - data[:-smoothen])
            cv2.normalize(data, data, 0, 99, cv2.NORM_MINMAX)
            data.sort()
            data = np.int32(np.around(np.flip(99 - data)))
            
            pts = np.int32(np.column_stack((bins/((lenData-1)/(width-1)) + padding,data+padding)))

            # Make sure there are no duplicate x-coordinates, and then the y-coordinates
            last = pts[-1]
            ptsx, idx = np.unique(pts[:,0], return_index=True)
            pts = pts[idx]
            ptsy, idx = np.unique(pts[:,1], return_index=True)
            pts = pts[idx]
            if(pts[-1].all() != last.all()):
                pts[-1] = last
            
            # Draw the data point lines
            cv2.polylines(qualityImage, [pts], False, (127,127,127), lineType=cv2.LINE_AA)
            
            # Draw Limit line
            limitPercent = (g.limit-1)/(lenData-1)
            cv2.line(qualityImage, (math.floor((width-1)*limitPercent)+padding,padding), (math.floor((width-1)*limitPercent) + padding, 100+padding), (255,0,0), lineType=cv2.LINE_AA)

            qualityImage = qualityImage[padding:-padding,padding:-padding]

            pixbuf = self.createPixbuf(qualityImage)
            self.qualityImage.set_from_pixbuf(pixbuf.copy())
        else:
            self.qualityImage.hide()
        
    # Updates the PSF Image to the current configuration for deblurring        
    def updatePSFImage(self, *args):
        sz = 110
        psf = np.zeros((sz,sz), np.float32)
        count = 0
        if(g.deconvolveCircular and g.deconvolveCircularDiameter > 1):
            # Circular
            psf += defocus_kernel(int(g.deconvolveCircularDiameter), sz)
            count += 1
        if(g.deconvolveGaussian and g.deconvolveGaussianDiameter > 1):
            # Gaussian
            kern = gaussian_kernel(int(g.deconvolveGaussianDiameter), int(g.deconvolveGaussianSpread), sz)
            psf += cv2.normalize(kern, kern, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
            count += 1
        if(g.deconvolveLinear and g.deconvolveLinearDiameter > 1):
            # Stretch&Rotate the Gaussian/Circular
            psf = stretchPSF(psf, (g.deconvolveLinearDiameter/max(g.deconvolveGaussianDiameter,g.deconvolveCircularDiameter)), g.deconvolveLinearAngle)
        if(count == 0 and g.deconvolveLinear and g.deconvolveLinearDiameter > 1):
            # Linear
            psf += motion_kernel(int(g.deconvolveLinearAngle), int(g.deconvolveLinearDiameter), sz)
            count += 1
        if(g.deconvolveCustom and g.deconvolveCustomFile is not None):
            # Custom
            kern = g.deconvolveCustomFile
            h, w = kern.shape[:2]
            kern = cv2.copyMakeBorder(kern, math.ceil((sz-h)/2), math.floor((sz-h)/2), math.ceil((sz-w)/2), math.floor((sz-w)/2), cv2.BORDER_CONSTANT)
            psf += kern
            count += 1
            
        if(count > 0):
            psf /= count
        else:
            psf[int(sz/2)][int(sz/2)] = 1
        
        psf = cv2.normalize(psf, psf, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)    
        psf = np.uint8(psf*255)
        
        psf = cv2.cvtColor(psf, cv2.COLOR_GRAY2RGB)
        
        pixbuf = self.createPixbuf(psf)
        self.psfImage.set_from_pixbuf(pixbuf.copy())
    
    # Updates the RGB histogram of the final image    
    def updateHistogram(self) -> None:
        allocation = self.builder.get_object("processTab").get_allocation()
        width = max(250, allocation.width-1)
        padding = 5
        histImage = np.full((100+padding*2,width+padding*2, 3), 225, np.uint8)
        bins = np.arange(256).reshape(256, 1)
        
        color = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
        display = ColorSpace.displayImage(self.process.finalImage, self.process.colorSpace)
        for i, col in enumerate(color):
            hist = np.log(cv2.calcHist([display],[i],None,[256],[0,256]) + 1)
            cv2.normalize(hist, hist, 0, 99, cv2.NORM_MINMAX)
            hist = 99 - hist
            hist = np.int32(np.around(hist))
            pts = np.int32(np.column_stack(((bins/(256/width))+padding,hist+padding)))
            cv2.polylines(histImage, [pts], False, col, lineType=cv2.LINE_AA)
            
        histImage = histImage[padding:-padding,padding:-padding]
            
        pixbuf = self.createPixbuf(histImage)
        self.histImage.set_from_pixbuf(pixbuf.copy())
        
    # Sets the reference frame to the current visible frame
    def setReference(self, *args):
        g.reference = str(int(self.frameSlider.get_value()))
        self.reference = self.video.getFrame(g.file, self.video.frames[int(g.reference)], g.actualColor(), lens=g.lens)
        self.builder.get_object("referenceLabel").set_text(g.reference)
        self.builder.get_object("alignButton").set_sensitive(True)
        if(self.tabs.get_current_page() == UI.ALIGN_TAB):
            self.updateImage()
        
    # Updates the progress bar
    def setProgress(self, i=0, total=0, text=""):
        def update():
            if(total == 0):
                self.progressBox.hide()
            else:
                self.progressBox.show()
                self.progress.set_fraction(i/total)
                self.progress.set_text(text + " " + str(round((i/total)*100)) + "%")
        GLib.idle_add(update)
        
    # Sets the start frame for trimming
    def setStartFrame(self, *args):
        g.startFrame = int(self.startFrame.get_value())
        self.endFrame.set_lower(g.startFrame+1)
        self.frameSlider.set_lower(g.startFrame)
        self.frameSlider.set_value(max(g.startFrame, self.frameSlider.get_value()))
        if(int(g.startFrame) > int(g.reference)):
            # Reference is outside of the range, fix it
            g.reference = str(int(g.startFrame))
            self.builder.get_object("referenceLabel").set_text(g.reference)
            self.reference = self.video.getFrame(g.file, self.video.frames[int(g.reference)], g.actualColor(), lens=g.lens)
        self.fixFrameSliderBug()
        if(self.tabs.get_current_page() == UI.ALIGN_TAB):
            self.updateImage()
        
    # Sets the end frame for trimming
    def setEndFrame(self, *args):
        g.endFrame = int(self.endFrame.get_value())
        self.startFrame.set_upper(g.endFrame-1)
        self.frameSlider.set_upper(g.endFrame)
        self.frameSlider.set_value(min(g.endFrame, self.frameSlider.get_value()))
        if(int(g.endFrame) < int(g.reference)):
            # Reference is outside of the range, fix it
            g.reference = str(int(g.endFrame))
            self.builder.get_object("referenceLabel").set_text(g.reference)
            self.reference = self.video.getFrame(g.file, self.video.frames[int(g.reference)], g.actualColor(), lens=g.lens)
        self.fixFrameSliderBug()
        if(self.tabs.get_current_page() == UI.ALIGN_TAB):
            self.updateImage()
        
    # Drift Point 1 Button Clicked
    def clickDriftP1(self, *args):
        self.clickedDriftP1 = False
        self.clickedDriftP2 = False
        self.clickedAreaOfInterest = False
        self.setDriftPoint()
        self.frameSlider.set_value(g.startFrame)
        self.clickedDriftP1 = True
        self.window.get_window().set_cursor(Gdk.Cursor(Gdk.CursorType.CROSSHAIR))
        
    # Drift Point 2 Button Clicked
    def clickDriftP2(self, *args):
        self.clickedDriftP1 = False
        self.clickedDriftP2 = False
        self.clickedAreaOfInterest = False
        self.setDriftPoint()
        self.frameSlider.set_value(g.endFrame)
        self.clickedDriftP2 = True
        self.window.get_window().set_cursor(Gdk.Cursor(Gdk.CursorType.CROSSHAIR))
        
    # Reset Drift Point 1 to (0, 0)
    def resetDriftP1(self, widget, event):
        if(event.button == 3): # Right Click
            g.driftP1 = (0, 0)
            self.clickedDriftP1 = False
            self.clickedDriftP2 = False
            self.clickedAreaOfInterest = False
            self.window.get_window().set_cursor(Gdk.Cursor(Gdk.CursorType.LEFT_PTR))
            self.overlay.queue_draw()
            
    # Reset Drift Point 2 to (0, 0)
    def resetDriftP2(self, widget, event):
        if(event.button == 3): # Right Click
            g.driftP2 = (0, 0)
            self.clickedDriftP1 = False
            self.clickedDriftP2 = False
            self.clickedAreaOfInterest = False
            self.window.get_window().set_cursor(Gdk.Cursor(Gdk.CursorType.LEFT_PTR))
            self.overlay.queue_draw()
    
    # Updates the drift point
    def setDriftPoint(self, *args):
        if(self.clickedDriftP1 or self.clickedDriftP2):
            if(self.clickedDriftP1):
                g.driftP1 = self.mousePosition
            elif(self.clickedDriftP2):
                g.driftP2 = self.mousePosition
            self.clickedDriftP1 = False
            self.clickedDriftP2 = False
            self.window.get_window().set_cursor(Gdk.Cursor(Gdk.CursorType.LEFT_PTR))
            self.overlay.queue_draw()
        
    # Area of Interest button clicked
    def clickAreaOfInterest(self, *args):
        g.areaOfInterestP1 = (0, 0)
        g.areaOfInterestP2 = (0, 0)
        self.clickedDriftP1 = False
        self.clickedDriftP2 = False
        self.clickedAreaOfInterest = True
        self.frameSlider.set_value(g.startFrame)
        self.window.get_window().set_cursor(Gdk.Cursor(Gdk.CursorType.CROSSHAIR))
        self.overlay.queue_draw()
        
    # Reset Area of Interest to (0, 0)
    def resetAreaOfInterest(self, widget, event):
        if(event.button == 3): # Right Click
            g.areaOfInterestP1 = (0, 0)
            g.areaOfInterestP2 = (0, 0)
            self.clickedDriftP1 = False
            self.clickedDriftP2 = False
            self.clickedAreaOfInterest = False
            self.window.get_window().set_cursor(Gdk.Cursor(Gdk.CursorType.LEFT_PTR))
            self.overlay.queue_draw()
        
    # First point int the Area of Interest clicked, drag started
    def dragBegin(self, *args):
        if(self.clickedAreaOfInterest):
            g.areaOfInterestP1 = self.mousePosition
        
    # Mouse released after dragging Area of Interest
    def dragEnd(self, *args):
        if(self.clickedAreaOfInterest):
            g.areaOfInterestP2 = self.mousePosition
            self.clickedAreaOfInterest = False
            self.window.get_window().set_cursor(Gdk.Cursor(Gdk.CursorType.LEFT_PTR))
    
    # Called when the mouse moves over the frame
    def updateMousePosition(self, *args):
        pointer = self.frame.get_pointer()
        self.mousePosition = (min(max(0, pointer.x), self.frame.get_allocation().width), min(max(0, pointer.y), self.frame.get_allocation().height))
        if(self.clickedAreaOfInterest):
            if(g.areaOfInterestP1 != (0, 0)):
                g.areaOfInterestP2 = self.mousePosition
            self.overlay.queue_draw()
    
    # Sets whether or not to normalize the frames during alignment
    def setNormalize(self, *args):
        g.normalize = self.normalize.get_active()
        self.updateImage()
        
    # Sets whether or not to dilate the frames during alignment
    def setDilate(self, *args):
        g.dilate = self.dilate.get_active()
        self.updateImage()
        
    # Sets whether or not to align channels separately
    def setAlignChannels(self, *args):
        g.alignChannels = self.alignChannels.get_active()
        # TCA correction and Align RGB are mutually exclusive
        self.updateLensSettings()
    
    # Sets the color mode of the input
    def setColorMode(self, *args):
        if(self.colorMode.is_sensitive()):
            oldColor = g.actualColor() if self.video is not None else None
            g.colorMode = self.colorMode.get_active()
            if(g.colorMode >= Video.COLOR_RGGB):
                self.preferences.set("bayerGuess", g.colorMode)
            if(oldColor is not None and g.actualColor() != oldColor):
                # Previous results were decoded with the old color mode, so discard them
                self.discardResults()
            self.updateImage()
            
    # Discards the results decoded with old settings (color mode, lens correction)
    def discardResults(self):
        self.process = None
        if(len(self.video.frames) > 0):
            # Video/Sequence
            self.stack = None
            self.reference = self.video.getFrame(g.file, self.video.frames[int(g.reference)], g.actualColor(), lens=g.lens)
            self.builder.get_object("alignTab").set_sensitive(True)
            self.builder.get_object("stackTab").set_sensitive(False)
            self.builder.get_object("processTab").set_sensitive(False)
            self.builder.get_object("exportTab").set_sensitive(False)
            
    # Fills the camera dropdown from the lensfun database, grouped by maker
    def populateLensCameras(self):
        if(len(self.lensCameras) > 0):
            return
        # Filled while detached, appending to a combo's model is very slow
        store = self.lensCamera.get_model()
        self.lensCamera.set_model(None)
        makers = {}
        for camera in sorted(Lens.database().cameras, key=lambda c: (c.maker.lower(), Lens.displayName(c).lower())):
            if(camera.maker not in makers):
                makers[camera.maker] = store.append(None, [camera.maker, -1])
            store.append(makers[camera.maker], [Lens.displayName(camera), len(self.lensCameras)])
            self.lensCameras.append(camera)
        self.lensCamera.set_model(store)
        self.lensCamera.set_active(0)
            
    # Fills the lens dropdown with the lenses that fit the camera (every lens without a camera), plus the extra lens
    def populateLensLenses(self, camera, extra=None):
        # A new store is filled while detached, clearing or appending to a combo's model is very slow
        store = Gtk.TreeStore(str, int)
        self.lensLens.set_model(None)
        store.append(None, ["None", -1])
        lenses = list(Lens.database().find_lenses(camera)) if camera is not None else list(Lens.database().lenses)
        if(extra is not None and not any(Lens.sameItem(lens, extra) for lens in lenses)):
            lenses.append(extra)
        self.lensLenses = []
        makers = {}
        for lens in sorted(lenses, key=lambda l: (l.maker.lower(), l.model.lower())):
            if(lens.maker not in makers):
                makers[lens.maker] = store.append(None, [lens.maker, -1])
            store.append(makers[lens.maker], [lens.model, len(self.lensLenses)])
            self.lensLenses.append(lens)
        self.lensLens.set_model(store)
        self.lensLens.set_active(0)
        self.lensListCamera = camera
        
    # Selects the item (None selects "None") in the camera or lens dropdown
    def selectLensItem(self, combo, items, item):
        index = -1
        for i, other in enumerate(items):
            if(item is not None and Lens.sameItem(other, item)):
                index = i
                break
        def find(iter):
            while(iter is not None):
                if(combo.get_model()[iter][1] == index):
                    return iter
                child = find(combo.get_model().iter_children(iter))
                if(child is not None):
                    return child
                iter = combo.get_model().iter_next(iter)
            return None
        combo.set_active_iter(find(combo.get_model().get_iter_first()))
        
    # Returns the selected item of the camera or lens dropdown, None when there is none
    def selectedLensItem(self, combo, items):
        iter = combo.get_active_iter()
        if(iter is None or combo.get_model()[iter][1] < 0):
            return None
        return items[combo.get_model()[iter][1]]
        
    # Fills in the lens correction from the file's metadata (EXIF or Nikon NCTG), corrections are on when the lens is found
    def prefillLens(self):
        self.lensUpdating = True
        self.populateLensCameras()
        self.lensMetadata = Lens.readMetadata(g.file)
        g.lens.cropRegion = self.lensMetadata.get("cropRegion")
        camera = Lens.findCamera(self.lensMetadata.get("make"), self.lensMetadata.get("model"))
        lens = Lens.findLens(camera, self.lensMetadata.get("lensMake"), self.lensMetadata.get("lens"))
        self.selectLensItem(self.lensCamera, self.lensCameras, camera)
        self.populateLensLenses(camera, lens)
        self.selectLensItem(self.lensLens, self.lensLenses, lens)
        self.lensFocal.set_value(self.lensMetadata.get("focal", 0))
        self.lensAperture.set_value(self.lensMetadata.get("aperture", 0))
        found = (lens is not None)
        self.lensDistortion.set_active(found)
        self.lensTCA.set_active(found and not self.alignChannels.get_active())
        self.lensVignetting.set_active(found)
        self.lensUpdating = False
        self.updateLensSettings()
        
    # Reads the lens correction widgets into g.lens, updates which corrections can be used and the status
    def updateLensSettings(self):
        self.lensUpdating = True
        camera = self.selectedLensItem(self.lensCamera, self.lensCameras)
        if(not Lens.sameItem(camera, self.lensListCamera)):
            # The lenses depend on the camera, keep the selected lens
            lens = self.selectedLensItem(self.lensLens, self.lensLenses)
            self.populateLensLenses(camera, lens)
            self.selectLensItem(self.lensLens, self.lensLenses, lens)
        lens = self.selectedLensItem(self.lensLens, self.lensLenses)
        g.lens.cameraMaker = camera.maker if camera is not None else None
        g.lens.cameraModel = camera.model if camera is not None else None
        g.lens.lensMaker = lens.maker if lens is not None else None
        g.lens.lensModel = lens.model if lens is not None else None
        g.lens.focal = self.lensFocal.get_value()
        g.lens.aperture = self.lensAperture.get_value()
        
        status = []
        geometry = True
        if(lens is None):
            if(self.lensMetadata.get("lens") is not None):
                status.append("Lens \"" + self.lensMetadata["lens"] + "\" not found")
            else:
                status.append("No lens selected")
        else:
            if(camera is None):
                if(self.lensMetadata.get("model") is not None):
                    status.append("Camera \"" + self.lensMetadata["model"] + "\" not found")
                status.append("Assuming the crop factor of the lens calibration (" + str(round(lens.crop_factor, 2)) + ")")
            if(self.imageSize is not None):
                crop, geometry = Lens.imageCropFactor(lens, camera, g.lens.cropRegion, *self.imageSize)
                status.append("Image crop factor " + str(round(crop, 2)))
            if(not geometry):
                status.append("The crop isn't centered on the sensor, so distortion and TCA can't be corrected")
            if(g.lens.focal <= 0 or g.lens.aperture <= 0):
                status.append("Enter the focal length and aperture")
        self.lensStatus.set_text("\n".join(status))
        
        self.lensDistortion.set_sensitive(lens is not None and geometry and len(lens.calib_distortion) > 0)
        self.lensTCA.set_sensitive(lens is not None and geometry and len(lens.calib_tca) > 0 and not self.alignChannels.get_active())
        self.lensVignetting.set_sensitive(lens is not None and len(lens.calib_vignetting) > 0)
        g.lens.distortion = self.lensDistortion.get_sensitive() and self.lensDistortion.get_active()
        g.lens.tca = self.lensTCA.get_sensitive() and self.lensTCA.get_active()
        g.lens.vignetting = self.lensVignetting.get_sensitive() and self.lensVignetting.get_active()
        self.alignChannels.set_sensitive(not g.lens.tca)
        self.lensUpdating = False
            
    # Sets the lens correction, results with a different correction are discarded
    def setLensCorrection(self, *args):
        if(self.lensUpdating):
            return
        oldKey = g.lens.key()
        self.updateLensSettings()
        if(self.video is not None and g.file is not None and g.lens.key() != oldKey):
            self.discardResults()
            self.updateImage()
    
    def setDriftType(self, *args):
        g.driftType = self.driftType.get_active()
        if(g.driftType == Align.DRIFT_MANUAL):
            self.builder.get_object("selectDriftPoint1").show()
            self.builder.get_object("selectDriftPoint2").show()
        else:
            g.driftP1 = (0, 0)
            g.driftP2 = (0, 0)
            self.builder.get_object("selectDriftPoint1").hide()
            self.builder.get_object("selectDriftPoint2").hide()
        self.updateImage()
    
    # Sets the type of transformation
    def setTransformation(self, *args):
        text = self.transformation.get_active_text()
        if(text == "None"):
            g.transformation = -1
        elif(text == "Translation"):
            g.transformation = Align.TRANSLATION
        elif(text == "Rigid Body"):
            g.transformation = Align.RIGID_BODY
        elif(text == "Scaled Rotation"):
            g.transformation = Align.SCALED_ROTATION
        elif(text == "Affine"):
            g.transformation = Align.AFFINE
            
    # Sets the drizzle scaling factor
    def setDrizzleFactor(self, *args):
        text = self.drizzleFactor.get_active_text()
        if(text == "0.25X"):
            g.drizzleFactor = 0.25
        elif(text == "0.50X"):
            g.drizzleFactor = 0.50
        elif(text == "0.75X"):
            g.drizzleFactor = 0.75
        elif(text == "1.0X"):
            g.drizzleFactor = 1.0
        elif(text == "1.5X"):
            g.drizzleFactor = 1.5
        elif(text == "2.0X"):
            g.drizzleFactor = 2.0
        elif(text == "2.5X"):
            g.drizzleFactor = 2.5
        elif(text == "3.0X"):
            g.drizzleFactor = 3.0
        if(self.stack is not None):
            self.stack.generateRefBG()
        self.updateImage()
            
    # Sets the drizzle scaling factor
    def setDrizzleInterpolation(self, *args):
        text = self.drizzleInterpolation.get_active_text()
        if(text == "Nearest Neighbor"):
            g.drizzleInterpolation = 0 # cv2.INTER_NEAREST
        elif(text == "Bilinear"):
            g.drizzleInterpolation = 1 # cv2.INTER_LINEAR
        elif(text == "Bicubic"):
            g.drizzleInterpolation = 2 # cv2.INTER_CUBIC
        elif(text == "Lanczos"):
            g.drizzleInterpolation = 4 # cv2.INTER_LANCZOS4
        if(self.stack is not None):
            self.stack.generateRefBG()
        self.updateImage()
        
    # Sets the frame sort method
    def setFrameSortMethod(self, *args):
        g.frameSortMethod = self.frameSort.get_active()
        if(self.stack is not None):
            self.stack.sortTmats()
            self.updateImage()
        
    # Sets whether or not to auto crop
    def setAutoCrop(self, *args):
        g.autoCrop = not self.autoCrop.get_active()
        self.updateImage()
            
    # Runs the Alignment step
    def clickAlign(self, *args):
        self.align = Align(self.video.frames[g.startFrame:g.endFrame+1])
        thread = Thread(target=self.align.run, args=())
        thread.start()
        self.disableUI()
        
    # Kills all pool processes
    def killPool(self):
        for pid in self.pids:
            if(psutil.pid_exists(pid)):
                p = psutil.Process(pid)
                p.kill()
        
    # Stops the current action being performed
    def stopProcessing(self, *args):
        self.killPool()
        g.pool = None
        self.setThreads()
        self.setProgress()
        self.enableUI()
        
    # Called when the Alignment is complete
    def finishedAlign(self):
        def update():
            self.stack = Stack(self.align.tmats)
            self.tabs.next_page()
            self.enableUI()
            self.builder.get_object("alignTab").set_sensitive(True)
            self.builder.get_object("stackTab").set_sensitive(True)
            self.builder.get_object("processTab").set_sensitive(False)
            self.builder.get_object("exportTab").set_sensitive(False)
            self.limit.set_upper(len(self.stack.tmats))
            self.limit.set_value(int(len(self.stack.tmats)/2))
            self.limitPercent.set_value(round(self.limit.get_value()/len(self.stack.tmats)*100))
            self.setLimit()
            self.setLimitPercent()
        GLib.idle_add(update)
        
    # Sets the number of frames to use in the Stack
    def setLimit(self, *args):
        self.limitPercent.disconnect(self.limitPercentSignal)
        self.limit.set_upper(len(self.stack.tmats))
        g.limit = int(self.limit.get_value())
        self.limitPercent.set_value(round(g.limit/len(self.stack.tmats)*100))
        self.limitPercentSignal = self.limitPercent.connect("value-changed", self.setLimitPercent)
        self.updateQualityImage()
        
    # Sets the number of frames to use in the Stack
    def setLimitPercent(self, *args):
        limitPercent = self.limitPercent.get_value()/100
        self.limit.set_value(round(limitPercent*len(self.stack.tmats)))
            
    # Stack Button clicked
    def clickStack(self, *args):
        try:
            self.stack.checkMemory()
            thread = Thread(target=self.stack.run, args=())
            thread.start()
            self.disableUI()
        except MemoryError as error:
            self.enableUI()
        
    # Called when the stack is complete
    def finishedStack(self) -> None:
        def update() -> None:
            self.process = Process(self.stack.stackedImage)
            self.showLevels()
            self.tabs.next_page()
            self.enableUI()
            self.builder.get_object("alignTab").set_sensitive(True)
            self.builder.get_object("stackTab").set_sensitive(True)
            self.builder.get_object("processTab").set_sensitive(True)
            self.builder.get_object("exportTab").set_sensitive(True)
        GLib.idle_add(update)
        
    # Processes the final Stacked image
    def processImage(self, *args: object) -> None:
        if(g.file is None or self.updatingLevels):
            return
        self.processSpinner.start()
        self.processSpinner.show()

        for i in range(0,5):
            g.sharpen[i] = (self.builder.get_object("sharpen" + str(i+1)).get_value())
            g.radius[i] = (self.builder.get_object("radius" + str(i+1)).get_value())
            g.denoise[i] = (self.builder.get_object("denoise" + str(i+1)).get_value())
            g.level[i] = (self.builder.get_object("level" + str(i+1)).get_active())
        
        g.deconvolveCircular = self.builder.get_object("deconvolveCircular").get_active()
        g.deconvolveGaussian = self.builder.get_object("deconvolveGaussian").get_active()
        g.deconvolveLinear = self.builder.get_object("deconvolveLinear").get_active()
        g.deconvolveCustom = self.builder.get_object("deconvolveCustom").get_active()
        
        g.deconvolveCircularDiameter = self.builder.get_object("deconvolveCircularDiameter").get_value()
        g.deconvolveCircularAmount = self.builder.get_object("deconvolveCircularAmount").get_value()
        g.deconvolveGaussianDiameter = self.builder.get_object("deconvolveGaussianDiameter").get_value()
        g.deconvolveGaussianAmount = self.builder.get_object("deconvolveGaussianAmount").get_value()
        g.deconvolveGaussianSpread = self.builder.get_object("deconvolveGaussianSpread").get_value()
        g.deconvolveLinearDiameter = self.builder.get_object("deconvolveLinearDiameter").get_value()
        g.deconvolveLinearAmount = self.builder.get_object("deconvolveLinearAmount").get_value()
        g.deconvolveLinearAngle = self.builder.get_object("deconvolveLinearAngle").get_value()
        g.deconvolveCustomAmount = self.builder.get_object("deconvolveCustomAmount").get_value()
        
        g.showAdaptive = self.builder.get_object("showAdaptive").get_active()
        g.showDark = self.builder.get_object("showDark").get_active()
        g.showBright = self.builder.get_object("showBright").get_active()
        g.deringAdaptive = int(self.builder.get_object("deringAdaptiveAdjust").get_value())
        g.deringDark = int(self.builder.get_object("deringDarkAdjust").get_value())
        g.deringBright = int(self.builder.get_object("deringBrightAdjust").get_value())
        g.deringSize = int(self.builder.get_object("deringSizeAdjust").get_value())
        g.deringBlend = int(self.builder.get_object("deringBlendAdjust").get_value())
        
        g.gamma = self.builder.get_object("gammaAdjust").get_value()
        self.readLevels()
        
        g.redAdjust = self.builder.get_object("redAdjust").get_value()
        g.greenAdjust = self.builder.get_object("greenAdjust").get_value()
        g.blueAdjust = self.builder.get_object("blueAdjust").get_value()
        g.saturation = self.builder.get_object("saturationAdjust").get_value()
        
        g.stretchFunction = self.builder.get_object("stretchFunction").get_active()
        self.updateExportNotice()
        
        created = False
        if(self.process is None):
            if(self.stack is not None):
                self.process = Process(self.stack.stackedImage)
            else:
                self.process = Process(g.file, True)
            self.showLevels()
            created = True
        
        processLevels = self.process.processLevelsAgain
        if(created):
            # Everything after the levels (applied when it was built) still needs to be processed
            processAgain = True
            processDeblur = False
            processDering = False
            processColor = False
        elif(len(args) > 0 and (self.blackLevel == args[0] or
                              self.exposure == args[0] or
                              self.autoBlackLevel == args[0] or
                              self.autoExposure == args[0])):
            # Black Level and Exposure precede the wavelets
            processAgain = False
            processDeblur = False
            processDering = False
            processColor = False
            processLevels = True
        elif(len(args) > 0 and (self.builder.get_object("gammaAdjust") == args[0] or
                                self.builder.get_object("stretchFunction") == args[0] or
                                self.builder.get_object("redAdjust") == args[0] or
                                self.builder.get_object("greenAdjust") == args[0] or
                                self.builder.get_object("blueAdjust") == args[0] or
                                self.builder.get_object("saturationAdjust") == args[0])):
            processAgain = self.process.processAgain
            processDeblur = self.process.processDeblurAgain
            processDering = self.process.processDeringAgain
            processColor = True
        elif(len(args) > 0 and (self.builder.get_object("deringAdaptiveAdjust") == args[0] or
                                self.builder.get_object("deringDarkAdjust") == args[0] or
                                self.builder.get_object("deringBrightAdjust") == args[0] or
                                self.builder.get_object("deringSizeAdjust") == args[0] or
                                self.builder.get_object("deringBlendAdjust") == args[0] or
                                self.builder.get_object("showAdaptive") == args[0] or
                                self.builder.get_object("showDark") == args[0] or
                                self.builder.get_object("showBright") == args[0])):
            processAgain = self.process.processAgain
            processDeblur = self.process.processDeblurAgain
            processDering = True
            processColor = False
        elif(len(args) > 0 and (self.builder.get_object("deconvolveCircular") == args[0] or
                                self.builder.get_object("deconvolveGaussian") == args[0] or
                                self.builder.get_object("deconvolveLinear") == args[0] or
                                self.builder.get_object("deconvolveCustom") == args[0] or
                                self.builder.get_object("deconvolveCircularDiameter") == args[0] or
                                self.builder.get_object("deconvolveCircularAmount") == args[0] or
                                self.builder.get_object("deconvolveGaussianDiameter") == args[0] or
                                self.builder.get_object("deconvolveGaussianAmount") == args[0] or
                                self.builder.get_object("deconvolveGaussianSpread") == args[0] or
                                self.builder.get_object("deconvolveLinearDiameter") == args[0] or
                                self.builder.get_object("deconvolveLinearAmount") == args[0] or
                                self.builder.get_object("deconvolveLinearAngle") == args[0] or
                                self.builder.get_object("deconvolveCustomFile") == args[0] or
                                self.builder.get_object("deconvolveCustomAmount") == args[0])):
            processAgain = self.process.processAgain
            processDeblur = True
            processDering = False
            processColor = False
        else:
            processAgain = True
            processDeblur = False
            processDering = False
            processColor = False
        
        if(self.processThread != None and self.processThread.is_alive()):
            self.process.processLevelsAgain = processLevels
            self.process.processAgain = processAgain
            self.process.processDeblurAgain = processDeblur
            self.process.processDeringAgain = processDering
            self.process.processColorAgain = processColor
        else:
            self.processThread = Thread(target=self.process.run,
                                        kwargs={"processAgain": processAgain,
                                                "processDeblurAgain": processDeblur,
                                                "processDeringAgain": processDering,
                                                "processColorAgain": processColor,
                                                "processLevelsAgain": processLevels})
            self.processThread.start()
        self.updatePSFImage(args)
    
    def readLevels(self) -> None:
        """Reads the Black Level and Exposure controls into the globals."""
        g.autoBlackLevel = self.autoBlackLevel.get_active()
        g.autoExposure = self.autoExposure.get_active()
        g.blackLevel = blackLevelFromSlider(self.blackLevel.get_value())
        g.exposure = self.exposure.get_value()
        
    def showLevels(self) -> None:
        """Shows the black level and exposure computed for the automatic toggles on their sliders."""
        if(self.process is None):
            return
        (blackLevel, exposure) = self.process.levels()
        self.updatingLevels = True
        if(g.autoBlackLevel):
            self.blackLevel.set_value(sliderFromBlackLevel(blackLevel))
        if(g.autoExposure):
            self.exposure.set_value(exposure)
        self.updatingLevels = False
        
    def setAutoLevels(self, *args: object) -> None:
        """Makes the sliders insensitive while their automatic toggle is on, and reprocesses when a toggle changed.

        Args:
            *args: Signal arguments, the toggled check button first; empty when called during setup.
        """
        self.builder.get_object("blackLevel").set_sensitive(not self.autoBlackLevel.get_active())
        self.builder.get_object("exposure").set_sensitive(not self.autoExposure.get_active())
        if(len(args) > 0):
            self.processImage(*args)
        
    def formatBlackLevel(self, scale: Gtk.Scale, value: float) -> str:
        """Formats the Black Level slider value as a percentage of full scale.

        Args:
            scale: The Black Level slider.
            value: Slider value, log10 of the percentage of full scale.

        Returns:
            The percentage shown next to the slider.
        """
        blackLevel = blackLevelFromSlider(value)*100
        if(blackLevel == 0):
            return "0%"
        elif(blackLevel < 1):
            return "%.3f%%" % blackLevel
        return "%.1f%%" % blackLevel
        
    def updateExportNotice(self, *args: object) -> None:
        """Shows the notice that float exports aren't linear when Stretch or Gamma is applied.

        Args:
            *args: Signal arguments, unused.
        """
        stretch = self.builder.get_object("stretchFunction").get_active()
        gamma = self.builder.get_object("gammaAdjust").get_value()
        floatFormat = self.getPixelFormat() in (Export.FORMAT_16FLOAT, Export.FORMAT_32FLOAT)
        self.nonLinearNotice.set_visible(floatFormat and (stretch != Process.STRETCH_NONE or max(1, gamma) != 100))
        
    # Called when processing is complete
    def finishedProcessing(self, *args: object) -> None:
        def update() -> None:
            if(self.process is None):
                # Discarded while processing (color mode or lens correction changed)
                self.processSpinner.stop()
                self.processSpinner.hide()
                return
            if(self.showMask.get_active() and self.process.thresh is not None):
                pixbuf = self.createPixbuf(np.around(self.process.thresh).astype('uint8'))
            else:
                pixbuf = self.createPixbuf(self.process.finalImage, self.process.colorSpace)
            self.frame.set_from_pixbuf(pixbuf.copy())
            self.showLevels()
            self.updateHistogram()
            self.processSpinner.stop()
            self.processSpinner.hide()
        GLib.idle_add(update)

    # Opens the preferences dialog
    def openPreferences(self, *args):
        self.preferencesDialog.run()
        self.preferencesDialog.hide()

    # Closes the application
    def close(self, *args):
        self.killPool()
        Gtk.main_quit()

# Used to initialize pids
def dummy(*args):
    return True

def run():
    if('SNAP' in os.environ):
        UI.SNAP_DIR = os.environ['SNAP'] + '/'
        UI.HOME_DIR = os.environ['SNAP_REAL_HOME']
        os.environ['HOME'] = UI.HOME_DIR # This might be bad, but it works
    else:
        # Newer versions of Adwaita scalable icons don't work well with older librsvg.
        # This can be removed when no longer being built with an older librsvg
        Gtk.IconTheme.get_default().prepend_search_path('share_override/icons')
    
    g.ui = UI()

    Gtk.main()
