from lazy import lazy
cv2 = lazy("cv2")
np = lazy("numpy")
rawpy = lazy("rawpy")
import math
from concurrent.futures.process import BrokenProcessPool
from Globals import *
from WorkerPools import FRAME_POOL
from ProgressBar import *
from FrameSource import FrameSource, ImageSequenceSource
import ColorSpace

class Video:

    COLOR_AUTO = 0
    COLOR_RGB = 1
    COLOR_GRAYSCALE = 2
    COLOR_RGGB = 3
    COLOR_GRBG = 4
    COLOR_GBRG = 5
    COLOR_BGGR = 6
    COLOR_RGGB_VNG = 7
    COLOR_GRBG_VNG = 8
    COLOR_GBRG_VNG = 9
    COLOR_BGGR_VNG = 10

    def __init__(self, source: FrameSource) -> None:
        self.source = source
        # Source indices of the frames kept on load; wrong-size or unreadable frames are left out
        self.frames: list[int] = []
        self.sharpest = 0
        self.sharps = []
        self.deltas = []
        self.sourceDescription: tuple[ColorSpace.DecodeParams, ColorSpace.WorkingSpace] | None = None
        self.descriptionMismatch = False
        # Decode parameters and working space of the last frame linearized
        self.decodeParams: ColorSpace.DecodeParams | None = None
        self.workingSpace: ColorSpace.WorkingSpace | None = None

    @staticmethod
    def decodeImage(file: str,
                    thumbnail: bool = False) -> tuple[np.ndarray, ColorSpace.DecodeParams, ColorSpace.WorkingSpace]:
        """Reads an image file's stored values and describes how to linearize them.

        Camera RAW files are decoded by LibRaw in camera RGB, with no curve, no white balance and their black level
        still in. Other images are read unchanged by OpenCV (without an alpha channel), and FITS files by astropy.

        Args:
            file: Path of the image file.
            thumbnail: Whether a half size decode is enough (RAW only).

        Returns:
            The stored image (BGR, or gray), its decode parameters and its working space.
        """
        raw = None
        try:
            raw = rawpy.imread(file)
        except Exception:
            pass
        if(raw is not None):
            with raw:
                img = raw.postprocess(half_size=thumbnail,
                                      no_auto_scale=True,
                                      user_black=0,
                                      user_wb=[1, 1, 1, 1],
                                      output_color=rawpy.ColorSpace.raw,
                                      gamma=(1, 1),
                                      no_auto_bright=True,
                                      output_bps=16)
                params, workingSpace = ColorSpace.fromRAW(raw)
            return (cv2.cvtColor(img, cv2.COLOR_RGB2BGR), params, workingSpace)
        img = cv2.imread(file, cv2.IMREAD_UNCHANGED)
        if(img is not None):
            if(len(img.shape) == 3 and img.shape[2] == 4):
                img = img[:, :, :3]
            params, workingSpace = ColorSpace.describeImage(file, channels=1 if len(img.shape) == 2 else 3)
            return (img, params, workingSpace)
        # Image might be a FITS image so give that a try
        from astropy.io import fits
        with fits.open(file) as hdus:
            hdu = next(hdu for hdu in hdus if hdu.data is not None)
            img = np.asarray(hdu.data)
            header = hdu.header
            if(img.ndim == 3):
                # Color planes (RGB) first
                img = cv2.cvtColor(np.ascontiguousarray(np.moveaxis(img[:3], 0, -1)), cv2.COLOR_RGB2BGR)
            params = ColorSpace.fromFITS(header=header, data=img)
        if(img.dtype not in (np.uint8, np.uint16)):
            img = img.astype(np.float32)
        return (img, params, ColorSpace.UNTAGGED)

    # Checks to see if there will be enough memory to process the image
    def checkMemory(self) -> None:
        width, height = self.source.frameSize()
        if(not g.ui.checkMemory(w=width, h=height)):
            raise MemoryError()

    def description(self) -> tuple[ColorSpace.DecodeParams, ColorSpace.WorkingSpace]:
        """Returns how every frame of the source is linearized, asking the source the first time.

        Returns:
            The source's decode parameters and working space.
        """
        if(self.sourceDescription is None):
            self.sourceDescription = self.source.description()
        return self.sourceDescription

    # Loads the frames of the source, calculating their sharpness and frame to frame deltas
    def run(self) -> None:
        progress = ProgressBar()
        futures = []
        try:
            progress.total = self.source.frameCount()
            countPerThread = math.ceil(progress.total/g.nThreads)

            g.decodeParams, g.workingSpace = self.description()
            g.guessedColorMode = self.guessColorMode()

            progress.setMessage("Preprocessing Frames")
            frameSize = self.source.frameSize()
            # Release the main process's file handles while the workers read; each worker opens its own
            self.source.close()
            for i in range(0, g.nThreads):
                indices = range(i*countPerThread, (i+1)*countPerThread)
                futures.append(FRAME_POOL.submit(loadFrames,
                                                 source=self.source,
                                                 indices=indices,
                                                 frameSize=frameSize,
                                                 colorMode=g.actualColor(),
                                                 description=self.description(),
                                                 progress=ProgressCounter(progress.counter(i), g.nThreads)))
            for i in range(0, g.nThreads):
                frames, sharps, deltas, mismatch = futures[i].result()
                self.frames += frames
                self.sharps += sharps
                self.deltas += deltas
                self.descriptionMismatch |= mismatch
        except BrokenProcessPool:
            progress.stop()
            return

        if(len(self.sharps) > 0):
            self.sharpest = self.sharps.index(max(self.sharps))
        else:
            self.sharpest = 0

        progress.stop()
        g.ui.finishedVideo()

    def skippedCount(self) -> int:
        """Returns how many of the source's frames were left out on load."""
        return self.source.frameCount() - len(self.frames)

    def getFrame(self,
                 index: int,
                 colorMode: int) -> np.ndarray:
        """Reads a frame of the source and converts it to linear working data.

        Every frame is linearized like the first one. The frame's decode parameters and working
        space, adjusted to the color mode, are kept in self.decodeParams and self.workingSpace.

        Args:
            index: The frame's index in the source.
            colorMode: The color mode the frame is in; COLOR_AUTO resolves it from the source
                and the frame.

        Returns:
            The linear 0-1 float32 BGR image.
        """
        params, workingSpace = self.description()
        return self.linearizeFrame(image=self.source.readFrame(index).image,
                                   params=params,
                                   workingSpace=workingSpace,
                                   colorMode=colorMode)

    def thumbnail(self) -> np.ndarray:
        """Makes a thumbnail of the source.

        The thumbnail's working space is kept in self.workingSpace, like getFrame's.

        Returns:
            The source's fast preview if it has one, or else its first frame; in the Auto color
            mode, as linear 0-1 float32 BGR.
        """
        frame = self.source.fastPreview()
        if frame is None:
            frame = self.source.readFrame(0)
        return self.linearizeFrame(image=frame.image,
                                   params=frame.params,
                                   workingSpace=frame.workingSpace,
                                   colorMode=Video.COLOR_AUTO)

    def linearizeFrame(self,
                       image: np.ndarray,
                       params: ColorSpace.DecodeParams,
                       workingSpace: ColorSpace.WorkingSpace,
                       colorMode: int) -> np.ndarray:
        """Applies the color mode to a frame's stored values and converts them to linear working data.

        The frame's decode parameters and working space, adjusted to the color mode, are kept in
        self.decodeParams and self.workingSpace.

        Args:
            image: The frame as the source read it.
            params: The frame's decode parameters.
            workingSpace: The frame's working space.
            colorMode: The color mode to apply; COLOR_AUTO resolves it from the source and the
                frame.

        Returns:
            The linear 0-1 float32 BGR image.
        """
        if(colorMode == Video.COLOR_AUTO):
            colorMode = self.guessColorMode(image=image,
                                            params=params)
        if(colorMode >= Video.COLOR_RGGB and image.dtype == np.uint16 and self.source.recordedBits() == 8):
            # Demosaic 8-bit video as 8-bit (VNG only works on uint8)
            image = np.around(image/257).astype(np.uint8)
        image, params, workingSpace = Video.applyColorMode(image, colorMode, params, workingSpace)
        self.decodeParams = params
        self.workingSpace = workingSpace
        return ColorSpace.linearize(img=image, params=params, bgr=True)

    @staticmethod
    def applyColorMode(image: np.ndarray,
                       colorMode: int,
                       params: ColorSpace.DecodeParams,
                       workingSpace: ColorSpace.WorkingSpace) -> tuple[np.ndarray,
                                                                       ColorSpace.DecodeParams,
                                                                       ColorSpace.WorkingSpace]:
        """Applies the color mode to stored values, and adjusts their description to match.

        Demosaiced data is sensor data: it has no curve and no primaries, and unless the file states its levels,
        black is 0 and white is the type's maximum. Other color modes keep the description.

        Args:
            image: The stored image (BGR, or gray).
            colorMode: The color mode, not COLOR_AUTO.
            params: The stored values' decode parameters.
            workingSpace: The stored values' working space.

        Returns:
            The image with 3 channels (BGR), its decode parameters and its working space.
        """
        if(colorMode < Video.COLOR_RGGB):
            return (Video._applyColorMode(img=image, colorMode=colorMode), params, workingSpace)
        if(params.whiteLevel is None):
            params = ColorSpace.DecodeParams(blackLevel=(0.0, 0.0, 0.0),
                                             whiteLevel=float(np.iinfo(image.dtype).max))
        elif(image.dtype not in (np.uint8, np.uint16)):
            # OpenCV only demosaics integers
            scale = 65535/params.whiteLevel
            image = np.around(np.clip(image*scale, 0, 65535)).astype(np.uint16)
            params = ColorSpace.DecodeParams(blackLevel=tuple(float(v)*scale for v in params.blackLevel),
                                             whiteLevel=65535.0)
        else:
            params = ColorSpace.DecodeParams(blackLevel=params.blackLevel, whiteLevel=params.whiteLevel)
        return (Video._applyColorMode(img=image, colorMode=colorMode), params, ColorSpace.UNTAGGED)

    @staticmethod
    def encodedCopy(image: np.ndarray, workingSpace: ColorSpace.WorkingSpace | None) -> np.ndarray:
        """Returns an sRGB-encoded uint8 copy of linear working data, for measurements made on display values.

        Args:
            image: Linear 0-1 BGR image.
            workingSpace: The working data's colors.

        Returns:
            The display-encoded uint8 BGR image.
        """
        return ColorSpace.displayImage(img=image[:, :, ::-1], workingSpace=workingSpace)[:, :, ::-1].copy()

    @staticmethod
    def alignImage(image: np.ndarray, workingSpace: ColorSpace.WorkingSpace | None) -> np.ndarray:
        """Returns the sRGB-encoded grayscale copy of linear working data that alignment works on.

        The curve roughly evens out photon noise before StackReg's squared-difference fit.

        Args:
            image: Linear 0-1 BGR image.
            workingSpace: The working data's colors.

        Returns:
            The display-encoded 0-1 float32 grayscale image.
        """
        encoded = ColorSpace.displayEncode(img=np.ascontiguousarray(image[:, :, ::-1]), workingSpace=workingSpace)
        return cv2.cvtColor(encoded, cv2.COLOR_RGB2GRAY)

    def guessColorMode(self,
                       image: np.ndarray | None = None,
                       params: ColorSpace.DecodeParams | None = None) -> int:
        """Resolves the Auto color mode for the source.

        Args:
            image: A frame of the source to guess from; the first frame is read if None.
            params: The frame's decode parameters; the source's are used if None.

        Returns:
            The color mode for the Bayer pattern the source states, or else the one guessed
            from the frame's pixels.
        """
        if(params is None):
            params = self.description()[0]
        if(params.bayerPattern is not None):
            return BAYER_COLOR_MODES[params.bayerPattern]
        if(image is None):
            frame = self.source.readFrame(0)
            if(frame is None):
                return Video.COLOR_RGB
            image = frame.image
        return self.guessImageColor(image)

    def guessImageColor(self, image: np.ndarray) -> int:
        """Guesses a frame's color mode from its pixels.

        A very simple guess, far from perfect: a frame with equal channels is grayscale unless
        demosaicing it reveals a Bayer pattern. The pattern is only told apart as RGGB/BGGR or
        GRBG/GBRG, and the user's last chosen Bayer mode replaces it if there is one.

        Args:
            image: A frame as FrameSource.readFrame returns it.

        Returns:
            The guessed color mode.
        """
        # Assume its color to start with
        colorMode = Video.COLOR_RGB
        rgb = Video._bgrFromSourceFrame(image, Video.COLOR_RGB)
        if(np.array_equal(rgb[:,:,0],rgb[:,:,1]) and
           np.array_equal(rgb[:,:,1],rgb[:,:,2])):
            # Not a color image
            colorMode = Video.COLOR_GRAYSCALE

            # Calculate all bayer options
            rggb = Video._bgrFromSourceFrame(image, Video.COLOR_RGGB)
            grbg = Video._bgrFromSourceFrame(image, Video.COLOR_GRBG)
            gbrg = Video._bgrFromSourceFrame(image, Video.COLOR_GBRG)
            bggr = Video._bgrFromSourceFrame(image, Video.COLOR_BGGR)

            diff1 = np.average(cv2.absdiff(cv2.cvtColor(rggb, cv2.COLOR_BGR2GRAY), cv2.cvtColor(bggr, cv2.COLOR_BGR2GRAY)))
            diff2 = np.average(cv2.absdiff(cv2.cvtColor(grbg, cv2.COLOR_BGR2GRAY), cv2.cvtColor(gbrg, cv2.COLOR_BGR2GRAY)))
            diff = 0
            if(diff1 > 0.0001 and diff2 > 0.0001):
                diff = abs(diff1 - diff2)/min(diff1, diff2)
            if(diff > 0.2):
                # Probably uses a bayer pattern, but which one...
                rggbDiff = np.average(cv2.absdiff(rggb[:,:,0], rggb[:,:,2]))
                grbgDiff = np.average(cv2.absdiff(grbg[:,:,0], grbg[:,:,2]))
                if(rggbDiff >= grbgDiff):
                    # Probably either RGGB or BGGR
                    colorMode = Video.COLOR_RGGB
                else:
                    # Probably either GRBG or GBRG
                    colorMode = Video.COLOR_GRBG
                colorMode = g.ui.preferences.get("bayerGuess", colorMode)
        return colorMode

    @staticmethod
    def _bgrFromSourceFrame(
        image: np.ndarray,
        colorMode: int,
        toDtype: type | None = None,
    ) -> np.ndarray:
        """Converts a frame read from a source to BGR, interpreting it by its color mode.

        The color mode is applied at the frame's depth, then the depth is converted.

        Args:
            image: A frame as FrameSource.readFrame returns it.
            colorMode: The color mode the frame is in. COLOR_AUTO isn't supported; resolve it
                with Video.guessColorMode first.
            toDtype: The output's dtype: np.float32 for a 0-1 scale, np.uint8 for 8 bits
                (16-bit frames keep their high byte), or None to keep the input's dtype.

        Returns:
            The frame as BGR.

        Raises:
            ValueError: If colorMode is COLOR_AUTO, or the frame's dtype or toDtype isn't supported.
        """
        if colorMode == Video.COLOR_AUTO:
            raise ValueError("COLOR_AUTO must be resolved before converting a frame")
        if image.dtype not in (np.uint8, np.uint16):
            raise ValueError(f"Unsupported frame dtype: {image.dtype}")
        if toDtype not in (None, np.float32, np.uint8):
            raise ValueError(f"Unsupported toDtype: {toDtype}")
        image = Video._applyColorMode(img=image, colorMode=colorMode)
        if toDtype == np.float32:
            return image.astype(np.float32) / np.iinfo(image.dtype).max
        if toDtype == np.uint8 and image.dtype == np.uint16:
            return (image >> 8).astype(np.uint8)
        return image

    @staticmethod
    def _applyColorMode(img: np.ndarray, colorMode: int) -> np.ndarray:
        """Demosaics or grays a frame by its color mode, keeping its depth.

        VNG demosaicing is only done on uint8 frames; deeper frames are demosaiced edge-aware.

        Args:
            img: A frame as FrameSource.readFrame returns it.
            colorMode: The color mode the frame is in; not COLOR_AUTO.

        Returns:
            The frame as BGR.
        """
        h, w = img.shape[:2]
        if(len(img.shape) == 2):
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        if(colorMode == Video.COLOR_RGB):
            pass
        elif(colorMode == Video.COLOR_GRAYSCALE):
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            if(colorMode >= Video.COLOR_RGGB_VNG and img.dtype == np.uint8): # Only do VNG if its a uint8 image
                img = cv2.copyMakeBorder(img, 4, 2, 2, 2, cv2.BORDER_REPLICATE)
                if(colorMode == Video.COLOR_RGGB_VNG):
                    img = cv2.cvtColor(img, cv2.COLOR_BAYER_BG2BGR_VNG)
                elif(colorMode == Video.COLOR_GRBG_VNG):
                    img = cv2.cvtColor(img, cv2.COLOR_BAYER_GB2BGR_VNG)
                elif(colorMode == Video.COLOR_GBRG_VNG):
                    img = cv2.cvtColor(img, cv2.COLOR_BAYER_GR2BGR_VNG)
                elif(colorMode == Video.COLOR_BGGR_VNG):
                    img = cv2.cvtColor(img, cv2.COLOR_BAYER_RG2BGR_VNG)
                img = img[2:h+2,2:w+2]
            else:
                img = cv2.copyMakeBorder(img, 2, 2, 2, 2, cv2.BORDER_REPLICATE)
                if(colorMode == Video.COLOR_RGGB or colorMode == Video.COLOR_RGGB_VNG):
                    img = cv2.cvtColor(img, cv2.COLOR_BAYER_BG2BGR_EA)
                elif(colorMode == Video.COLOR_GRBG or colorMode == Video.COLOR_GRBG_VNG):
                    img = cv2.cvtColor(img, cv2.COLOR_BAYER_GB2BGR_EA)
                elif(colorMode == Video.COLOR_GBRG or colorMode == Video.COLOR_GBRG_VNG):
                    img = cv2.cvtColor(img, cv2.COLOR_BAYER_GR2BGR_EA)
                elif(colorMode == Video.COLOR_BGGR or colorMode == Video.COLOR_BGGR_VNG):
                    img = cv2.cvtColor(img, cv2.COLOR_BAYER_RG2BGR_EA)
                img = img[2:h+2,2:w+2]
        return img

# Resize image so it does things faster
def resize(image):
    from Align import dilateImg
    image = dilateImg(image, 9, 3)
    h, w = image.shape[:2]
    hScaleFactor = min(1.0, (100/h))
    wScaleFactor = (int(w*hScaleFactor))/w
    image = cv2.resize(image, (int(w*hScaleFactor), int(h*hScaleFactor)), interpolation=cv2.INTER_LINEAR)
    return (image, wScaleFactor, hScaleFactor)

# Calculates frame to frame an approximate translation matrix
def calculateDelta(image, prevImage, wScaleFactor, hScaleFactor):
    if(prevImage is not None):
        try:
            T = np.eye(2, 3, dtype=np.float32)
            criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10,  1e-10)
            (cc, T) = cv2.findTransformECC(image, prevImage, T, cv2.MOTION_TRANSLATION, criteria)

            M = np.identity(3, dtype=np.float64)
            M[0][2] = T[0][2]/wScaleFactor # X
            M[1][2] = T[1][2]/hScaleFactor # Y
        except Exception:
            M = np.identity(3)
    else:
        M = np.identity(3)
    return M

# Returns a number based on how sharp the image is (higher = sharper)
def calculateSharpness(image):
    h, w = image.shape[:2]
    return cv2.Laplacian(cv2.resize(image, (int(max(100, w*0.1)), int(max(100, h*0.1)))), cv2.CV_8U).var()

def loadFrames(source: FrameSource,
               indices: range,
               frameSize: tuple[int, int],
               colorMode: int,
               description: tuple[ColorSpace.DecodeParams, ColorSpace.WorkingSpace],
               progress: ProgressCounter) -> tuple[list[int], list[float], list[np.ndarray], bool]:
    """Reads a chunk of frames, calculating each one's sharpness and its delta from the previous frame.

    Sharpness and deltas are measured on an sRGB-encoded uint8 copy of the linear working data,
    so their thresholds keep their meaning. Runs in a worker process.

    Args:
        source: The source to read from.
        indices: The consecutive frame indices of the chunk. The frame before the chunk is read
            too, as the previous frame for the first delta.
        frameSize: The (width, height) frames must have; frames of any other size are skipped.
        colorMode: The color mode to apply before calculating sharpness and deltas.
        description: How every frame is linearized, the source's description.
        progress: Counts the frames as they are read.

    Returns:
        The indices of the loaded frames, their sharpness and their deltas, and whether any
        loaded frame is described differently from the source.
    """
    video = Video(source)
    frames = []
    sharps = []
    deltas = []
    mismatch = False

    def encodedFrame(image: np.ndarray) -> np.ndarray:
        """Returns the sRGB-encoded uint8 copy of a frame, linearized like the source's first one."""
        if(source.recordedBits() == 8):
            # Measured at the depth it was recorded at, like the thresholds were tuned on
            image = np.around(image/257).astype(np.uint8)
        linear = video.linearizeFrame(image=image,
                                      params=description[0],
                                      workingSpace=description[1],
                                      colorMode=colorMode)
        return Video.encodedCopy(linear, video.workingSpace)

    prevImage = None
    if(indices.start > 0 and len(indices) > 0):
        prevFrame = source.readFrame(indices.start - 1)
        if(prevFrame is not None):
            prevImage = cv2.cvtColor(encodedFrame(prevFrame.image), cv2.COLOR_BGR2GRAY)
            prevImage, wScaleFactor, hScaleFactor = resize(prevImage)

    for i, index in enumerate(indices):
        progress.count(i, len(indices))
        frame = source.readFrame(index)
        if(frame is None):
            continue
        h, w = frame.image.shape[:2]
        if((w, h) != frameSize):
            # Only add if dimensions match
            continue
        frames.append(index)
        mismatch |= (frame.params, frame.workingSpace) != description
        image = encodedFrame(frame.image)

        # Calculate sharpness
        sharps.append(calculateSharpness(image))

        # Calculate Frame Deltas
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image, wScaleFactor, hScaleFactor = resize(image)
        M = calculateDelta(image=image,
                           prevImage=prevImage,
                           wScaleFactor=wScaleFactor,
                           hScaleFactor=hScaleFactor)
        deltas.append(M)

        prevImage = image
    progress.countExtra()
    return (frames, sharps, deltas, mismatch)


def loadPsf(path: str) -> np.ndarray:
    """Reads an image file's stored values as a custom PSF.

    Args:
        path: The image file.

    Returns:
        The PSF as grayscale float32, stretched to 0-1.

    Raises:
        ValueError: If the image is larger than 100x100 or has only one value.
    """
    source = ImageSequenceSource(path)
    try:
        psf = source.readFrame(0).image
    finally:
        source.close()
    if psf.ndim == 3:
        psf = cv2.cvtColor(psf, cv2.COLOR_BGR2GRAY)
    psf = psf.astype(np.float32)
    if max(psf.shape) > 100:
        raise ValueError("The PSF must be no larger than 100x100")
    if len(np.unique(psf)) <= 1:
        raise ValueError("The PSF must have more than one value")
    cv2.normalize(psf, psf, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
    return psf


# Color modes for the Bayer patterns sources state. A stop-gap until color modes are reworked.
BAYER_COLOR_MODES: dict[ColorSpace.BayerPattern, int] = {
    "RGGB": Video.COLOR_RGGB,
    "GRBG": Video.COLOR_GRBG,
    "GBRG": Video.COLOR_GBRG,
    "BGGR": Video.COLOR_BGGR,
}
