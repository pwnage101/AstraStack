from lazy import lazy
cv2 = lazy("cv2")
np = lazy("numpy")
import math
from concurrent.futures.process import BrokenProcessPool
from Globals import *
from WorkerPools import FRAME_POOL
from ProgressBar import *
from FrameSource import FrameSource, ImageSequenceSource
from FrameSink import FrameSink
import ColorSpace

class Video:

    SAVE_EXTENSIONS = FrameSink.EXTENSIONS

    def __init__(self, source: FrameSource) -> None:
        self.source = source
        # Source indices of the frames kept on load; wrong-size or unreadable frames are left out
        self.frames: list[int] = []
        self.sharpest = 0
        self.sharps = []
        self.deltas = []
        self.descriptionMismatch = False
        # The working space of the last frames read
        self.workingSpace: ColorSpace.WorkingSpace | None = None

    @staticmethod
    def open(paths: str | list[str]) -> "Video":
        """Opens files as a video.

        Args:
            paths: A file, or the files of an image sequence.

        Returns:
            The video.

        Raises:
            ValueError: If the files can't be opened.
        """
        return Video(FrameSource.open(paths))

    # Checks to see if there will be enough memory to process the image
    def checkMemory(self) -> None:
        width, height = self.source.frameSize()
        if(not g.ui.checkMemory(w=width, h=height)):
            raise MemoryError()

    # Loads the frames of the source, calculating their sharpness and frame to frame deltas
    def run(self) -> None:
        progress = ProgressBar()
        futures = []
        try:
            progress.total = self.source.frameCount()
            countPerThread = math.ceil(progress.total/g.nThreads)

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

    def getFrame(self, index: int) -> np.ndarray:
        """Reads a frame of the source and converts it to linear working data.

        Every frame is linearized like the first one. The working space of the frames is kept in self.workingSpace.

        Args:
            index: The frame's index in the source.

        Returns:
            The linear 0-1 float32 BGR image.
        """
        self.workingSpace = self.source.frameWorkingSpace()
        return self.source.readFrame(index).image

    def outputWorkingSpace(self) -> ColorSpace.WorkingSpace:
        """Returns the working space of the working data, which is demosaiced as the source's demosaic params say."""
        return self.source.frameWorkingSpace()

    def displayImage(self, img: np.ndarray) -> np.ndarray:
        """Returns working data as a display-referred, quantized sRGB image.

        Args:
            img: Linear 0-1 RGB image.

        Returns:
            The sRGB uint8 image.
        """
        return ColorSpace.displayImage(img=img, workingSpace=self.outputWorkingSpace())

    def displayEncode(self, img: np.ndarray) -> np.ndarray:
        """Returns working data as display-referred sRGB values, for previews that edit them.

        Args:
            img: Linear 0-1 RGB image.

        Returns:
            The sRGB-encoded float32 image, clipped to 0-1.
        """
        return ColorSpace.displayEncode(img=img, workingSpace=self.outputWorkingSpace())

    def save(self, path: str, img: np.ndarray, bits: int) -> None:
        """Saves working data as an image file.

        Args:
            path: The file to write; its extension must be one of SAVE_EXTENSIONS.
            img: Linear 0-1 RGB image.
            bits: 8 or 16.
        """
        FrameSink(path=path, bits=bits).write(img=img, workingSpace=self.outputWorkingSpace())

    @staticmethod
    def measurementCopy(image: np.ndarray) -> np.ndarray:
        """Returns the gamma 2.2 copy of working data that measurements are made on.

        The curve evens out photon noise and keeps bright discs from dominating the fits. It is applied
        to each working channel, without a primaries conversion, and isn't display encoding.

        Args:
            image: Linear 0-1 image.

        Returns:
            The 0-1 float32 image, clipped to 0-1 first.
        """
        return np.clip(image, 0, 1).astype(np.float32)**(1/2.2)

    @staticmethod
    def encodedCopy(image: np.ndarray) -> np.ndarray:
        """Returns the uint8 measurement copy of linear working data, for sharpness and duplicate-frame checks.

        Args:
            image: Linear 0-1 BGR image.

        Returns:
            The gamma 2.2 uint8 BGR image.
        """
        return np.around(Video.measurementCopy(image)*255).astype(np.uint8)

    @staticmethod
    def alignImage(image: np.ndarray) -> np.ndarray:
        """Returns the gamma 2.2 grayscale copy of linear working data that alignment works on.

        Args:
            image: Linear 0-1 BGR image.

        Returns:
            The 0-1 float32 grayscale image.
        """
        return cv2.cvtColor(Video.measurementCopy(image), cv2.COLOR_BGR2GRAY)

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
               progress: ProgressCounter) -> tuple[list[int], list[float], list[np.ndarray], bool]:
    """Reads a chunk of frames, calculating each one's sharpness and its delta from the previous frame.

    Sharpness and deltas are measured on an gamma 2.2 uint8 copy of the linear working data,
    so their thresholds keep their meaning. Runs in a worker process.

    Args:
        source: The source to read from, demosaiced as its demosaic params say before calculating sharpness and deltas.
        indices: The consecutive frame indices of the chunk. The frame before the chunk is read
            too, as the previous frame for the first delta.
        frameSize: The (width, height) frames must have; frames of any other size are skipped.
        progress: Counts the frames as they are read.

    Returns:
        The indices of the loaded frames, their sharpness and their deltas, and whether any
        loaded frame is described differently from the source.
    """
    frames = []
    sharps = []
    deltas = []
    mismatch = False

    prevImage = None
    if(indices.start > 0 and len(indices) > 0):
        prevFrame = source.readFrame(indices.start - 1, atRecordedDepth=True)
        if(prevFrame is not None):
            prevImage = cv2.cvtColor(Video.encodedCopy(prevFrame.image), cv2.COLOR_BGR2GRAY)
            prevImage, wScaleFactor, hScaleFactor = resize(prevImage)

    for i, index in enumerate(indices):
        progress.count(i, len(indices))
        frame = source.readFrame(index, atRecordedDepth=True)
        if(frame is None):
            continue
        h, w = frame.image.shape[:2]
        if((w, h) != frameSize):
            # Only add if dimensions match
            continue
        frames.append(index)
        mismatch |= frame.mismatch
        image = Video.encodedCopy(frame.image)

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
        psf = source.readStored(0).image
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
