from __future__ import annotations
from lazy import lazy
cv2 = lazy("cv2")
np = lazy("numpy")
rawpy = lazy("rawpy")
av = lazy("av")
from abc import ABC, abstractmethod
from typing import NamedTuple
from natsort import natsorted, ns
import ColorSpace

class Frame(NamedTuple):
    """A frame as stored, and how to linearize it."""
    image: np.ndarray
    params: ColorSpace.DecodeParams
    workingSpace: ColorSpace.WorkingSpace

class FrameSource(ABC):
    """A sequence of frames stored in one or more files, addressed by integer index.

    Sources are sent to worker processes, so they must pickle as just what identifies their
    frames and open files lazily wherever they are read.
    """

    @classmethod
    def open(cls, path: str | list[str]) -> FrameSource:
        """Opens the chosen file or files as the one kind of source that can open them.

        Args:
            path: A file, or a list of files in any order.

        Returns:
            The source.

        Raises:
            ValueError: If no file was chosen or no kind of source can open the files; the
                message is meant for the user.
        """
        if(len(path) == 0):
            raise ValueError("No file was chosen.")
        for source in (ImageSequenceSource, VideoSource):
            if(source.canOpen(path)):
                return source(path)
        raise ValueError("The chosen files can't be opened. Choose one video, or one or more images.")

    @staticmethod
    @abstractmethod
    def canOpen(path: str | list[str]) -> bool:
        """Returns whether this kind of source can open the chosen file or files.

        Files are recognized by their contents rather than their extensions. Kinds of source
        don't overlap, so at most one can open any choice.

        Args:
            path: A file, or a list of files in any order.

        Returns:
            Whether this kind of source can open them.
        """

    @abstractmethod
    def filePaths(self) -> list[str]:
        """Returns the files the source reads.

        Returns:
            The files, in frame order.
        """

    @abstractmethod
    def frameCount(self) -> int:
        """Returns the number of frames in the source.

        Returns:
            The number of frames; indices run from 0.
        """

    @abstractmethod
    def readFrame(self, index: int) -> Frame | None:
        """Reads one frame as stored, with its own description.

        Args:
            index: The frame's index.

        Returns:
            The frame in BGR order, or with a single channel for gray or Bayer mosaic frames,
            at its stored depth (uint8 or uint16, or float32 for float FITS data); or None if
            the source has no readable frame at index.
        """

    @abstractmethod
    def frameSize(self) -> tuple[int, int]:
        """Returns the size of the frames.

        Returns:
            The (width, height) of the first frame.
        """

    def fastPreview(self) -> Frame | None:
        """Reads a preview of the first frame, if the file offers a quick way to.

        Return None unless this is much quicker than reading the first frame; the caller then
        makes the preview from readFrame(0) itself.

        Returns:
            The first frame like readFrame returns it, possibly at a reduced size; or None.
        """
        return None

    def description(self) -> tuple[ColorSpace.DecodeParams, ColorSpace.WorkingSpace]:
        """Returns how to linearize every frame of the source.

        Every frame is linearized like the first one, so the images of a sequence that are
        tagged differently are all decoded alike.

        Returns:
            The first frame's decode parameters and working space.
        """
        frame = self.readFrame(0)
        return (frame.params, frame.workingSpace)

    def recordedBits(self) -> int | None:
        """Returns the bit depth the frames were recorded at, when readFrame widens them.

        Returns:
            The recorded depth, or None if frames are read at the depth they were recorded at.
        """
        return None

    def close(self) -> None:
        """Releases any open file handles. The source reopens them if it is read again."""

class ImageSequenceSource(FrameSource):
    """A list of image files, one frame per file."""

    def __init__(self, path: str | list[str]) -> None:
        """Initializes the source.

        Args:
            path: An image file, or a list of image files (in any order).
        """
        self.paths = [path] if isinstance(path, str) else natsorted(path, alg=ns.IGNORECASE)

    @staticmethod
    def canOpen(path: str | list[str]) -> bool:
        """Accepts an image file, or a list of one or more image files.

        Each file must be recognized, without decoding its pixels, by one of the readers
        Video.decodeImage decodes with, in its order: a raw camera file, an image format OpenCV
        reads, then a FITS file with image data.
        """
        paths = [path] if isinstance(path, str) else path
        for file in paths:
            try:
                with rawpy.imread(file):
                    continue
            except Exception:
                pass
            if(cv2.haveImageReader(file)):
                continue
            try:
                from astropy.io import fits
                with fits.open(file) as hdus:
                    if(any(hdu.header.get("NAXIS", 0) > 0 for hdu in hdus)):
                        continue
            except Exception:
                pass
            return False
        return len(paths) > 0

    def filePaths(self) -> list[str]:
        return self.paths

    def frameCount(self) -> int:
        return len(self.paths)

    def readFrame(self, index: int) -> Frame | None:
        """Each file is described by its own color tags."""
        from Video import Video
        if(index >= len(self.paths)):
            return None
        return Frame(*Video.decodeImage(self.paths[index]))

    def fastPreview(self) -> Frame | None:
        """A raw camera file is decoded at half size; other files have none."""
        # TODO: Read a raw camera file's embedded JPEG preview (rawpy's extract_thumb) instead,
        # falling back to the half size decode for files without one. It is far quicker, and it
        # keeps LibRaw's OpenMP code out of the main process.
        from Video import Video
        try:
            rawpy.imread(self.paths[0]).close()
        except Exception:
            # Not a raw camera file
            return None
        return Frame(*Video.decodeImage(file=self.paths[0],
                                        thumbnail=True))

    def frameSize(self) -> tuple[int, int]:
        height, width = self.readFrame(0).image.shape[:2]
        return (width, height)

class VideoSource(FrameSource):
    """A video file decoded with PyAV, described by its stream's color tags.

    Frames are indexed by their position in presentation order and found by seeking to their
    timestamps. Reading a frame less than a second after the last one read decodes forward
    instead of seeking. The timestamps are read from the video's packets, without decoding,
    the first time they are needed, and pickled so worker processes don't read them again.
    """

    def __init__(self, path: str) -> None:
        """Initializes the source.

        Args:
            path: The video file.
        """
        self.path = path
        self.timestamps: list[int] | None = None
        self.container: av.container.InputContainer | None = None

    def __getstate__(self) -> dict[str, str | list[int] | None]:
        """Returns the state to pickle: the path and the timestamps, without the open container."""
        return {"path": self.path, "timestamps": self.timestamps}

    def __setstate__(self, state: dict[str, str | list[int] | None]) -> None:
        """Restores a pickled source, which opens its container when first read.

        Args:
            state: The state from __getstate__.
        """
        self.path = state["path"]
        self.timestamps = state["timestamps"]
        self.container = None

    @staticmethod
    def canOpen(path: str | list[str]) -> bool:
        """Accepts a single file PyAV demuxes at least two video frames from.

        One-frame videos are rejected, and left to other FrameSource implementations to handle.
        """
        if(not isinstance(path, str)):
            return False
        try:
            with av.open(path) as container:
                stream = container.streams.video[0]
                packets = 0
                for packet in container.demux(stream):
                    packets += packet.size > 0
                    if(packets >= 2):
                        return True
        except Exception:
            pass
        return False

    def filePaths(self) -> list[str]:
        return [self.path]

    def openContainer(self) -> None:
        """Opens the video, if needed, and describes it from its stream's color tags."""
        if(self.container is not None):
            return
        self.container = av.open(self.path)
        self.stream = self.container.streams.video[0]
        codec = self.stream.codec_context
        self.width, self.height = codec.width, codec.height
        self.videoBits = 8
        formatName = ""
        isYUV = True
        if(codec.format is not None):
            self.videoBits = codec.format.components[0].bits
            formatName = codec.format.name
            isYUV = not codec.format.is_rgb and len(codec.format.components) >= 3
        self.params, self.workingSpace = ColorSpace.fromVideoTags(primaries=codec.color_primaries,
                                                                  transfer=codec.color_trc,
                                                                  matrix=codec.colorspace,
                                                                  colorRange=codec.color_range,
                                                                  isYUV=isYUV,
                                                                  fullRangeFormat=formatName.startswith("yuvj"))
        self.decoder = None
        self.lastTimestamp = None
        self.lastFrame = None

    def frameTimestamps(self) -> list[int]:
        """Returns the presentation timestamps of the frames, read from the packets the first time.

        Returns:
            The sorted timestamps, in the stream's time base.
        """
        if(self.timestamps is None):
            self.openContainer()
            timestamps = set()
            for packet in self.container.demux(self.stream):
                timestamp = packet.pts if packet.pts is not None else packet.dts
                if(timestamp is not None and packet.size > 0):
                    timestamps.add(timestamp)
            self.decoder = None
            self.timestamps = sorted(timestamps)
        return self.timestamps

    def frameCount(self) -> int:
        """The count is the number of distinct timestamps among the video's packets."""
        return len(self.frameTimestamps())

    def frameToImage(self, videoFrame: av.VideoFrame) -> np.ndarray:
        """Converts a decoded frame to full range RGB with the stream's YUV -> RGB matrix and range.

        swscale only upsamples the chroma, at the source's bit depth; the range expansion and the
        matrix are applied here, since swscale's 16-bit RGB output expands limited range
        inaccurately.

        Args:
            videoFrame: The decoded frame.

        Returns:
            The full range uint16 BGR image.
        """
        if(self.params.yuvMatrix is not None):
            sourceBits = max(8, min(16, videoFrame.format.components[0].bits))
            yuvFormat = "yuv444p" if sourceBits == 8 else "yuv444p" + str(sourceBits) + "le"
            yuvFrame = videoFrame.reformat(format=yuvFormat)
            dtype = np.uint8 if sourceBits == 8 else np.uint16
            planes = [np.frombuffer(plane, dtype).reshape(yuvFrame.height, -1)[:, :yuvFrame.width]
                      for plane in yuvFrame.planes]
            rgb = ColorSpace.yuvToRGB(yuv=np.stack(planes),
                                      bits=sourceBits,
                                      matrix=self.params.yuvMatrix,
                                      colorRange=self.params.range)
            image = ColorSpace.quantize(img=rgb, bits=16)
        else:
            # Converted at the source's bit depth and widened exactly, so untagged RGB/gray loads unchanged
            sourceBits = 8 if videoFrame.format.components[0].bits <= 8 else 16
            image = videoFrame.reformat(format="rgb24" if sourceBits == 8 else "rgb48le").to_ndarray()
            if(sourceBits == 8):
                image = image.astype(np.uint16)*257
            if(self.params.range == "limited"):
                scaled = (image.astype(np.float32)/65535*255 - 16)/219
                image = np.around(np.clip(scaled, 0, 1)*65535).astype(np.uint16)
        return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    def readFrame(self, index: int) -> Frame | None:
        """Frames are converted to 16-bit BGR, whatever the stream's depth."""
        timestamps = self.frameTimestamps()
        if(index >= len(timestamps)):
            return None
        self.openContainer()
        target = timestamps[index]
        if(self.lastFrame is None or target != self.lastTimestamp):
            if(self.decoder is None or self.lastTimestamp is None or target < self.lastTimestamp or
               target - self.lastTimestamp > 1/self.stream.time_base):
                self.container.seek(target, stream=self.stream, backward=True)
                self.decoder = self.container.decode(self.stream)
            self.lastFrame = None
            for videoFrame in self.decoder:
                timestamp = videoFrame.pts if videoFrame.pts is not None else videoFrame.dts
                if(timestamp is not None and timestamp >= target):
                    self.lastTimestamp = timestamp
                    self.lastFrame = videoFrame
                    break
            if(self.lastFrame is None):
                self.decoder = None
                return None
        if(self.lastTimestamp != target):
            # The frame couldn't be decoded
            return None
        return Frame(image=self.frameToImage(self.lastFrame),
                     params=self.params,
                     workingSpace=self.workingSpace)

    def description(self) -> tuple[ColorSpace.DecodeParams, ColorSpace.WorkingSpace]:
        """The stream's color tags describe every frame, so nothing is decoded."""
        self.openContainer()
        return (self.params, self.workingSpace)

    def recordedBits(self) -> int | None:
        self.openContainer()
        return 8 if self.videoBits <= 8 else None

    def frameSize(self) -> tuple[int, int]:
        self.openContainer()
        return (self.width, self.height)

    def close(self) -> None:
        if(self.container is not None):
            self.container.close()
            self.container = None

