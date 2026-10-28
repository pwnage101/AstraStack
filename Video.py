from lazy import lazy
cv2 = lazy("cv2")
np = lazy("numpy")
rawpy = lazy("rawpy")
av = lazy("av")
import math
from concurrent.futures.process import BrokenProcessPool
from Globals import *
from ProgressBar import *
import ColorSpace
import Lens

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

    def __init__(self):
        self.frames = []
        self.sharpest = 0
        self.sharps = []
        self.deltas = []
        self.colorSpace = None
        self.colorSpaceMismatch = False
        self.container = None
        
    @staticmethod
    # Reads an image file
    def readImage(file, flags=cv2.IMREAD_COLOR, dtype=np.uint16, thumbnail=False):
        return Video.decodeImage(file, flags, dtype, thumbnail, describe=False)[0]
        
    @staticmethod
    # Reads an image file and its color description (None when it has none or describe is False)
    def decodeImage(file, flags=cv2.IMREAD_COLOR, dtype=np.uint16, thumbnail=False, describe=True):
        img = None
        try:
            # First see if its a raw file, decoded linear in camera RGB
            with rawpy.imread(file) as raw:
                img = raw.postprocess(half_size=thumbnail, output_color=rawpy.ColorSpace.raw, gamma=(1, 1),
                                      no_auto_bright=True, use_camera_wb=True, output_bps=16)
                colorSpace = ColorSpace.fromRAW(raw) if describe else None
        except Exception:
            pass
        if(img is not None):
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            if(dtype == np.uint8):
                img = (img >> 8).astype(np.uint8)
            return (img, colorSpace)
        img = cv2.imread(file, flags)
        if(img is not None):
            # Normal img
            if(describe):
                return ColorSpace.describeImage(file, img)
            return (img, None)
        else:
            # Image might be a FITS image so give that a try
            from astropy.io import fits
            img = fits.getdata(file)
            maxVal = 1
            try:
                maxVal = np.iinfo(img.dtype).max
            except:
                pass
            img = (img.astype(np.float32)/maxVal)*np.iinfo(dtype).max
            cv2.normalize(img, img, alpha=0, beta=np.iinfo(dtype).max, norm_type=cv2.NORM_MINMAX)
            img = img.astype(dtype)
            if(flags == cv2.IMREAD_COLOR):
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        return (img, None)
        
    # Opens the video with PyAV
    def openVideo(self, file):
        self.container = av.open(file)
        self.stream = self.container.streams.video[0]
        codec = self.stream.codec_context
        self.width, self.height = codec.width, codec.height
        self.videoBits = 8
        if(codec.format is not None):
            self.videoBits = codec.format.components[0].bits
        self.videoColorSpace = ColorSpace.fromCICP(codec.color_primaries, codec.color_trc)
        self.decoder = None
        self.lastPts = None
        self.lastFrame = None
        
    # Returns the time (ms) of the pts
    def videoTime(self, pts):
        return float(pts*self.stream.time_base*1000)
        
    # Returns the times (ms) of the video's frames, read from the packets so nothing is decoded
    def videoFrameTimes(self):
        pts = set()
        for packet in self.container.demux(self.stream):
            p = packet.pts if packet.pts is not None else packet.dts
            if(p is not None and packet.size > 0):
                pts.add(p)
        self.decoder = None
        return [self.videoTime(p) for p in sorted(pts)]
        
    # Converts the decoded frame to BGR (rgb24 = 8-bit, rgb48le = 16-bit) with the stream's YUV -> RGB matrix and range
    def videoFrameToImage(self, videoFrame, format):
        matrix = {1: "ITU709", 4: "FCC", 5: "ITU601", 6: "ITU601", 7: "SMPTE240M", 9: "BT2020", 10: "BT2020"}.get(videoFrame.colorspace)
        if(matrix is None):
            # Untagged, guess like most players
            matrix = "ITU709" if videoFrame.height >= 720 else "ITU601"
        colorRange = {1: "MPEG", 2: "JPEG"}.get(videoFrame.color_range)
        if(colorRange is None):
            # Untagged, YUV is usually limited range
            isYUV = (not videoFrame.format.is_rgb and len(videoFrame.format.components) >= 3)
            colorRange = "MPEG" if isYUV and not videoFrame.format.name.startswith("yuvj") else "JPEG"
        # One thread, swscale's thread pool deadlocks in processes forked after it was used
        image = videoFrame.reformat(format=format, src_colorspace=matrix, src_color_range=colorRange, dst_color_range="JPEG", threads=1).to_ndarray()
        return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
    # Decodes the video frame at the time (ms) or the first one after it, returns it as BGR and the video's color description
    def readVideoFrame(self, file, time, format="rgb48le"):
        if(self.container is None):
            self.openVideo(file)
        target = int(round(time/1000/self.stream.time_base))
        if(self.lastFrame is None or target != self.lastPts):
            if(self.decoder is None or self.lastPts is None or target < self.lastPts or 
               target - self.lastPts > 1/self.stream.time_base):
                # Seek unless the frame is less than a second ahead
                self.container.seek(target, stream=self.stream, backward=True)
                self.decoder = self.container.decode(self.stream)
            self.lastFrame = None
            for videoFrame in self.decoder:
                pts = videoFrame.pts if videoFrame.pts is not None else videoFrame.dts
                if(pts is not None and pts >= target):
                    self.lastPts = pts
                    self.lastFrame = videoFrame
                    break
            if(self.lastFrame is None):
                self.decoder = None
                raise EOFError("No video frame at " + str(time) + " ms")
        self.lastTime = self.videoTime(self.lastPts)
        return (self.videoFrameToImage(self.lastFrame, format), self.videoColorSpace)
            
    # Checks to see if there will be enough memory to process the image
    def checkMemory(self):
        if(isinstance(g.file, list)):
            # Image Sequence
            h, w = Video.readImage(g.file[0]).shape[:2]
            if(not g.ui.checkMemory(w, h)):
                raise MemoryError()
        else:
            # Video
            with av.open(g.file) as container:
                codec = container.streams.video[0].codec_context
                width, height = codec.width, codec.height
            if(not g.ui.checkMemory(w=width,h=height)):
                raise MemoryError()

    # Returns a list of file paths for the frames of the given video fileName
    def run(self):
        from natsort import natsorted, ns
        progress = ProgressBar()
        futures = []
        try:
            if(isinstance(g.file, list)):
                # Image Sequence
                progress.total = len(g.file)
                countPerThread = math.ceil(progress.total/g.nThreads)
                g.file = natsorted(g.file, alg=ns.IGNORECASE)
                
                g.guessedColorMode = self.guessColorMode(g.file)
                
                progress.setMessage("Preprocessing Frames")
                image, g.colorSpace = Video.decodeImage(g.file[0])
                g.colorSpaceFromImage = True
                height, width = image.shape[:2]
                self.width, self.height = width, height
                spaces = []
                for i in range(0, g.nThreads):
                    futures.append(g.pool.submit(loadFramesSequence, g.file[max(0, (i*countPerThread)-1):(i+1)*countPerThread], i, width, height, g.actualColor(), ProgressCounter(progress.counter(i), g.nThreads)))
                for i in range(0, g.nThreads):
                    frames, sharp, deltas, colorSpaces = futures[i].result()
                    self.frames += frames
                    self.sharps += sharp
                    self.deltas += deltas
                    spaces += colorSpaces
                    
                # The first image's color space is used for all of them
                self.colorSpaceMismatch = any(not ColorSpace.same(colorSpace, g.colorSpace) for colorSpace in spaces)
            else:
                # Video
                self.openVideo(g.file)
                times = self.videoFrameTimes()
                progress.total = len(times)
                g.colorSpace = self.videoColorSpace
                g.colorSpaceFromImage = False

                g.guessedColorMode = self.guessColorMode(g.file)
                
                progress.setMessage("Preprocessing Frames")
                countPerThread = math.ceil(progress.total/g.nThreads)
                for i in range(0, g.nThreads):
                    start = i*countPerThread
                    previous = times[start-1] if start > 0 and start <= len(times) else None
                    futures.append(g.pool.submit(loadFramesVideo, g.file, times[start:start+countPerThread], previous, g.actualColor(), ProgressCounter(progress.counter(i), g.nThreads)))
                for i in range(0, g.nThreads):
                    frames, sharp, deltas = futures[i].result()
                    self.frames += frames
                    self.sharps += sharp
                    self.deltas += deltas
                
        except BrokenProcessPool:
            progress.stop()
            return

        if(len(self.sharps) > 0):
            self.sharpest = self.sharps.index(max(self.sharps))
        else:
            self.sharpest = 0

        progress.stop()
        g.ui.finishedVideo()
        
    # Returns the frame as linear float32 (0-1, BGR), its color description is stored in self.colorSpace
    # lens (Lens.LensCorrection) corrects the lens distortion, TCA and vignetting
    def getFrame(self, file, frame, colorMode, dtype=np.uint16, thumbnail=False, lens=None):
        if(isinstance(frame, str) or isinstance(file, list)):
            # Specific file or Image Sequence
            if(not isinstance(frame, str)):
                frame = file[frame]
            image, colorSpace = Video.decodeImage(frame, cv2.IMREAD_UNCHANGED, dtype, thumbnail)
            if(colorMode == Video.COLOR_AUTO):
                colorMode = self.guessColorMode(file, image=image)
        else:
            # Video
            image, colorSpace = self.readVideoFrame(file, frame)
            if(colorMode == Video.COLOR_AUTO):
                colorMode = self.guessColorMode(file, image=image)
            if(colorMode >= Video.COLOR_RGGB and self.videoBits <= 8):
                # Debayer 8-bit video as 8-bit (VNG only works on uint8)
                image = (image >> 8).astype(np.uint8)
        image = Video.colorMode(image, colorMode)
        # Linearize and scale to 0-1, only color images and grayscale image files keep their color description
        if(colorMode != Video.COLOR_RGB and
           not (colorMode == Video.COLOR_GRAYSCALE and (isinstance(frame, str) or isinstance(file, list)))):
            colorSpace = None
        if(colorSpace is not None):
            image = colorSpace.linearizeInteger(image, bgr=True)
        else:
            image = image.astype(np.float32)/np.iinfo(image.dtype).max
        image = Lens.correct(image, lens)
        self.colorSpace = colorSpace
        return image
        
    # Returns a display encoded (0-255) copy of the frame (0-1, BGR) for alignment, 8-bit for video like the decoded frames
    # Frames without a color description are used as decoded
    @staticmethod
    def alignImage(image, file, colorSpace):
        if(colorSpace is None):
            image = np.clip(image, 0, 1)*255
        else:
            image = cv2.cvtColor(ColorSpace.displayEncode(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), colorSpace)*255, cv2.COLOR_RGB2BGR)
        if(not isinstance(file, list)):
            # Video
            image = np.around(image).astype(np.uint8)
        return image
      
    def guessColorMode(self, file, image=None):
        colorMode = Video.COLOR_RGB
        if(image is None):
            
            if(isinstance(file, list)):
                # Image Sequence
                image = Video.readImage(file[0], cv2.IMREAD_UNCHANGED)
            else:
                try:
                    # First try as image
                    image = Video.readImage(file, cv2.IMREAD_UNCHANGED)
                except Exception:
                    # Now try as video
                    image = Video().readVideoFrame(file, 0, "rgb24")[0]
        if(image is not None):
            colorMode = self.guessImageColor(image)
        return colorMode
      
    # Very simple auto-detect of image type (far from perfect)
    def guessImageColor(self, image):
        # Assume its color to start with
        colorMode = Video.COLOR_RGB
        rgb = Video.colorMode(image, Video.COLOR_RGB)
        if(np.array_equal(rgb[:,:,0],rgb[:,:,1]) and 
           np.array_equal(rgb[:,:,1],rgb[:,:,2])):
            # Not a color image
            colorMode = Video.COLOR_GRAYSCALE
            
            # Calculate all bayer options
            rggb = Video.colorMode(image, Video.COLOR_RGGB)
            grbg = Video.colorMode(image, Video.COLOR_GRBG)
            gbrg = Video.colorMode(image, Video.COLOR_GBRG)
            bggr = Video.colorMode(image, Video.COLOR_BGGR)
            
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
      
    # Changes the color mode of the image
    def colorMode(img, colorMode):
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
        except Exception as e:
            M = np.identity(3)
    else:
        M = np.identity(3)
    return M
       
# Returns a number based on how sharp the image is (higher = sharper)
def calculateSharpness(image):
    h, w = image.shape[:2]
    return cv2.Laplacian(cv2.resize(image, (int(max(100, w*0.1)), int(max(100, h*0.1)))), cv2.CV_8U).var()
   
# Multiprocess function to load frames from an image sequence
def loadFramesSequence(files, thread, width, height, colorMode, progress):
    frames = []
    sharps = []
    deltas = []
    colorSpaces = []
    
    prevImage = None
    if(thread > 0 and len(files) > 0):
        prevImage = Video.readImage(files.pop(0), dtype=np.uint8)
        prevImage = cv2.cvtColor(Video.colorMode(prevImage, colorMode), cv2.COLOR_BGR2GRAY)
        prevImage, wScaleFactor, hScaleFactor = resize(prevImage)
    
    for i, file in enumerate(files):
        progress.count(i, len(files))
        image, colorSpace = Video.decodeImage(file, dtype=np.uint8)
        h, w = image.shape[:2]
        if(h == height and w == width):
            # Only add if dimensions match
            frames.append(file)
            colorSpaces.append(colorSpace)
            image = Video.colorMode(image, colorMode)

            # Calculate sharpness
            sharps.append(calculateSharpness(image))
            
            # Calculate Frame Deltas
            image = cv2.cvtColor(Video.colorMode(image, colorMode), cv2.COLOR_BGR2GRAY)
            image, wScaleFactor, hScaleFactor = resize(image)
            M = calculateDelta(image, prevImage, wScaleFactor, hScaleFactor)
            deltas.append(M)
            
            prevImage = image
    progress.countExtra()
    return (frames, sharps, deltas, colorSpaces)
        
# Multiprocess function to load frames from a video source (times in ms)
def loadFramesVideo(file, times, previous, colorMode, progress):
    video = Video()
    frames = []
    sharps = []
    deltas = []
    
    prevImage = None
    if(previous is not None):
        prevImage = video.readVideoFrame(file, previous, "rgb24")[0]
        prevImage = cv2.cvtColor(Video.colorMode(prevImage, colorMode), cv2.COLOR_BGR2GRAY)
        prevImage, wScaleFactor, hScaleFactor = resize(prevImage)
        
    for i, time in enumerate(times):
        progress.count(i, len(times))
        try:
            image = video.readVideoFrame(file, time, "rgb24")[0]
        except EOFError:
            break
        if(video.lastTime != time):
            # Frame couldn't be decoded
            continue
        frames.append(time)
        
        # Calculate sharpness
        sharps.append(calculateSharpness(image))

        # Calculate Frame Deltas
        image = cv2.cvtColor(Video.colorMode(image, colorMode), cv2.COLOR_BGR2GRAY)
        image, wScaleFactor, hScaleFactor = resize(image)
        M = calculateDelta(image, prevImage, wScaleFactor, hScaleFactor)
        deltas.append(M)
        
        prevImage = image
    progress.countExtra()
    return (frames, sharps, deltas)
