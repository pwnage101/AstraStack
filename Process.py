from __future__ import annotations
from lazy import lazy
cv2 = lazy("cv2")
np = lazy("numpy")
pywt = lazy("pywt")
import math
import copy
import gc
from concurrent.futures import ProcessPoolExecutor, wait
from multiprocessing import Manager, Lock
from time import sleep
from deconvolution import *
from Video import Video
from Globals import *
import ColorSpace
import Export

pool = ProcessPoolExecutor(3)

class Process:

    STRETCH_NONE = 0
    STRETCH_SQRT = 1
    STRETCH_CBRT = 2
    STRETCH_LOG = 3
    STRETCH_ASINH = 4
    
    LEVEL = 5
    
    # Exposure slider range in EV
    EXPOSURE_MIN = -5
    EXPOSURE_MAX = 10
    
    # Black Level slider: log10 of the percentage of full scale, the far left end is 0
    BLACK_LEVEL_MIN = -2
    BLACK_LEVEL_MAX = 2
    
    # This is a crude estimate on how much memory will be used by the sharpening
    # Width * Height * 3Channels * 4bytes * 4coefficients * 5layers * 2(worst case from copying every coefficients layer)
    def estimateMemoryUsage(width, height):
        return width*height*3*4*4*Process.LEVEL*2
    
    def __init__(self, stackedImage: np.ndarray | str, isFile: bool = False) -> None:
        if(isFile):
            # Single image provided
            video = Video()
            stackedImage = video.getFrame(None, stackedImage, g.actualColor(), lens=g.lens)
        else:
            # Use the higher bit depth version from the stacking process
            pass
        # Linear, in the input's primaries
        self.inputImage = cv2.cvtColor(stackedImage, cv2.COLOR_BGR2RGB)
        self.colorSpace = g.actualColorSpace()
        self.luminance = ColorSpace.luminanceWeights(self.colorSpace)
        self.white = ColorSpace.whiteRGB(self.colorSpace)
        self.autoBlackLevel = estimateBlackLevel(self.inputImage)
        self.maxValue = float(self.inputImage.max())
        self.h, self.w = stackedImage.shape[:2]
        self.mask = None
        self.thresh = None
        self.calculateCoefficients()
        self.sharpenedImage = self.stackedImage
        self.debluredImage = self.stackedImage
        self.deringedImage = self.stackedImage
        self.finalImage = self.stackedImage
        self.processingRecord = Export.processingRecord(blackLevel=self.blackLevel,
                                                        exposure=self.exposure,
                                                        gains=(1.0, 1.0, 1.0),
                                                        saturation=1.0,
                                                        stretch=STRETCH_NAMES[Process.STRETCH_NONE],
                                                        gamma=1.0)
        self.processLevelsAgain = False
        self.processAgain = False
        self.processDeblurAgain = False
        self.processDeringAgain = False
        self.processColorAgain = False
        
    def run(self,
            processAgain: bool = True,
            processDeblurAgain: bool = False,
            processDeringAgain: bool = False,
            processColorAgain: bool = False,
            processLevelsAgain: bool = False) -> None:
        self.processLevelsAgain = processLevelsAgain
        self.processAgain = processAgain
        self.processDeblurAgain = processDeblurAgain
        self.processDeringAgain = processDeringAgain
        self.processColorAgain = processColorAgain
        while(self.processLevelsAgain or self.processAgain or self.processDeblurAgain or
              self.processDeringAgain or self.processColorAgain):
            if(self.processLevelsAgain):
                # Process black level & exposure, sharpening, deblur and color
                self.processLevelsAgain = False
                self.processAgain = False
                self.processDeblurAgain = False
                self.processDeringAgain = False
                self.processColorAgain = False
                self.calculateCoefficients()
                self.sharpenLayers()
                self.deblur()
                self.dering()
                self.processColor()
            elif(self.processAgain):
                # Process sharpening, deblur and color
                self.processAgain = False
                self.processDeblurAgain = False
                self.processDeringAgain = False
                self.processColorAgain = False
                self.sharpenLayers()
                self.deblur()
                self.dering()
                self.processColor()
            elif(self.processDeblurAgain):
                # Process deblur and color
                self.processAgain = False
                self.processDeblurAgain = False
                self.processDeringAgain = False
                self.processColorAgain = False
                self.deblur()
                self.dering()
                self.processColor()
            elif(self.processDeringAgain):
                # Only process color
                self.processAgain = False
                self.processDeblurAgain = False
                self.processDeringAgain = False
                self.processColorAgain = False
                self.dering()
                self.processColor()
            else:
                # Only process color
                self.processAgain = False
                self.processDeblurAgain = False
                self.processDeringAgain = False
                self.processColorAgain = False
                self.processColor()
            g.ui.finishedProcessing()
        
    def levels(self) -> tuple[float, float]:
        """Returns the black level and exposure to use, computed for the toggles that are automatic.

        Returns:
            The black level (fraction of full scale) and the exposure (EV).
        """
        if(g.autoBlackLevel):
            blackLevel = self.autoBlackLevel
        else:
            blackLevel = g.blackLevel
        if(g.autoExposure):
            exposure = autoExposure(self.maxValue, blackLevel)
        else:
            exposure = g.exposure
        return (blackLevel, exposure)
    
    # Applies the black level and exposure, then calculates the wavelet coefficents for each channel
    def calculateCoefficients(self) -> None:
        (self.blackLevel, self.exposure) = self.levels()
        self.stackedImage = applyLevels(img=self.inputImage, blackLevel=self.blackLevel, exposure=self.exposure)

        # Wavelets run on square root encoded data, which evens out photon noise
        (R, G, B) = cv2.split(sqrtEncode(self.stackedImage))

        # Frozen so the forked manager process never collects the objects it inherits (see UI.setThreads), without
        # collecting them first since this can run outside the GTK thread
        gc.freeze()
        try:
            manager = Manager()
        finally:
            gc.unfreeze()
        with manager:
            num = manager.Value('i', 0)
            lock = manager.Lock()
                
            futures = []
            futures.append(pool.submit(calculateChannelCoefficients, R, 'R', num, lock))
            futures.append(pool.submit(calculateChannelCoefficients, G, 'G', num, lock))
            futures.append(pool.submit(calculateChannelCoefficients, B, 'B', num, lock))
            
            wait(futures)
        
    # Sharpens each channel
    def sharpenLayers(self) -> None:
        gCopy = cloneGlobals()

        futures = []
        futures.append(pool.submit(sharpenChannelLayers, gCopy))
        futures.append(pool.submit(sharpenChannelLayers, gCopy))
        futures.append(pool.submit(sharpenChannelLayers, gCopy))
        
        for future in futures:
            result = future.result()
            if(result[0] == 'R'):
                R = result[1]
            elif(result[0] == 'G'):
                G = result[1]
            elif(result[0] == 'B'):
                B = result[1]
                
        self.sharpenedImage = sqrtDecode(cv2.merge([R, G, B])[:self.h,:self.w])
        
    def deblur(self) -> None:
        img = self.sharpenedImage
        # Deconvolve
        if((g.deconvolveCircular and g.deconvolveCircularDiameter > 1) or 
           (g.deconvolveGaussian and g.deconvolveGaussianDiameter > 1) or 
           (g.deconvolveLinear and g.deconvolveLinearDiameter > 1) or
           (g.deconvolveCustom and g.deconvolveCustomFile is not None)):
            # Decompose
            beforeAvg = np.average(img)
            (R, G, B) = cv2.split(img)

            gCopy = cloneGlobals()
            
            futures = []
            futures.append(pool.submit(deconvolve, R, gCopy))
            futures.append(pool.submit(deconvolve, G, gCopy))
            futures.append(pool.submit(deconvolve, B, gCopy))
            
            R = futures[0].result()
            G = futures[1].result()
            B = futures[2].result()
        
            # Recompose
            img = cv2.merge([R, G, B])
            # Brightness tends to darken the image slightly, so adjust the brightness
            try:
                afterAvg = np.average(img)
                factor = beforeAvg / afterAvg
                img *= factor
            except:
                pass
        self.debluredImage = img
        
    # Apply dering/star mask to sharpened image
    def dering(self) -> None:
        if((g.showAdaptive and g.deringAdaptive > 0) or 
           (g.showDark and g.deringDark > 0) or 
           (g.showBright and g.deringBright > 0)):
            # Thresholds work on display encoded 0-255 luminance
            gray = displayLuminance(self.stackedImage, self.luminance)
            # Calculate Dark & Bright thresholds and merge them
            threshAdaptive = np.zeros((self.h, self.w), dtype = "uint8")
            threshDark = np.zeros((self.h, self.w), dtype = "uint8")
            threshBright = np.zeros((self.h, self.w), dtype = "uint8")
            if(g.showAdaptive and g.deringAdaptive > 0):
                threshAdaptive = cv2.adaptiveThreshold(gray.astype(np.uint8), 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 30-g.deringAdaptive)
            if(g.showDark and g.deringDark > 0):
                ret,threshDark = cv2.threshold(gray, (g.deringDark*2)-1, 255, cv2.THRESH_BINARY_INV)
            if(g.showBright and g.deringBright > 0):
                ret,threshBright = cv2.threshold(gray, 255-(g.deringBright*2), 255, cv2.THRESH_BINARY)
            thresh = np.maximum(np.maximum(threshAdaptive, threshDark), threshBright)
            
            # Dialate the threshold
            if(g.deringSize != 0):
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (abs(g.deringSize)*2+1,abs(g.deringSize)*2+1))
                if(g.deringSize > 0):
                    thresh = cv2.dilate(thresh, kernel)
                else:
                    thresh = cv2.erode(thresh, kernel)
            
            # Blur the thresholds
            if(g.deringBlend > 0):
                thresh = cv2.GaussianBlur(thresh, (0,0), g.deringBlend)
            
            self.thresh = cv2.cvtColor(thresh, cv2.COLOR_GRAY2RGB)
            mask = self.thresh.astype(np.float32)/255
            inv_mask = 1 - mask
            orig = self.stackedImage * mask
            processed = self.debluredImage * inv_mask
            self.deringedImage = orig + processed
        else:
            self.thresh = cv2.cvtColor(np.zeros((self.h, self.w), dtype = "uint8"), cv2.COLOR_GRAY2RGB)
            self.deringedImage = self.debluredImage
    
    # Apply color gains, then the tone and color controls
    def processColor(self) -> None:
        img = self.deringedImage
        
        gains = (g.redAdjust/100, g.greenAdjust/100, g.blueAdjust/100)
        saturation = g.saturation/100
        stretchFunction = g.stretchFunction
        gamma = max(1, g.gamma)/100
        
        # Red/Green/Blue gains (linear)
        img = img*np.array(gains, dtype=np.float32)
        
        # Ratio preserving tone and color
        img = saturate(img=img, saturation=saturation, luminance=self.luminance, white=self.white)
        img = stretch(img, stretchFunction)
        img = applyGamma(img, gamma)
        
        self.processingRecord = Export.processingRecord(blackLevel=self.blackLevel,
                                                        exposure=self.exposure,
                                                        gains=gains,
                                                        saturation=saturation,
                                                        stretch=STRETCH_NAMES[stretchFunction],
                                                        gamma=gamma)
        self.finalImage = img.astype(np.float32)
        
def estimateBlackLevel(img: np.ndarray) -> float:
    """Estimates the black level of the image.

    Args:
        img: Linear RGB image.

    Returns:
        The minimum of the blurred, downscaled lightness ((max + min)/2 of the channels), clipped to 0-1.
    """
    small = cv2.resize(src=img, dsize=(100, 100), interpolation=cv2.INTER_LINEAR)
    lightness = (small.max(axis=2) + small.min(axis=2))/2
    lightness = cv2.GaussianBlur(src=lightness, ksize=(0,0), sigmaX=2)
    return float(min(max(lightness.min(), 0), 1))

def blackLevelFromSlider(value: float) -> float:
    """Converts the Black Level slider value to a fraction of full scale.

    Args:
        value: Slider value, log10 of the percentage of full scale.

    Returns:
        The black level as a fraction of full scale, 0 at the far left end of the slider.
    """
    if(value <= Process.BLACK_LEVEL_MIN):
        return 0
    return 10**value/100

def sliderFromBlackLevel(blackLevel: float) -> float:
    """Converts a fraction of full scale to the Black Level slider value.

    Args:
        blackLevel: Black level as a fraction of full scale.

    Returns:
        The slider value, limited to the slider range.
    """
    if(blackLevel <= 10**Process.BLACK_LEVEL_MIN/100):
        return Process.BLACK_LEVEL_MIN
    return min(math.log10(blackLevel*100), Process.BLACK_LEVEL_MAX)

def autoExposure(maxValue: float, blackLevel: float) -> float:
    """Computes the exposure that puts the brightest value after black level subtraction at 1.0.

    Args:
        maxValue: Brightest channel value of the image.
        blackLevel: Black level subtracted from the image.

    Returns:
        The exposure in EV, limited to the Exposure slider range (0 when nothing is above the black level).
    """
    if(maxValue - blackLevel <= 0):
        return 0
    return float(np.clip(a=-math.log2(maxValue - blackLevel), a_min=Process.EXPOSURE_MIN, a_max=Process.EXPOSURE_MAX))

def applyLevels(img: np.ndarray, blackLevel: float, exposure: float) -> np.ndarray:
    """Subtracts the black level and applies the exposure, without clipping.

    Args:
        img: Linear image.
        blackLevel: Value subtracted from every channel.
        exposure: Gain in EV (stops).

    Returns:
        The float32 image (img - blackLevel)*2^exposure.
    """
    return ((img - np.float32(blackLevel))*np.float32(2**exposure)).astype(np.float32)

def sqrtEncode(img: np.ndarray) -> np.ndarray:
    """Square root encodes the image, keeping the sign.

    Args:
        img: Linear image.

    Returns:
        The float32 image sqrt(|x|)*sign(x).
    """
    return (np.sqrt(np.abs(img))*np.sign(img)).astype(np.float32)

def sqrtDecode(img: np.ndarray) -> np.ndarray:
    """Decodes a square root encoded image back to linear.

    Args:
        img: Image encoded by sqrtEncode.

    Returns:
        The float32 image x*|x|.
    """
    return (img*np.abs(img)).astype(np.float32)

def displayLuminance(img: np.ndarray, luminance: np.ndarray) -> np.ndarray:
    """Computes the display encoded luminance, for thresholds that work on 0-255 values.

    Args:
        img: Linear RGB image.
        luminance: Luminance weights of the image's primaries.

    Returns:
        The float32 luminance clipped to 0-1, encoded with the sRGB curve and scaled to 0-255.
    """
    Y = np.clip(a=img.dot(luminance), a_min=0, a_max=1)
    return (ColorSpace.colour.models.eotf_inverse_sRGB(Y)*255).astype(np.float32)

def scalePixels(img: np.ndarray, ratio: np.ndarray) -> np.ndarray:
    """Multiplies all channels of each pixel by the same ratio.

    Args:
        img: RGB image (h x w x 3).
        ratio: Ratio for each pixel (h x w), pixels with a NaN or infinite ratio become 0.

    Returns:
        The scaled image.
    """
    return img*np.nan_to_num(x=ratio, nan=0, posinf=0, neginf=0)[:,:,None]

def saturate(img: np.ndarray, saturation: float, luminance: np.ndarray, white: np.ndarray) -> np.ndarray:
    """Mixes each pixel toward its own luminance gray (Y*white), or away from it.

    Args:
        img: Linear RGB image.
        saturation: Saturation factor, 0 is gray and 1 leaves the image unchanged.
        luminance: Luminance weights of the image's primaries.
        white: White point in the image's RGB with luminance 1.

    Returns:
        The float32 image Y*white + saturation*(rgb - Y*white), not clipped.
    """
    if(saturation == 1):
        return img
    gray = img.dot(luminance)[:,:,None]*white
    return (gray + np.float32(saturation)*(img - gray)).astype(np.float32)

# Stretch names as shown in the Stretch dropdown
STRETCH_NAMES = {Process.STRETCH_NONE: "None",
                 Process.STRETCH_SQRT: "Square Root",
                 Process.STRETCH_CBRT: "Cube Root",
                 Process.STRETCH_LOG: "Logarithm",
                 Process.STRETCH_ASINH: "ArcSinH"}

# Stretch curves (normalized so f(1) = 1) of the intensity
STRETCH_CURVES = {Process.STRETCH_SQRT: lambda I: np.sqrt(I),
                  Process.STRETCH_CBRT: lambda I: np.cbrt(I),
                  Process.STRETCH_LOG: lambda I: np.log1p(255*I)/np.log(256),
                  Process.STRETCH_ASINH: lambda I: np.arcsinh(255*I)/np.arcsinh(255)}

def stretch(img: np.ndarray, stretchFunction: int) -> np.ndarray:
    """Applies the stretch curve to the intensity (R+G+B)/3, keeping the R:G:B ratios.

    Pixels with an intensity of 0 or less become 0, and pixels with a channel above 1.0 after the stretch are scaled
    down as a whole.

    Args:
        img: Linear RGB image.
        stretchFunction: One of the Process.STRETCH_* constants.

    Returns:
        The stretched float32 image, or img unchanged for Process.STRETCH_NONE.
    """
    if(stretchFunction not in STRETCH_CURVES):
        return img
    I = img.mean(axis=2)
    with np.errstate(divide='ignore', invalid='ignore'):
        img = scalePixels(img, np.where(I > 0, STRETCH_CURVES[stretchFunction](np.maximum(I, 0))/I, 0))
        peak = img.max(axis=2)
        img = scalePixels(img, np.where(peak > 1, 1/peak, 1))
    return img.astype(np.float32)

def applyGamma(img: np.ndarray, gamma: float) -> np.ndarray:
    """Applies the gamma to the intensity (R+G+B)/3, keeping the R:G:B ratios.

    Args:
        img: RGB image.
        gamma: Gamma, the intensity I becomes I^(1/gamma); pixels with I <= 0 become 0.

    Returns:
        The float32 image, or img unchanged when gamma is 1.
    """
    if(gamma == 1):
        return img
    I = img.mean(axis=2)
    with np.errstate(divide='ignore', invalid='ignore'):
        img = scalePixels(img, np.where(I > 0, np.power(np.maximum(I, 0), 1/gamma)/I, 0))
    return img.astype(np.float32)
        
# Calculates the wavelet coefficents for the specified channel
def calculateChannelCoefficients(C, channel, num, lock):
    # Pad the image so that there is a border large enough so that edge artifacts don't occur
    padding = 2**(Process.LEVEL)
    C = cv2.copyMakeBorder(C, padding, padding, padding, padding, cv2.BORDER_REFLECT)
    
    # Pad so that dimensions are multiples of 2**level
    h, w = C.shape[:2]
    
    hR = (h % 2**Process.LEVEL)
    wR = (w % 2**Process.LEVEL)

    C = cv2.copyMakeBorder(C, 0, (2**Process.LEVEL - hR), 0, (2**Process.LEVEL - wR), cv2.BORDER_REFLECT)
    
    C =  np.float32(C)

    # Compute coefficients
    g.coeffs = list(pywt.swt2(C, 'haar', level=Process.LEVEL, trim_approx=True))
    g.channel = channel
    
    # Make sure that the other processes have completed as well
    with lock:
        num.value += 1
    while(num.value < 3):
        sleep(0.01)
    
# Reconstructs the wavelet using varying intensities of coefficients, 
# as well as unsharp mask for additional sharpening
# and gaussian denoise
def sharpenChannelLayers(gCopy):
    c = g.coeffs
    # Go through each wavelet layer and apply sharpening
    cCopy = []
    for i in range(1, len(c)):
        level = (len(c) - i - 1)
        # Copy the layer if a change is made to the coefficients
        if(gCopy.level[level] and (gCopy.radius[level] > 0 or 
                                   gCopy.sharpen[level] > 0 or 
                                   gCopy.denoise[level] > 0)):
            cCopy.append(copy.deepcopy(c[i]))
        else:
            cCopy.append(None)
            
        # Process Layers
        if(gCopy.level[level]):
            for ci in c[i]:
                # Apply Unsharp Mask
                if(gCopy.radius[level] > 0):
                    unsharp(ci, gCopy.radius[level], 2)
                # Multiply the layer to increase intensity
                if(gCopy.sharpen[level] > 0):
                    factor = (100 - 10*level)
                    cv2.add(ci, ci*gCopy.sharpen[level]*factor, ci)
                # Denoise
                if(gCopy.denoise[level] > 0):
                    unsharp(ci, gCopy.denoise[level], -1)
    
    # Reconstruction
    padding = 2**(Process.LEVEL)
    img=pywt.iswt2(c, 'haar')
    
    # Reset coefficients
    for i in range(1, len(c)):
        if(cCopy[i-1] is not None):
            c[i] = cCopy[i-1]
    
    # Prepare image for saving
    img = img[padding:,padding:]
    
    return (g.channel, img)
    
# Applies an unsharp mask to the specified image
def unsharp(image, radius, strength):
    blur = cv2.GaussianBlur(image, (0,0), radius)
    sharp = cv2.addWeighted(image, 1+strength, blur, -strength, 0, image)

