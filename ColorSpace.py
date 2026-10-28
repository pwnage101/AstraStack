from __future__ import annotations
from lazy import lazy
np = lazy("numpy")
colour = lazy("colour")
Image = lazy("PIL.Image")
ImageCms = lazy("PIL.ImageCms")
import io
import struct
import warnings

# colour-science warns on import that its optional plotting features are missing
warnings.filterwarnings("ignore", message='"Matplotlib" related API features are not available')

# sRGB/Rec.709 primaries and D65 white point (CIE xy)
SRGB_PRIMARIES = ((0.64, 0.33), (0.30, 0.60), (0.15, 0.06))
D65 = (0.3127, 0.3290)
D50_XYZ = (0.9642, 1.0, 0.8249)

# Luminance weights of Rec.709/sRGB primaries
REC709_LUMINANCE = (0.2126, 0.7152, 0.0722)

# Transfer functions (encoded -> linear)
TRANSFER_LINEAR = "Linear"
TRANSFER_SRGB = "sRGB"
TRANSFER_BT709 = "BT.709" # Inverse BT.709 OETF, also used for BT.601
TRANSFER_BT2020 = "BT.2020" # Inverse BT.2020 OETF (10/12-bit)
TRANSFER_HLG = "HLG" # Inverse HLG OETF
TRANSFER_PQ = "PQ" # PQ EOTF, reference white = 1.0
TRANSFER_GAMMA = "Gamma" # params = exponent
TRANSFER_ICC = "ICC" # params = ICC tone curve for each channel (R, G, B)

# BT.2408 reference white for PQ (cd/m2)
PQ_REFERENCE_WHITE = 203

# Gamma of the intermediate encoding used when converting LUT based ICC profiles (u8Fixed8 of 2.2)
LUT_GAMMA = 563/256

# CICP (ITU-T H.273) color primaries
CICP_PRIMARIES = {1: "ITU-R BT.709",
                  4: "ITU-R BT.470 - 525",
                  5: "ITU-R BT.470 - 625",
                  6: "SMPTE C",
                  7: "SMPTE 240M",
                  8: "ITU-T H.273 - Generic Film",
                  9: "ITU-R BT.2020",
                  11: "DCI-P3",
                  12: "Display P3",
                  22: "EBU Tech. 3213-E"}

# CICP (ITU-T H.273) transfer characteristics
CICP_TRANSFERS = {1: (TRANSFER_BT709, None),
                  4: (TRANSFER_GAMMA, 2.2),
                  5: (TRANSFER_GAMMA, 2.8),
                  6: (TRANSFER_BT709, None),
                  7: (TRANSFER_BT709, None),
                  8: (TRANSFER_LINEAR, None),
                  13: (TRANSFER_SRGB, None),
                  14: (TRANSFER_BT2020, None),
                  15: (TRANSFER_BT2020, None),
                  16: (TRANSFER_PQ, None),
                  18: (TRANSFER_HLG, None)}

# Lookup tables for linearizing integer images, by transfer function and size
LUTS = {}

# ColorSpaces of the ICC profiles read so far (every frame of a sequence is read)
ICC_CACHE = {}

class ColorSpace:

    # primaries and white are CIE xy, matrix (optional) is RGB -> XYZ when it can't be derived from the primaries
    def __init__(self, name, primaries, white, transfer, params=None, matrix=None, icc=None):
        self.name = name
        self.primaries = tuple(tuple(float(v) for v in xy) for xy in primaries)
        self.white = tuple(float(v) for v in white)
        self.transfer = transfer
        self.params = params
        self.matrix = None if matrix is None else tuple(tuple(float(v) for v in row) for row in matrix)
        self.icc = icc
        self.cache = {}

    # The colour objects are rebuilt when needed after pickling
    def __getstate__(self):
        state = self.__dict__.copy()
        state["cache"] = {}
        return state

    def __repr__(self):
        return "ColorSpace(" + self.name + ", " + self.transfer + ")"

    # Returns the colour.RGB_Colourspace for the primaries and white point
    def colourspace(self):
        if("colourspace" not in self.cache):
            if(self.matrix is None):
                self.cache["colourspace"] = colour.RGB_Colourspace(self.name, self.primaries, self.white)
            else:
                self.cache["colourspace"] = colour.RGB_Colourspace(self.name, self.primaries, self.white,
                                                                    matrix_RGB_to_XYZ=np.array(self.matrix),
                                                                    matrix_XYZ_to_RGB=np.linalg.inv(self.matrix))
        return self.cache["colourspace"]

    # Returns the RGB -> XYZ matrix
    def matrixRGBToXYZ(self):
        return self.colourspace().matrix_RGB_to_XYZ

    # Returns whether the primaries and white point are sRGB's
    def isSRGB(self):
        if("isSRGB" not in self.cache):
            self.cache["isSRGB"] = np.allclose(self.matrixRGBToXYZ(), SRGB.matrixRGBToXYZ(), atol=1e-6)
        return self.cache["isSRGB"]

    # Returns the matrix converting linear RGB to linear sRGB (Bradford adaptation to D65)
    def matrixToSRGB(self):
        if("matrixToSRGB" not in self.cache):
            self.cache["matrixToSRGB"] = colour.matrix_RGB_to_RGB(self.colourspace(), SRGB.colourspace(), "Bradford")
        return self.cache["matrixToSRGB"]

    # Converts the 8/16-bit image to linear float32 (0-1) through a lookup table, channels in BGR order if bgr
    def linearizeInteger(self, img, bgr=False):
        n = np.iinfo(img.dtype).max + 1
        key = (self.transfer, repr(self.params), n)
        if(key not in LUTS):
            codes = np.arange(n, dtype=np.float32)/(n - 1)
            if(self.transfer == TRANSFER_ICC):
                LUTS[key] = np.stack([self.linearize(np.dstack([codes]*3))[0,:,i] for i in range(3)]).astype(np.float32)
            else:
                LUTS[key] = self.linearize(codes).astype(np.float32)
        lut = LUTS[key]
        if(self.transfer == TRANSFER_ICC):
            if(bgr):
                lut = lut[::-1]
            return np.dstack([lut[i][img[:,:,i]] for i in range(3)])
        return lut[img]
        
    # Converts the encoded values (0-1) to linear values, channels in BGR order if bgr
    def linearize(self, img, bgr=False):
        dtype = img.dtype
        if(self.transfer == TRANSFER_LINEAR):
            return img
        elif(self.transfer == TRANSFER_SRGB):
            img = colour.models.eotf_sRGB(img)
        elif(self.transfer == TRANSFER_BT709):
            img = colour.models.oetf_inverse_BT709(img)
        elif(self.transfer == TRANSFER_BT2020):
            img = colour.models.oetf_inverse_BT2020(img)
        elif(self.transfer == TRANSFER_HLG):
            img = colour.models.oetf_inverse_BT2100_HLG(np.maximum(img, 0))
        elif(self.transfer == TRANSFER_PQ):
            img = colour.models.eotf_ST2084(np.maximum(img, 0))/PQ_REFERENCE_WHITE
        elif(self.transfer == TRANSFER_GAMMA):
            img = colour.gamma_function(img, self.params, "Mirror")
        elif(self.transfer == TRANSFER_ICC):
            curves = self.params[::-1] if bgr else self.params
            img = np.dstack([curveToLinear(curve, img[:,:,i]) for i, curve in enumerate(curves)])
        return img.astype(dtype)

# Returns the named colour-science color space with the given transfer function
def named(name, transfer, params=None, icc=None):
    cs = colour.RGB_COLOURSPACES[name]
    return ColorSpace(name, cs.primaries, cs.whitepoint, transfer, params, icc=icc)

SRGB = ColorSpace("sRGB", SRGB_PRIMARIES, D65, TRANSFER_SRGB)

# Returns whether both describe the same primaries, white point and transfer function (or both have no description)
def same(a, b):
    if(a is None or b is None):
        return a is b
    return (np.allclose(a.matrixRGBToXYZ(), b.matrixRGBToXYZ(), atol=1e-4) and
            a.transfer == b.transfer and
            repr(a.params) == repr(b.params))

# Converts the linear image (0-1, RGB, in the color space's primaries) to linear sRGB primaries
def toSRGBPrimaries(img, colorSpace):
    if(colorSpace is None or colorSpace.isSRGB()):
        return img
    M = colorSpace.matrixToSRGB().astype(np.float32)
    return (img.reshape(-1, 3) @ M.T).reshape(img.shape)

# Encodes the linear image (RGB) for display: sRGB primaries, clipped to 0-1 and the sRGB curve
# Input without a color description (None) only gets the sRGB curve
def displayEncode(img, colorSpace):
    img = np.clip(toSRGBPrimaries(img, colorSpace), 0, 1)
    return colour.models.eotf_inverse_sRGB(img).astype(np.float32)

def luminanceWeights(colorSpace: ColorSpace | None) -> np.ndarray:
    """Returns the luminance weights of the color space's primaries.

    Args:
        colorSpace: Color description of the input, None when there is none.

    Returns:
        The Y row of the RGB -> XYZ matrix (float32), Rec.709's weights when colorSpace is None.
    """
    if(colorSpace is None):
        return np.array(REC709_LUMINANCE, dtype=np.float32)
    return np.asarray(colorSpace.matrixRGBToXYZ()[1], dtype=np.float32)

def whiteRGB(colorSpace: ColorSpace | None) -> np.ndarray:
    """Returns the white point in the color space's RGB.

    Args:
        colorSpace: Color description of the input, None when there is none.

    Returns:
        The white point with luminance 1 (float32), (1, 1, 1) when colorSpace is None.
    """
    if(colorSpace is None):
        return np.ones(3, dtype=np.float32)
    white = np.linalg.solve(colorSpace.matrixRGBToXYZ(), colour.xy_to_XYZ(colorSpace.white))
    return (white/white.dot(colorSpace.matrixRGBToXYZ()[1])).astype(np.float32)

# Returns the display encoded image (0-1) as 8-bit
def quantize(img):
    return np.around(np.clip(img, 0, 1)*255).astype(np.uint8)

# Returns the linear image (RGB) display encoded as 8-bit
def displayImage(img, colorSpace):
    return quantize(displayEncode(img, colorSpace))

# Converts a value from an ICC tone curve to linear
def curveToLinear(curve, x):
    if(curve[0] == "gamma"):
        return colour.gamma_function(x, curve[1], "Mirror")
    elif(curve[0] == "table"):
        table = np.array(curve[1], dtype=np.float64)/65535
        return np.interp(x, np.linspace(0, 1, len(table)), table)
    elif(curve[0] == "para"):
        # All parametric curve types as type 4: Y = (aX + b)^g + e for X >= d, else cX + f
        functionType, p = curve[1], curve[2]
        if(functionType == 0):
            g, a, b, c, d, e, f = p[0], 1, 0, 0, 0, 0, 0
        elif(functionType == 1):
            g, a, b, c, d, e, f = p[0], p[1], p[2], 0, -p[2]/p[1], 0, 0
        elif(functionType == 2):
            g, a, b, c, d, e, f = p[0], p[1], p[2], 0, -p[2]/p[1], p[3], p[3]
        elif(functionType == 3):
            g, a, b, c, d, e, f = p[0], p[1], p[2], p[3], p[4], 0, 0
        else:
            g, a, b, c, d, e, f = p
        return np.where(x >= d, np.maximum(a*x + b, 0)**g + e, c*x + f)

# Returns the tags of the ICC profile as a dict of signature -> data
def iccTags(icc):
    tags = {}
    count = struct.unpack(">I", icc[128:132])[0]
    for i in range(count):
        signature, offset, size = struct.unpack(">4sII", icc[132 + i*12:144 + i*12])
        tags[signature.decode("latin-1")] = icc[offset:offset + size]
    return tags

# Returns the ICC s15Fixed16 numbers in the data
def readS15Fixed16(data):
    return np.array(struct.unpack(">" + str(len(data)//4) + "i", data))/65536

# Returns the tone curve of an ICC curv or para tag
def iccCurve(tag):
    if(tag[:4] == b"curv"):
        count = struct.unpack(">I", tag[8:12])[0]
        if(count == 0):
            return ("gamma", 1.0)
        elif(count == 1):
            return ("gamma", struct.unpack(">H", tag[12:14])[0]/256)
        return ("table", struct.unpack(">" + str(count) + "H", tag[12:12 + count*2]))
    elif(tag[:4] == b"para"):
        functionType = struct.unpack(">H", tag[8:10])[0]
        count = [1, 3, 4, 5, 7][functionType]
        return ("para", functionType, tuple(readS15Fixed16(tag[12:12 + count*4])))
    raise ValueError("Unsupported ICC curve type")

# Returns whether the profile is a matrix/TRC RGB profile (no LUTs)
def isMatrixProfile(icc):
    tags = iccTags(icc)
    return (icc[16:20] == b"RGB " and
            "A2B0" not in tags and
            all(t in tags for t in ("rXYZ", "gXYZ", "bXYZ", "rTRC", "gTRC", "bTRC")))

# Returns whether the profile is a gray TRC profile
def isGrayProfile(icc):
    return icc[16:20] == b"GRAY" and "kTRC" in iccTags(icc)

# Returns the ColorSpace of a gray TRC ICC profile: its tone curve with sRGB primaries, which only matter for neutral values
# The ICC bytes aren't kept since a gray profile can't describe RGB data
def fromGrayICC(icc):
    curve = iccCurve(iccTags(icc)["kTRC"])
    return ColorSpace("Gray ICC", SRGB_PRIMARIES, D65, TRANSFER_ICC, (curve, curve, curve))

# Returns the ColorSpace of a matrix/TRC ICC profile
def fromICC(icc):
    if(icc in ICC_CACHE):
        return ICC_CACHE[icc]
    tags = iccTags(icc)
    colorants = np.column_stack([readS15Fixed16(tags[t][8:20]) for t in ("rXYZ", "gXYZ", "bXYZ")])
    # Colorants are stored adapted to D50, undo the adaptation to get the native primaries and white
    if("chad" in tags):
        chad = readS15Fixed16(tags["chad"][8:44]).reshape(3, 3)
    else:
        white = readS15Fixed16(tags["wtpt"][8:20]) if "wtpt" in tags else np.array(D50_XYZ)
        chad = bradford(white, D50_XYZ)
    M = np.linalg.inv(chad) @ colorants
    white = colour.XYZ_to_xy(M @ np.ones(3))
    primaries = colour.XYZ_to_xy(M.T)
    curves = tuple(iccCurve(tags[t]) for t in ("rTRC", "gTRC", "bTRC"))
    name = "ICC"
    try:
        name = ImageCms.getProfileDescription(ImageCms.ImageCmsProfile(io.BytesIO(icc))).strip()
    except Exception:
        pass
    colorSpace = ColorSpace(name, primaries, white, TRANSFER_ICC, curves, icc=icc)

    # Use the exact sRGB description when the profile is sRGB (within the precision of ICC numbers)
    x = np.linspace(0, 1, 4096)
    if(np.allclose(colorSpace.primaries, SRGB_PRIMARIES, atol=5e-4) and np.allclose(colorSpace.white, D65, atol=5e-4) and
       all(np.abs(curveToLinear(curve, x) - colour.models.eotf_sRGB(x)).max() < 1e-4 for curve in curves)):
        colorSpace = ColorSpace("sRGB", SRGB_PRIMARIES, D65, TRANSFER_SRGB, icc=icc)
    ICC_CACHE[icc] = colorSpace
    return colorSpace

# Linear Rec.2020 is the input's color space for LUT based ICC profiles, stored with the LUT_GAMMA curve
def lutColorSpace(icc):
    return named("ITU-R BT.2020", TRANSFER_GAMMA, LUT_GAMMA, icc=icc)

# Converts the 8-bit BGR image through the LUT based ICC profile to Rec.2020 (LUT_GAMMA encoded) with LittleCMS
def convertLUT(img, icc):
    rec2020 = lutColorSpace(icc)
    profile = iccProfile(rec2020.matrixRGBToXYZ(), rec2020.white, "Rec.2020", LUT_GAMMA)
    transform = ImageCms.buildTransform(ImageCms.ImageCmsProfile(io.BytesIO(icc)), ImageCms.ImageCmsProfile(io.BytesIO(profile)),
                                        "RGB", "RGB", renderingIntent=ImageCms.Intent.RELATIVE_COLORIMETRIC)
    rgb = Image.fromarray(np.ascontiguousarray(img[:,:,::-1]), "RGB")
    return np.asarray(ImageCms.applyTransform(rgb, transform))[:,:,::-1].copy()

# Returns the ColorSpace for CICP (ITU-T H.273) codes, unknown codes are Rec.709
def fromCICP(primaries, transfer):
    transfer, params = CICP_TRANSFERS.get(transfer, (TRANSFER_BT709, None))
    return named(CICP_PRIMARIES.get(primaries, "ITU-R BT.709"), transfer, params)

# Returns the ColorSpace of a RAW file (rawpy.RawPy) postprocessed in camera RGB with the camera white balance
def fromRAW(raw):
    wb = np.array(raw.camera_whitebalance[:3], dtype=np.float64)
    daylight = np.array(raw.daylight_whitebalance[:3], dtype=np.float64)
    if(not np.all(wb > 0)):
        wb = daylight
    xyzToCamera = np.array(raw.rgb_xyz_matrix[:3,:3], dtype=np.float64)
    rgbCamera = np.array(raw.color_matrix[:3,:3], dtype=np.float64)
    if(np.any(xyzToCamera) and np.all(wb > 0)):
        # Rows normalized so camera white (1,1,1) maps to D65
        cameraD65 = xyzToCamera @ colour.xy_to_XYZ(D65)
        cameraToXYZ = np.linalg.inv(xyzToCamera/cameraD65[:,None])
        gains = wb*cameraD65 # White balance gains relative to D65's
    elif(np.any(rgbCamera) and np.all(wb > 0) and np.all(daylight > 0)):
        # LibRaw only has the camera -> sRGB matrix (e.g. DNG), which already maps camera white (1,1,1) to D65
        cameraToXYZ = SRGB.matrixRGBToXYZ() @ rgbCamera
        gains = wb/daylight
    else:
        # No camera matrix
        return ColorSpace("Camera RGB", SRGB_PRIMARIES, D65, TRANSFER_LINEAR)
    # Undo the white balance gains, so camera neutral after white balance maps to the white point
    M = cameraToXYZ @ np.diag(1/gains)
    M /= (M @ np.ones(3))[1]
    white = colour.XYZ_to_xy(M @ np.ones(3))
    primaries = colour.XYZ_to_xy(M.T)
    return ColorSpace("Camera RGB", primaries, white, TRANSFER_LINEAR, matrix=M)

# Returns the chunks (type, data) of the PNG file before the first IDAT
def pngChunks(file):
    chunks = []
    with open(file, "rb") as f:
        if(f.read(8) != b"\x89PNG\r\n\x1a\n"):
            return chunks
        while(True):
            header = f.read(8)
            if(len(header) < 8 or header[4:8] == b"IDAT"):
                break
            length = struct.unpack(">I", header[:4])[0]
            chunks.append((header[4:8], f.read(length)))
            f.seek(4, 1) # CRC
    return chunks

# Returns the ColorSpace from the PNG cICP, sRGB, cHRM and gAMA chunks, or None if there aren't any
def fromPNGChunks(file):
    chunks = dict(pngChunks(file))
    if(b"cICP" in chunks):
        return fromCICP(chunks[b"cICP"][0], chunks[b"cICP"][1])
    if(b"sRGB" in chunks):
        return SRGB
    if(b"gAMA" in chunks or b"cHRM" in chunks):
        primaries, white = SRGB_PRIMARIES, D65
        if(b"cHRM" in chunks):
            v = np.array(struct.unpack(">8I", chunks[b"cHRM"][:32]))/100000
            white, primaries = v[0:2], v[2:8].reshape(3, 2)
        transfer, params = TRANSFER_SRGB, None
        if(b"gAMA" in chunks):
            gamma = struct.unpack(">I", chunks[b"gAMA"][:4])[0]/100000
            transfer, params = (TRANSFER_LINEAR, None) if abs(gamma - 1) < 1e-3 else (TRANSFER_GAMMA, 1/gamma)
        return ColorSpace("PNG", primaries, white, transfer, params)
    return None

# Returns the embedded ICC profile of the image file, or None
def readICC(file):
    try:
        with Image.open(file) as img:
            return img.info.get("icc_profile")
    except Exception:
        pass
    try:
        # Pillow can't open every TIFF
        tifffile = lazy("tifffile")
        with tifffile.TiffFile(file) as tif:
            tag = tif.pages[0].tags.get("InterColorProfile")
            return None if tag is None else bytes(tag.value)
    except Exception:
        return None

# Returns the color description of the (non RAW) image file and its 8/16-bit image (BGR), which is converted for LUT based ICC profiles
def describeImage(file, img):
    icc = readICC(file)
    if(icc):
        try:
            if(isGrayProfile(icc)):
                return (img, fromGrayICC(icc))
            elif(isMatrixProfile(icc)):
                return (img, fromICC(icc))
            elif(icc[16:20] == b"RGB " and len(img.shape) == 3):
                if(img.dtype != np.uint8):
                    # LittleCMS in Pillow only converts 8-bit RGB
                    img = np.around(img.astype(np.float32)/np.iinfo(img.dtype).max*255).astype(np.uint8)
                return (convertLUT(img, icc), lutColorSpace(icc))
        except Exception as error:
            print("Unsupported ICC profile, using sRGB:", error)
    if(file.lower().endswith(".png")):
        colorSpace = fromPNGChunks(file)
        if(colorSpace is not None):
            return (img, colorSpace)
    # Untagged
    return (img, SRGB)

# Bradford chromatic adaptation matrix from one white point (XYZ) to another
def bradford(src, dst):
    return colour.adaptation.matrix_chromatic_adaptation_VonKries(np.asarray(src), np.asarray(dst), "Bradford")

# Generates an ICC v2 matrix/TRC display profile from the RGB -> XYZ matrix and white point (CIE xy) with a gamma tone curve
def iccProfile(matrix, white, description, gamma=1.0):
    def s15Fixed16(values):
        return b"".join(struct.pack(">i", int(round(v*65536))) for v in values)
    def xyzTag(XYZ):
        return b"XYZ " + bytes(4) + s15Fixed16(XYZ)
    def descTag(text):
        ascii = text.encode("ascii") + b"\0"
        return b"desc" + bytes(4) + struct.pack(">I", len(ascii)) + ascii + bytes(4 + 4 + 2 + 1 + 67)
    def textTag(text):
        return b"text" + bytes(4) + text.encode("ascii") + b"\0"

    # Colorants and white are stored relative to the D50 PCS
    chad = bradford(colour.xy_to_XYZ(white), D50_XYZ)
    colorants = chad @ np.asarray(matrix)
    if(gamma == 1.0):
        curve = b"curv" + bytes(4) + struct.pack(">I", 0) # Count 0 = identity (gamma 1.0)
    else:
        curve = b"curv" + bytes(4) + struct.pack(">IH", 1, int(round(gamma*256)))
    tags = [(b"desc", descTag(description)),
            (b"cprt", textTag("No copyright, use freely")),
            (b"wtpt", xyzTag(D50_XYZ)),
            (b"chad", b"sf32" + bytes(4) + s15Fixed16(chad.flatten())),
            (b"rXYZ", xyzTag(colorants[:,0])),
            (b"gXYZ", xyzTag(colorants[:,1])),
            (b"bXYZ", xyzTag(colorants[:,2])),
            (b"rTRC", curve),
            (b"gTRC", curve),
            (b"bTRC", curve)]

    # Tag table followed by the 4-byte aligned tag data
    offset = 128 + 4 + 12*len(tags)
    table = struct.pack(">I", len(tags))
    data = b""
    for signature, tag in tags:
        table += signature + struct.pack(">II", offset + len(data), len(tag))
        data += tag + bytes(-len(tag) % 4)
    size = offset + len(data)
    header = (struct.pack(">I", size) + bytes(4) + struct.pack(">I", 0x02100000) +
              b"mntrRGB XYZ " + struct.pack(">6H", 2026, 1, 1, 0, 0, 0) + b"acsp" +
              bytes(4 + 4 + 4 + 4 + 8 + 4) + s15Fixed16(D50_XYZ) + bytes(4 + 16 + 28))
    return header + table + data
