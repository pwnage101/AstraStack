from lazy import lazy
cv2 = lazy("cv2")
np = lazy("numpy")
lensfunpy = lazy("lensfunpy")
Image = lazy("PIL.Image")
import math
import os
import struct

# Nikon NCTG tags (moov/udta/NCDT/NCTG)
NCTG_MAKE = 0x0001
NCTG_MODEL = 0x0002
NCTG_FNUMBER = 0x110829d
NCTG_FOCAL_LENGTH = 0x110920a
NCTG_LENS_MAKE = 0x110a433
NCTG_LENS_MODEL = 0x110a434
NCTG_CROP_HISPEED = 0x200001b

# Sizes of the NCTG/TIFF data types
TYPE_SIZES = {1: 1, 2: 1, 3: 2, 4: 4, 5: 8, 6: 1, 7: 1, 8: 2, 9: 4, 10: 8, 11: 4, 12: 8}

# Mounts with 4:3 sensors, every other sensor is assumed to be 3:2
MOUNTS_4_3 = ("4/3 System", "Micro 4/3 System")

# The lensfun database, loaded when first needed
DATABASE = None

# Coordinate and vignetting maps of the last geometry, reused for every frame
MAPS = {}
mapsComputed = 0

# The lens correction settings, picklable so they can be passed to the pool processes
class LensCorrection:

    def __init__(self):
        self.cameraMaker = None
        self.cameraModel = None
        self.lensMaker = None
        self.lensModel = None
        self.focal = 0.0
        self.aperture = 0.0
        self.distortion = False
        self.tca = False
        self.vignetting = False
        # Nikon crop region (mode, sensor width, sensor height, crop width, crop height, crop x, crop y)
        self.cropRegion = None

    # Returns whether any correction would be applied
    def enabled(self):
        return (self.lensModel is not None and self.focal > 0 and self.aperture > 0 and
                (self.distortion or self.tca or self.vignetting))

    # Returns everything the correction of an image depends on
    def key(self):
        if(not self.enabled()):
            return None
        return (self.cameraMaker, self.cameraModel, self.lensMaker, self.lensModel, self.focal, self.aperture,
                self.distortion, self.tca, self.vignetting, self.cropRegion)

# Returns the lensfun database (only the database bundled with lensfunpy, which matches its lensfun version)
def database():
    global DATABASE
    if(DATABASE is None):
        DATABASE = lensfunpy.Database(load_common=False)
    return DATABASE

# Returns the lensfun camera, None if it isn't in the database
def findCamera(maker, model):
    if(not maker or not model):
        return None
    cameras = database().find_cameras(maker, model)
    if(len(cameras) == 0):
        return None
    return cameras[0]

# Returns the lensfun lens, None if it isn't in the database
def findLens(camera, maker, model):
    if(not model):
        return None
    if(camera is not None):
        lenses = database().find_lenses(camera, maker or None, model)
        if(len(lenses) > 0):
            return lenses[0]
    # Without a camera (or a lens for another mount) every lens is searched
    model = model.strip().lower()
    matches = [lens for lens in database().lenses if lens.model.lower() == model]
    if(maker):
        sameMaker = [lens for lens in matches if lens.maker.lower() == maker.strip().lower()]
        if(len(sameMaker) > 0):
            matches = sameMaker
    if(len(matches) > 0):
        return matches[0]
    return None

# Returns whether the cameras or lenses (or None) are the same entry of the database
def sameItem(item1, item2):
    if(item1 is None or item2 is None):
        return (item1 is None and item2 is None)
    return (type(item1) == type(item2) and item1.maker == item2.maker and item1.model == item2.model and
            item1.crop_factor == item2.crop_factor and getattr(item1, "variant", None) == getattr(item2, "variant", None))
    
# Returns a short name of the camera or lens for the dropdowns
def displayName(item):
    maker = item.maker.split(" ")[0]
    if(item.model.lower().startswith(maker.lower())):
        return item.model
    return item.maker + " " + item.model

# Returns the image's crop factor and whether geometry (distortion, TCA) can be corrected
def imageCropFactor(lens, camera, cropRegion, width, height):
    if(camera is not None):
        cameraCrop = camera.crop_factor
    else:
        # Without a camera, assume the camera the lens was calibrated with
        cameraCrop = lens.crop_factor
    if(cropRegion is not None):
        # Nikon crop region in sensor pixels
        mode, sensorWidth, sensorHeight, cropWidth, cropHeight, cropX, cropY = cropRegion
        centered = (abs(cropX - (sensorWidth - cropWidth)/2) <= 1 and abs(cropY - (sensorHeight - cropHeight)/2) <= 1)
        return (cameraCrop*math.hypot(sensorWidth, sensorHeight)/math.hypot(cropWidth, cropHeight), centered)
    # The image spans the sensor's width and is cropped symmetrically (or its height if it's narrower than the sensor)
    sensorAspect = 4/3 if camera is not None and camera.mount in MOUNTS_4_3 else 3/2
    imageAspect = max(width, height)/min(width, height)
    if(imageAspect >= sensorAspect):
        ratio = math.hypot(1, 1/sensorAspect)/math.hypot(1, 1/imageAspect)
    else:
        ratio = math.hypot(1, sensorAspect)/math.hypot(1, imageAspect)
    return (cameraCrop*ratio, True)

# Returns the coordinate maps and vignetting gain of the correction, computed once per geometry
def getMaps(settings, width, height):
    global mapsComputed
    key = (settings.key(), width, height)
    if(key in MAPS):
        return MAPS[key]
    MAPS.clear()
    camera = findCamera(settings.cameraMaker, settings.cameraModel)
    lens = findLens(camera, settings.lensMaker, settings.lensModel)
    maps = None
    if(lens is not None):
        crop, geometry = imageCropFactor(lens, camera, settings.cropRegion, width, height)
        flags = lensfunpy.ModifyFlags
        gain = None
        coords = None
        if(settings.vignetting):
            modifier = lensfunpy.Modifier(lens, crop, width, height)
            modifier.initialize(settings.focal, settings.aperture, pixel_format=np.float32, flags=flags.VIGNETTING)
            gain = np.ones((height, width, 3), np.float32)
            if(modifier.apply_color_modification(gain)):
                gain = np.ascontiguousarray(gain[:,:,0])
            else:
                gain = None
        if(geometry and (settings.distortion or settings.tca)):
            modifierFlags = flags.SCALE
            if(settings.distortion):
                modifierFlags |= flags.DISTORTION
            if(settings.tca):
                modifierFlags |= flags.TCA
            modifier = lensfunpy.Modifier(lens, crop, width, height)
            modifier.initialize(settings.focal, settings.aperture, scale=0.0, pixel_format=np.float32, flags=modifierFlags)
            if(settings.tca):
                coords = modifier.apply_subpixel_geometry_distortion()
                if(coords is not None):
                    # One map per channel (lensfun's are RGB, the frames are BGR)
                    coords = [(np.ascontiguousarray(coords[:,:,c,0]), np.ascontiguousarray(coords[:,:,c,1])) for c in (2, 1, 0)]
            else:
                coords = modifier.apply_geometry_distortion()
                if(coords is not None):
                    coords = (np.ascontiguousarray(coords[:,:,0]), np.ascontiguousarray(coords[:,:,1]))
        if(gain is not None or coords is not None):
            maps = (gain, coords)
    mapsComputed += 1
    MAPS[key] = maps
    return maps

# Corrects the linear frame (BGR, float32), the frame is returned unchanged when there's nothing to correct
def correct(image, settings):
    if(settings is None or not settings.enabled()):
        return image
    height, width = image.shape[:2]
    maps = getMaps(settings, width, height)
    if(maps is None):
        return image
    gain, coords = maps
    # Vignetting is calibrated on the uncorrected image, so it's corrected first
    if(gain is not None):
        image = image*gain[:,:,None]
    if(isinstance(coords, list)):
        image = cv2.merge([cv2.remap(np.ascontiguousarray(image[:,:,c]), x, y, cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
                           for c, (x, y) in enumerate(coords)])
    elif(coords is not None):
        image = cv2.remap(image, coords[0], coords[1], cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return image

# Returns the decoded value of an NCTG/EXIF entry
def decodeValue(data, type, count):
    if(type == 2):
        return data.split(b"\0")[0].decode("latin-1").strip()
    if(type == 3):
        values = struct.unpack(">" + "H"*count, data)
    elif(type == 8):
        values = struct.unpack(">" + "h"*count, data)
    elif(type == 4):
        values = struct.unpack(">" + "I"*count, data)
    elif(type == 9):
        values = struct.unpack(">" + "i"*count, data)
    elif(type == 5 or type == 10):
        parts = struct.unpack(">" + ("I" if type == 5 else "i")*(2*count), data)
        values = tuple(parts[i]/parts[i+1] if parts[i+1] != 0 else 0.0 for i in range(0, len(parts), 2))
    else:
        return data
    return values[0] if count == 1 else values

# Returns the position and end of the atom at the path (list of types) in a QuickTime file, None if there isn't one
def findAtom(f, path, start, end):
    position = start
    while(position + 8 <= end):
        f.seek(position)
        size, type = struct.unpack(">I4s", f.read(8))
        header = 8
        if(size == 1):
            size = struct.unpack(">Q", f.read(8))[0]
            header = 16
        elif(size == 0):
            size = end - position
        if(size < header):
            return None
        if(type == path[0]):
            if(len(path) == 1):
                return (position + header, min(end, position + size))
            # There can be more than one atom of a type (e.g. udta), so keep looking if this one doesn't have the rest
            atom = findAtom(f, path[1:], position + header, min(end, position + size))
            if(atom is not None):
                return atom
        position += size
    return None

# Returns the tags (id -> value) of the Nikon NCTG atom of a MOV/MP4, None if there isn't one
def readNCTG(file):
    try:
        with open(file, "rb") as f:
            atom = findAtom(f, [b"moov", b"udta", b"NCDT", b"NCTG"], 0, os.path.getsize(file))
            if(atom is None):
                return None
            f.seek(atom[0])
            data = f.read(atom[1] - atom[0])
    except Exception:
        return None
    tags = {}
    i = 0
    while(i + 8 <= len(data)):
        id, type, count = struct.unpack(">IHH", data[i:i+8])
        i += 8
        size = TYPE_SIZES.get(type, 1)*count
        if(i + size > len(data)):
            break
        try:
            tags[id] = decodeValue(data[i:i+size], type, count)
        except Exception:
            pass
        i += size
    return tags

# Returns the lens metadata of the file (make, model, lensMake, lens, focal, aperture, cropRegion), empty when there's none
def readMetadata(file):
    if(isinstance(file, list)):
        file = file[0]
    metadata = {}
    tags = readNCTG(file)
    if(tags is not None):
        # Nikon video
        metadata["make"] = tags.get(NCTG_MAKE)
        metadata["model"] = tags.get(NCTG_MODEL)
        metadata["lensMake"] = tags.get(NCTG_LENS_MAKE)
        metadata["lens"] = tags.get(NCTG_LENS_MODEL)
        metadata["focal"] = tags.get(NCTG_FOCAL_LENGTH)
        metadata["aperture"] = tags.get(NCTG_FNUMBER)
        crop = tags.get(NCTG_CROP_HISPEED)
        if(isinstance(crop, tuple) and len(crop) == 7 and min(crop[1:5]) > 0):
            metadata["cropRegion"] = crop
    else:
        # Still image EXIF
        try:
            with Image.open(file) as image:
                exif = image.getexif()
                exifIFD = exif.get_ifd(0x8769)
                metadata["make"] = exif.get(0x010f)
                metadata["model"] = exif.get(0x0110)
                metadata["lensMake"] = exifIFD.get(0xa433)
                metadata["lens"] = exifIFD.get(0xa434)
                metadata["focal"] = exifIFD.get(0x920a)
                metadata["aperture"] = exifIFD.get(0x829d)
        except Exception:
            pass
    for name in ("make", "model", "lensMake", "lens"):
        if(isinstance(metadata.get(name), str)):
            metadata[name] = metadata[name].strip("\0 ") or None
        else:
            metadata.pop(name, None)
    for name in ("focal", "aperture"):
        try:
            metadata[name] = float(metadata[name])
        except Exception:
            metadata.pop(name, None)
    return metadata
