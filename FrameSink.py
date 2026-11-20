from __future__ import annotations
from lazy import lazy
cv2 = lazy("cv2")
np = lazy("numpy")
pypng = lazy("png")
import io
import struct
import ColorSpace

class FrameSink:
    """An image file that linear working data is saved to, as display-referred sRGB.

    PNGs are tagged with cICP, sRGB and gAMA chunks; JPEGs and TIFFs are untagged, which readers take as sRGB.
    """

    EXTENSIONS = (".png", ".jpg", ".jpeg", ".tif", ".tiff")
    # The chunks that tag a PNG as sRGB
    PNG_SRGB_CHUNKS = [(b"cICP", bytes([1, 13, 0, 1])),
                       (b"sRGB", bytes([0])),
                       (b"gAMA", struct.pack(">I", 45455))]

    def __init__(self, path: str, bits: int) -> None:
        """Initializes the sink.

        Args:
            path: Path of the file; its extension must be one of EXTENSIONS.
            bits: 8 or 16 (16 only for PNG and TIFF).

        Raises:
            ValueError: The extension or bit depth isn't supported.
        """
        self.path = path
        self.bits = bits
        self.extension = path.lower()[path.rfind("."):] if "." in path else ""
        if(self.extension not in FrameSink.EXTENSIONS):
            raise ValueError("Unsupported file extension " + repr(self.extension))
        if(bits == 16 and self.extension in (".jpg", ".jpeg")):
            raise ValueError("JPEG can't store 16-bit images")

    def write(self, img: np.ndarray, workingSpace: ColorSpace.WorkingSpace | None) -> None:
        """Writes the file.

        Args:
            img: Linear RGB image (channels last, RGB order).
            workingSpace: The working data's colors.

        Raises:
            ValueError: The image could not be encoded.
        """
        rgb = ColorSpace.displayImage(img=img, workingSpace=workingSpace, bits=self.bits)
        options = [cv2.IMWRITE_JPEG_QUALITY, 95] if self.extension in (".jpg", ".jpeg") else []
        success, encoded = cv2.imencode(self.extension, np.ascontiguousarray(rgb[:, :, ::-1]), options)
        if(not success):
            raise ValueError("The image could not be encoded")
        data = encoded.tobytes()
        if(self.extension == ".png"):
            data = FrameSink.insertPNGChunks(png=data, chunks=FrameSink.PNG_SRGB_CHUNKS)
        with open(self.path, "wb") as f:
            f.write(data)

    @staticmethod
    def insertPNGChunks(png: bytes, chunks: list[tuple[bytes, bytes]]) -> bytes:
        """Inserts chunks into a PNG right after its IHDR.

        Args:
            png: The PNG file's bytes.
            chunks: The chunks to insert as (type, data).

        Returns:
            The new PNG file's bytes.
        """
        existing = list(pypng.Reader(bytes=png).chunks())
        out = io.BytesIO()
        pypng.write_chunks(out, existing[:1] + chunks + existing[1:])
        return out.getvalue()
