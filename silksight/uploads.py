"""Validate a user-uploaded photo in memory. Nothing is written to disk and the returned image
carries no EXIF/GPS metadata."""

import io
import warnings

from PIL import Image, ImageOps, UnidentifiedImageError

MAX_UPLOAD_BYTES = 10 * 1024 * 1024
MAX_PIXELS = 40_000_000
ALLOWED_FORMATS = {"JPEG", "PNG", "WEBP"}


class UploadError(ValueError):
    """The upload is not an acceptable photo; ``too_large`` maps to HTTP 413."""

    def __init__(self, message: str, too_large: bool = False) -> None:
        super().__init__(message)
        self.too_large = too_large


def read_upload(data: bytes) -> Image.Image:
    if not data:
        raise UploadError("empty file")
    if len(data) > MAX_UPLOAD_BYTES:
        raise UploadError("file is larger than 10 MB", too_large=True)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(data)) as image:
                # Check format and size from the header before decoding any pixels.
                if image.format not in ALLOWED_FORMATS:
                    raise UploadError("use a JPEG, PNG or WebP photo")
                if image.width * image.height > MAX_PIXELS:
                    raise UploadError("image is larger than 40 megapixels", too_large=True)
                image.load()
                return ImageOps.exif_transpose(image).convert("RGB")
    except (Image.DecompressionBombWarning, Image.DecompressionBombError) as error:
        raise UploadError("image is too large to decode", too_large=True) from error
    except (UnidentifiedImageError, OSError) as error:
        raise UploadError("not a readable image") from error
