from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from PIL import Image
from PIL.ExifTags import GPS, IFD, Base
from PIL.TiffImagePlugin import IFDRational

FakeModel = Callable[[Image.Image], list[dict[str, float | str]]]


def save_image(
    path: Path,
    *,
    color: tuple[int, int, int] = (20, 40, 60),
    size: tuple[int, int] = (64, 64),
    exif: Image.Exif | None = None,
    image_format: str = "JPEG",
) -> Path:
    image = Image.new("RGB", size, color)
    if exif is None:
        image.save(path, image_format)
    else:
        image.save(path, image_format, exif=exif)
    return path


def stripes(horizontal: bool) -> Image.Image:
    image = Image.new("RGB", (128, 128))
    for y in range(128):
        for x in range(128):
            band = (y if horizontal else x) // 8
            image.putpixel((x, y), (255, 255, 255) if band % 2 == 0 else (0, 0, 0))
    return image


def camera_exif(
    *,
    when: str = "2026:01:10 09:30:00",
    make: str = "Apple",
    model: str = "iPhone 15",
    software: str = "iOS 17.4",
    latitude: tuple[float, float, float] | None = (37.0, 46.0, 30.0),
    latitude_ref: str = "N",
    longitude: tuple[float, float, float] | None = (122.0, 25.0, 10.0),
    longitude_ref: str = "W",
) -> Image.Exif:
    exif = Image.Exif()
    if make:
        exif[Base.Make] = make
    if model:
        exif[Base.Model] = model
    if software:
        exif[Base.Software] = software
    if when:
        exif[Base.DateTime] = when
        exif.get_ifd(IFD.Exif)[Base.DateTimeOriginal] = when
    if latitude is not None and longitude is not None:
        gps = exif.get_ifd(IFD.GPSInfo)
        gps[GPS.GPSLatitudeRef] = latitude_ref
        gps[GPS.GPSLatitude] = tuple(IFDRational(int(part), 1) for part in latitude)
        gps[GPS.GPSLongitudeRef] = longitude_ref
        gps[GPS.GPSLongitude] = tuple(IFDRational(int(part), 1) for part in longitude)
    return exif


def scores(ai_generated: float, real: float | None = None) -> FakeModel:
    real_score = 1.0 - ai_generated if real is None else real

    def _model(_image: Image.Image) -> list[dict[str, float | str]]:
        return [
            {"label": "real", "score": real_score},
            {"label": "ai_generated", "score": ai_generated},
        ]

    return _model
