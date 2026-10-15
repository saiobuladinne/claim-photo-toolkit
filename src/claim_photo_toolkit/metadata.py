"""EXIF triage for claim photos."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date, datetime

from PIL import Image
from PIL.ExifTags import GPS, IFD, Base

from claim_photo_toolkit.models import GpsLocation, MetadataResult

_EXIF_FORMATS = frozenset({"JPEG", "TIFF", "MPO"})
_EDITOR_MARKERS = (
    "photoshop",
    "lightroom",
    "gimp",
    "affinity photo",
    "paint.net",
    "photoscape",
    "snapseed",
    "pixelmator",
    "capture one",
    "luminar",
    "darktable",
    "rawtherapee",
    "acdsee",
    "photopea",
    "facetune",
    "meitu",
    "picsart",
    "canva",
    "fotor",
    "remini",
    "stable diffusion",
    "midjourney",
    "dall-e",
    "dall·e",
    "dalle",
    "adobe firefly",
    "firefly",
    "comfyui",
    "automatic1111",
    "invokeai",
)
_TIMESTAMP_FORMATS = ("%Y:%m:%d %H:%M:%S", "%Y-%m-%d %H:%M:%S")


@dataclass(frozen=True, slots=True)
class ExifFacts:
    """Plain EXIF values read from one image."""

    has_exif: bool = False
    image_format: str | None = None
    timestamp: datetime | None = None
    timestamp_unreadable: bool = False
    make: str | None = None
    model: str | None = None
    software: str | None = None
    gps: GpsLocation | None = None
    gps_invalid: bool = False


def read_exif(image: Image.Image) -> ExifFacts:
    """Read capture time, GPS, device, and software from an open image."""
    exif = image.getexif()
    exif_ifd = exif.get_ifd(IFD.Exif)
    gps_ifd = exif.get_ifd(IFD.GPSInfo)
    pointers = {int(IFD.Exif), int(IFD.GPSInfo)}
    own_tags = [int(tag) for tag in exif if int(tag) not in pointers]
    timestamp, unreadable = _timestamp(exif, exif_ifd)
    gps, gps_invalid = _gps(gps_ifd)
    return ExifFacts(
        has_exif=bool(own_tags or exif_ifd or gps_ifd),
        image_format=image.format,
        timestamp=timestamp,
        timestamp_unreadable=unreadable,
        make=_text(exif.get(Base.Make)),
        model=_text(exif.get(Base.Model)),
        software=_text(exif.get(Base.Software)),
        gps=gps,
        gps_invalid=gps_invalid,
    )


def assess_metadata(facts: ExifFacts, *, claim_date: date | datetime | None) -> MetadataResult:
    """Flag missing, stripped, and suspicious EXIF fields."""
    flags: list[str] = []
    if not facts.has_exif:
        flags.append("missing_exif")
    usable_time = facts.timestamp is not None
    has_device = facts.make is not None or facts.model is not None
    image_format = (facts.image_format or "").upper()
    if image_format in _EXIF_FORMATS and not usable_time and not has_device:
        flags.append("stripped_metadata")
    if facts.software is not None and _is_editor(facts.software):
        flags.append("editing_software")
    if (
        facts.timestamp is not None
        and claim_date is not None
        and _after_claim(facts.timestamp, claim_date)
    ):
        flags.append("timestamp_after_claim")
    if facts.timestamp_unreadable:
        flags.append("unreadable_timestamp")
    if facts.gps_invalid:
        flags.append("invalid_gps")
    flagged = bool(flags)
    return MetadataResult(
        status="flagged" if flagged else "ok",
        flagged=flagged,
        timestamp=facts.timestamp,
        gps=facts.gps,
        device=_device(facts.make, facts.model),
        software=facts.software,
        image_format=facts.image_format,
        flags=tuple(flags),
        detail=None,
    )


def error_metadata(detail: str) -> MetadataResult:
    return MetadataResult(
        status="error",
        flagged=False,
        timestamp=None,
        gps=None,
        device=None,
        software=None,
        image_format=None,
        flags=(),
        detail=detail,
    )


def _timestamp(
    exif: Mapping[int, object],
    exif_ifd: Mapping[int, object],
) -> tuple[datetime | None, bool]:
    raw_values = (
        exif_ifd.get(int(Base.DateTimeOriginal)),
        exif_ifd.get(int(Base.DateTimeDigitized)),
        exif.get(int(Base.DateTime)),
    )
    saw_raw = False
    for raw in raw_values:
        text = _text(raw)
        if text is None:
            continue
        saw_raw = True
        parsed = _parse_timestamp(text)
        if parsed is not None:
            return parsed, False
    return None, saw_raw


def _parse_timestamp(text: str) -> datetime | None:
    for fmt in _TIMESTAMP_FORMATS:
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            continue
    return None


def _gps(gps_ifd: Mapping[int, object]) -> tuple[GpsLocation | None, bool]:
    if not gps_ifd:
        return None, False
    try:
        latitude = _axis(
            gps_ifd.get(int(GPS.GPSLatitude)),
            gps_ifd.get(int(GPS.GPSLatitudeRef)),
            axis="latitude",
        )
        longitude = _axis(
            gps_ifd.get(int(GPS.GPSLongitude)),
            gps_ifd.get(int(GPS.GPSLongitudeRef)),
            axis="longitude",
        )
    except (TypeError, ValueError, ZeroDivisionError, OverflowError):
        return None, True
    if latitude is None or longitude is None:
        return None, True
    if not -90.0 <= latitude <= 90.0 or not -180.0 <= longitude <= 180.0:
        return None, True
    return GpsLocation(round(latitude, 6), round(longitude, 6)), False


def _axis(value: object, ref: object, *, axis: str) -> float | None:
    if value is None and ref is None:
        return None
    if value is None:
        raise ValueError(f"missing {axis}")
    magnitude = abs(_degrees(value))
    direction = _text(ref)
    if direction is None:
        raise ValueError(f"missing {axis} reference")
    direction = direction.upper()
    if axis == "latitude" and direction not in {"N", "S"}:
        raise ValueError("bad latitude reference")
    if axis == "longitude" and direction not in {"E", "W"}:
        raise ValueError("bad longitude reference")
    if direction in {"S", "W"}:
        return -magnitude
    return magnitude


def _degrees(value: object) -> float:
    if isinstance(value, (list, tuple)):
        if len(value) != 3:
            raise ValueError("GPS coordinate must have three components")
        degrees, minutes, seconds = (_number(part) for part in value)
        return degrees + minutes / 60.0 + seconds / 3600.0
    return _number(value)


def _number(value: object) -> float:
    if isinstance(value, bool):
        raise TypeError("boolean is not a coordinate")
    if isinstance(value, (int, float)):
        return float(value)
    numerator = getattr(value, "numerator", None)
    denominator = getattr(value, "denominator", None)
    if isinstance(numerator, int) and isinstance(denominator, int):
        if denominator == 0:
            raise ZeroDivisionError
        return numerator / denominator
    raise TypeError("unsupported EXIF number")


def _text(value: object) -> str | None:
    if isinstance(value, bytes):
        value = value.decode("utf-8", "replace")
    if not isinstance(value, str):
        return None
    text = value.replace("\x00", "").strip()
    return text or None


def _device(make: str | None, model: str | None) -> str | None:
    if make and model:
        if model.casefold().startswith(make.casefold()):
            return model
        return f"{make} {model}"
    return make or model


def _is_editor(software: str) -> bool:
    folded = software.casefold()
    return any(marker in folded for marker in _EDITOR_MARKERS)


def _after_claim(captured: datetime, claim: date | datetime) -> bool:
    if isinstance(claim, datetime):
        return captured.replace(tzinfo=None) > claim.replace(tzinfo=None)
    return captured.date() > claim
