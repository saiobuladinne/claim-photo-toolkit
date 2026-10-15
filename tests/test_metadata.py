from __future__ import annotations

from datetime import date, datetime
from pathlib import Path

from PIL import Image
from PIL.ExifTags import GPS, IFD, Base

from claim_photo_toolkit.metadata import ExifFacts, assess_metadata, read_exif
from claim_photo_toolkit.models import GpsLocation
from tests.helpers import camera_exif, save_image


def test_clean_camera_jpeg_is_not_flagged(tmp_path: Path) -> None:
    path = save_image(tmp_path / "clean.jpg", exif=camera_exif())
    with Image.open(path) as image:
        facts = read_exif(image)
    result = assess_metadata(facts, claim_date=date(2026, 2, 1))
    assert result.status == "ok"
    assert result.flags == ()
    assert result.timestamp == datetime(2026, 1, 10, 9, 30)
    assert result.device == "Apple iPhone 15"
    assert result.software == "iOS 17.4"
    assert result.gps == GpsLocation(37.775, -122.419444)
    assert result.image_format == "JPEG"


def test_device_string_does_not_repeat_the_make() -> None:
    facts = ExifFacts(
        has_exif=True,
        make="Apple",
        model="Apple iPhone 15",
        timestamp=datetime(2026, 1, 1),
    )
    result = assess_metadata(facts, claim_date=None)
    assert result.device == "Apple iPhone 15"
    assert result.flags == ()


def test_bare_jpeg_is_missing_and_stripped(tmp_path: Path) -> None:
    path = save_image(tmp_path / "bare.jpg")
    with Image.open(path) as image:
        result = assess_metadata(read_exif(image), claim_date=None)
    assert result.flags == ("missing_exif", "stripped_metadata")
    assert result.flagged


def test_png_without_exif_is_missing_but_not_stripped(tmp_path: Path) -> None:
    path = save_image(tmp_path / "shot.png", image_format="PNG")
    with Image.open(path) as image:
        result = assess_metadata(read_exif(image), claim_date=None)
    assert result.flags == ("missing_exif",)
    assert result.image_format == "PNG"


def test_editing_software_and_partial_strip(tmp_path: Path) -> None:
    exif = Image.Exif()
    exif[Base.Software] = "Adobe Photoshop 25.0"
    path = save_image(tmp_path / "edited.jpg", exif=exif)
    with Image.open(path) as image:
        result = assess_metadata(read_exif(image), claim_date=date(2026, 1, 1))
    assert result.software == "Adobe Photoshop 25.0"
    assert result.flags == ("stripped_metadata", "editing_software")


def test_timestamp_after_claim_uses_dates_and_datetimes() -> None:
    on_claim_date = ExifFacts(
        has_exif=True,
        image_format="JPEG",
        timestamp=datetime(2026, 3, 1, 23, 59),
        make="Apple",
        model="iPhone 15",
    )
    assert (
        "timestamp_after_claim"
        not in assess_metadata(on_claim_date, claim_date=date(2026, 3, 1)).flags
    )

    next_day = ExifFacts(
        has_exif=True,
        image_format="JPEG",
        timestamp=datetime(2026, 3, 2, 0, 0),
        make="Apple",
        model="iPhone 15",
    )
    assert assess_metadata(next_day, claim_date=date(2026, 3, 1)).flags == (
        "timestamp_after_claim",
    )

    same_instant = assess_metadata(
        next_day,
        claim_date=datetime(2026, 3, 2, 0, 0),
    )
    assert "timestamp_after_claim" not in same_instant.flags
    later = assess_metadata(next_day, claim_date=datetime(2026, 3, 1, 23, 59, 59))
    assert later.flags == ("timestamp_after_claim",)


def test_unreadable_timestamp_and_invalid_gps(tmp_path: Path) -> None:
    exif = Image.Exif()
    exif[Base.Make] = "Apple"
    exif[Base.Model] = "iPhone 15"
    exif[Base.DateTime] = "yesterday"
    gps = exif.get_ifd(IFD.GPSInfo)
    gps[GPS.GPSLatitudeRef] = "Q"
    gps[GPS.GPSLatitude] = (1.0, 2.0, 3.0)
    gps[GPS.GPSLongitudeRef] = "E"
    gps[GPS.GPSLongitude] = (4.0, 5.0, 6.0)
    path = save_image(tmp_path / "odd.jpg", exif=exif)
    with Image.open(path) as image:
        result = assess_metadata(read_exif(image), claim_date=None)
    assert "unreadable_timestamp" in result.flags
    assert "invalid_gps" in result.flags
    assert result.gps is None
    assert result.device == "Apple iPhone 15"


def test_editor_names_are_case_insensitive() -> None:
    for software in ("GIMP 2.10", "stable diffusion webui", "Snapseed", "Adobe Firefly"):
        result = assess_metadata(
            ExifFacts(
                has_exif=True,
                image_format="JPEG",
                timestamp=datetime(2026, 1, 1),
                make="Apple",
                software=software,
            ),
            claim_date=None,
        )
        assert "editing_software" in result.flags
    phone = assess_metadata(
        ExifFacts(
            has_exif=True,
            image_format="JPEG",
            timestamp=datetime(2026, 1, 1),
            make="Google",
            model="Pixel 8",
            software="HDR+ 1.0.123",
        ),
        claim_date=None,
    )
    assert phone.flags == ()
