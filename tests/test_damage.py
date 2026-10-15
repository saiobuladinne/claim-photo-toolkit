from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from claim_photo_toolkit import inspect
from claim_photo_toolkit.models import DamageAssessment, PartFinding
from tests.helpers import camera_exif, save_image, scores


def test_part_finding_normalizes_and_rejects_bad_values() -> None:
    finding = PartFinding(part="  front bumper ", severity="moderate", confidence=1)
    assert finding.part == "front bumper"
    assert finding.confidence == 1.0
    with pytest.raises(ValueError, match="severity"):
        PartFinding(part="door", severity="wrecked")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="confidence"):
        PartFinding(part="door", severity="minor", confidence=1.2)
    with pytest.raises(ValueError, match="part"):
        PartFinding(part="   ", severity="none")


def test_default_backend_is_unavailable(tmp_path: Path) -> None:
    path = save_image(tmp_path / "a.jpg", exif=camera_exif())
    report = inspect(path, run_integrity=False)
    damage = report.photos[0].damage
    assert damage.status == "unavailable"
    assert damage.backend == "unavailable"
    assert damage.parts == ()
    assert damage.detail is not None
    assert "damage_backend" in damage.detail
    assert not report.photos[0].flagged


def test_custom_backend_findings_are_reported(tmp_path: Path) -> None:
    path = save_image(tmp_path / "a.jpg")
    seen: list[Path] = []

    class BumperModel:
        name = "bumper-model"

        def assess(self, image: Image.Image, image_path: Path) -> DamageAssessment:
            assert image.mode == "RGB"
            seen.append(image_path)
            return DamageAssessment(
                parts=(PartFinding(part="front_bumper", severity="severe", confidence=0.82),),
                summary="Front bumper is crushed.",
            )

    report = inspect([path], run_integrity=False, damage_backend=BumperModel())
    damage = report.photos[0].damage
    assert damage.status == "ok"
    assert damage.backend == "bumper-model"
    assert damage.summary == "Front bumper is crushed."
    assert damage.parts[0].part == "front_bumper"
    assert damage.parts[0].severity == "severe"
    assert seen == [path]
    assert "damage" not in "".join(report.photos[0].flags)


def test_backend_errors_stay_on_the_photo(tmp_path: Path) -> None:
    path = save_image(tmp_path / "a.jpg")

    class Broken:
        name = "broken"

        def assess(self, image: Image.Image, image_path: Path) -> DamageAssessment:
            del image, image_path
            raise RuntimeError("weights missing")

    report = inspect(
        path,
        integrity_model=scores(0.1),
        damage_backend=Broken(),
    )
    assert report.photos[0].damage.status == "error"
    assert report.photos[0].damage.detail == "weights missing"
    assert report.photos[0].integrity.status == "ok"


def test_backend_must_be_named() -> None:
    class Nameless:
        name = "  "

        def assess(self, image: Image.Image, image_path: Path) -> DamageAssessment:
            del image, image_path
            return DamageAssessment()

    with pytest.raises(ValueError, match="name"):
        inspect([], damage_backend=Nameless(), run_integrity=False)
