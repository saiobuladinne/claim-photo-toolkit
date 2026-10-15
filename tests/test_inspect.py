from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest
from PIL import Image

from claim_photo_toolkit import __version__, inspect
from claim_photo_toolkit.models import DamageAssessment, PartFinding
from tests.helpers import camera_exif, save_image, scores, stripes


def test_string_path_and_empty_input(tmp_path: Path) -> None:
    path = save_image(tmp_path / "a.jpg")
    report = inspect(str(path), claim_date="2026-03-01T00:00:00Z", run_integrity=False)
    assert report.photos[0].path == str(path)
    assert report.claim_date == datetime(2026, 3, 1, tzinfo=timezone.utc)
    assert report.version == __version__

    empty = inspect([], run_integrity=False)
    assert empty.photos == ()
    assert empty.flagged_count == 0
    assert empty.error_count == 0


def test_bad_arguments_are_rejected(tmp_path: Path) -> None:
    path = save_image(tmp_path / "a.jpg")
    with pytest.raises(ValueError, match="integrity_threshold"):
        inspect(path, integrity_threshold=1.5, run_integrity=False)
    with pytest.raises(TypeError, match="integrity_threshold"):
        inspect(path, integrity_threshold=True, run_integrity=False)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="duplicate_distance"):
        inspect(path, duplicate_distance=-1, run_integrity=False)
    with pytest.raises(ValueError, match="claim_date"):
        inspect(path, claim_date="tomorrow", run_integrity=False)


def test_unreadable_photo_is_an_error_and_does_not_stop_the_claim(tmp_path: Path) -> None:
    good = save_image(tmp_path / "good.jpg")
    missing = tmp_path / "missing.jpg"
    junk = tmp_path / "notes.jpg"
    junk.write_text("not an image")
    report = inspect([good, missing, junk], integrity_model=scores(0.2))
    assert report.error_count == 2
    assert report.photos[0].error is None
    assert report.photos[0].integrity.status == "ok"
    assert report.photos[1].integrity.status == "error"
    assert report.photos[1].damage.status == "error"
    assert report.photos[1].duplicates.status == "error"
    assert report.photos[1].metadata.status == "error"
    assert report.photos[1].error is not None
    assert "cannot read image" in report.photos[1].error
    payload = json.loads(json.dumps(report.to_dict()))
    assert payload["photo_count"] == 3
    assert payload["error_count"] == 2


def test_report_combines_every_check(tmp_path: Path) -> None:
    flagged = tmp_path / "flagged.jpg"
    stripes(horizontal=True).save(
        flagged,
        exif=camera_exif(when="2026:05:01 08:00:00", software="Adobe Photoshop 25.0"),
    )
    clear = tmp_path / "clear.jpg"
    dark = Image.new("RGB", (128, 128))
    dark.save(clear, exif=camera_exif())
    copy = tmp_path / "flagged-copy.jpg"
    copy.write_bytes(flagged.read_bytes())

    def model(image: Image.Image) -> list[dict[str, float | str]]:
        total = sum(image.getchannel("R").tobytes())
        return scores(0.91 if total > 1_000_000 else 0.05)(image)

    class BumperModel:
        name = "bumper-model"

        def assess(self, image: Image.Image, image_path: Path) -> DamageAssessment:
            del image
            if image_path.name == "clear.jpg":
                return DamageAssessment(
                    parts=(PartFinding("hood", "minor", 0.4),),
                    summary="Light scuff on the hood.",
                )
            return DamageAssessment()

    report = inspect(
        [flagged, clear, copy],
        claim_date="2026-04-01",
        integrity_model=model,
        damage_backend=BumperModel(),
        duplicate_distance=0,
    )
    assert report.photos[0].integrity.flagged
    assert "metadata:editing_software" in report.photos[0].flags
    assert "metadata:timestamp_after_claim" in report.photos[0].flags
    assert report.photos[1].integrity.status == "ok"
    assert report.photos[1].flags == ()
    assert report.photos[1].damage.parts[0].part == "hood"
    assert report.photos[0].duplicates.flagged
    assert report.photos[1].duplicates.matches == ()
    assert report.flagged_count == 2
    body = report.to_dict()
    assert body["duplicate_groups"] == [{"paths": [str(flagged), str(copy)], "max_distance": 0}]
    integrity = body["photos"][0]["integrity"]
    assert isinstance(integrity, dict)
    assert str(integrity["model_id"]).endswith("claim-photo-integrity-detector-v5")
    damage = body["photos"][1]["damage"]
    assert isinstance(damage, dict)
    parts = damage["parts"]
    assert isinstance(parts, list)
    assert parts[0]["severity"] == "minor"


def test_custom_model_and_skip_do_not_load_the_checkpoint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = save_image(tmp_path / "a.jpg")
    calls = {"n": 0}

    def explode() -> object:
        calls["n"] += 1
        raise AssertionError("checkpoint should not load")

    monkeypatch.setattr("claim_photo_toolkit.integrity._build_pipeline", explode)
    inspect(path, run_integrity=False)
    inspect(path, integrity_model=scores(0.2))
    assert calls["n"] == 0
