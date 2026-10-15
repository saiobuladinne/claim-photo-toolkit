from __future__ import annotations

import json
from pathlib import Path

import pytest

from claim_photo_toolkit.cli import main
from claim_photo_toolkit.errors import IntegrityModelUnavailable
from tests.helpers import save_image


def test_json_report(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    path = save_image(tmp_path / "front.jpg")
    code = main(["inspect", str(path), "--claim-date", "2026-03-01", "--json", "--skip-integrity"])
    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    photo = payload["photos"][0]
    assert payload["claim_date"] == "2026-03-01"
    assert payload["photo_count"] == 1
    assert photo["path"] == str(path)
    assert photo["integrity"]["status"] == "skipped"
    assert photo["damage"]["status"] == "unavailable"
    assert photo["duplicates"]["algorithm"] == "phash"
    assert photo["metadata"]["flags"] == ["missing_exif", "stripped_metadata"]
    assert photo["flagged"] is True


def test_text_report(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    path = save_image(tmp_path / "front.jpg")
    code = main(["inspect", str(path), "--skip-integrity"])
    assert code == 0
    text = capsys.readouterr().out
    assert str(path) in text
    assert "integrity: skipped" in text
    assert "damage: unavailable" in text
    assert "metadata: flagged missing_exif, stripped_metadata" in text
    assert text.endswith("claim: 1/1 photos flagged\n")


def test_bad_claim_date_exits_with_usage(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    path = save_image(tmp_path / "front.jpg")
    code = main(["inspect", str(path), "--claim-date", "yesterday", "--skip-integrity"])
    assert code == 2
    assert "claim_date" in capsys.readouterr().err


def test_missing_command_exits() -> None:
    with pytest.raises(SystemExit) as caught:
        main([])
    assert caught.value.code != 0


def test_default_integrity_is_unavailable_without_the_extra(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def missing() -> object:
        raise IntegrityModelUnavailable(
            "Integrity checks need the ml extra: pip install 'claim-photo-toolkit[ml]'"
        )

    monkeypatch.setattr("claim_photo_toolkit.integrity.load_integrity_model", missing)
    path = save_image(tmp_path / "front.jpg")
    code = main(["inspect", str(path), "--json"])
    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    integrity = payload["photos"][0]["integrity"]
    assert integrity["status"] == "unavailable"
    assert "ml extra" in integrity["detail"]
