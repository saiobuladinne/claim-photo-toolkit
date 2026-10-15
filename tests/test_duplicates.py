from __future__ import annotations

from pathlib import Path

from claim_photo_toolkit import inspect
from claim_photo_toolkit.duplicates import hamming_distance, match_hashes
from tests.helpers import save_image, stripes


def test_hamming_distance_and_groups() -> None:
    assert hamming_distance("0000000000000000", "0000000000000000") == 0
    assert hamming_distance("0000000000000000", "0000000000000001") == 1
    items = (
        ("a.jpg", "0000000000000000"),
        ("b.jpg", "0000000000000001"),
        ("c.jpg", "0000000000000003"),
        ("d.jpg", "ffffffffffffffff"),
    )
    matches, groups = match_hashes(items, distance_threshold=1)
    assert [match.path for match in matches[0]] == ["b.jpg"]
    assert matches[0][0].distance == 1
    assert [match.path for match in matches[1]] == ["a.jpg", "c.jpg"]
    assert [match.path for match in matches[3]] == []
    assert len(groups) == 1
    assert groups[0].paths == ("a.jpg", "b.jpg", "c.jpg")
    assert groups[0].max_distance == 1

    exact, exact_groups = match_hashes(items[:2], distance_threshold=0)
    assert exact[0] == ()
    assert exact_groups == ()


def test_identical_files_are_duplicates_and_different_photos_are_not(tmp_path: Path) -> None:
    original = tmp_path / "front.png"
    stripes(horizontal=True).save(original)
    copy = tmp_path / "front-copy.png"
    copy.write_bytes(original.read_bytes())
    other = tmp_path / "side.png"
    stripes(horizontal=False).save(other)

    report = inspect([original, copy, other], run_integrity=False, duplicate_distance=0)
    front = report.photos[0].duplicates
    assert front.flagged
    assert front.algorithm == "phash"
    assert front.image_hash
    assert [(match.path, match.distance) for match in front.matches] == [(str(copy), 0)]
    assert report.photos[1].duplicates.matches[0].path == str(original)
    assert report.photos[2].duplicates.matches == ()
    assert report.photos[2].duplicates.status == "ok"
    assert report.duplicate_groups[0].paths == (str(original), str(copy))
    assert report.duplicate_groups[0].max_distance == 0
    assert "duplicate" in report.photos[0].flags
    assert "duplicate" not in report.photos[2].flags


def test_single_photo_has_no_duplicate(tmp_path: Path) -> None:
    path = save_image(tmp_path / "only.jpg")
    report = inspect(path, run_integrity=False)
    assert report.photos[0].duplicates.status == "ok"
    assert report.photos[0].duplicates.matches == ()
    assert report.duplicate_groups == ()
