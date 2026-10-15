"""Run every check and build one claim report."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import date, datetime
from pathlib import Path

from PIL import Image

from claim_photo_toolkit import damage as damage_check
from claim_photo_toolkit import duplicates as duplicate_check
from claim_photo_toolkit import integrity as integrity_check
from claim_photo_toolkit import metadata as metadata_check
from claim_photo_toolkit.damage import DamageBackend, UnavailableDamageBackend
from claim_photo_toolkit.errors import DamageModelUnavailable, IntegrityModelUnavailable
from claim_photo_toolkit.integrity import IntegrityModel
from claim_photo_toolkit.metadata import ExifFacts
from claim_photo_toolkit.models import (
    ClaimReport,
    DamageResult,
    DuplicateGroup,
    DuplicateResult,
    IntegrityResult,
    MetadataResult,
    PhotoReport,
)


def inspect(
    paths: str | Path | Sequence[str | Path],
    *,
    claim_date: date | datetime | str | None = None,
    damage_backend: DamageBackend | None = None,
    integrity_model: IntegrityModel | None = None,
    integrity_threshold: float = 0.5,
    run_integrity: bool = True,
    duplicate_distance: int = 8,
    hash_size: int = 8,
) -> ClaimReport:
    """Check claim photos and return a per-photo and per-claim report.

    A string path is one photo. Integrity loads
    ``obuladinnesai/claim-photo-integrity-detector-v5`` on first use unless a
    model is passed or ``run_integrity`` is false. Damage stays unavailable
    until ``damage_backend`` is set.
    """
    parsed_claim_date = parse_claim_date(claim_date)
    _validate_options(integrity_threshold, duplicate_distance, hash_size)
    backend = UnavailableDamageBackend() if damage_backend is None else damage_backend
    _validate_backend(backend)
    loaded = [_load(path) for path in _coerce_paths(paths)]
    integrity = _integrity_results(
        loaded,
        model=integrity_model,
        threshold=integrity_threshold,
        enabled=run_integrity,
    )
    damage = _damage_results(loaded, backend)
    duplicates, duplicate_groups = _duplicate_section(
        loaded,
        distance=duplicate_distance,
        hash_size=hash_size,
    )
    metadata = _metadata_results(loaded, claim_date=parsed_claim_date)
    photos = tuple(
        _photo(item, item_integrity, item_damage, item_duplicates, item_metadata)
        for item, item_integrity, item_damage, item_duplicates, item_metadata in zip(
            loaded, integrity, damage, duplicates, metadata, strict=True
        )
    )
    return ClaimReport(
        version=_version(),
        claim_date=parsed_claim_date,
        photos=photos,
        duplicate_groups=duplicate_groups,
        flagged_count=sum(1 for photo in photos if photo.flagged),
        error_count=sum(1 for photo in photos if photo.error is not None),
    )


def parse_claim_date(value: date | datetime | str | None) -> date | datetime | None:
    """Accept a date, datetime, or ISO-8601 string."""
    if value is None or isinstance(value, datetime):
        return value
    if isinstance(value, date):
        return value
    if not isinstance(value, str):
        raise TypeError("claim_date must be a date, datetime, or ISO-8601 string")
    text = value.strip()
    if not text:
        raise ValueError("claim_date must be an ISO-8601 date or datetime")
    if text.endswith(("Z", "z")):
        text = f"{text[:-1]}+00:00"
    if "T" not in text and " " in text:
        text = text.replace(" ", "T", 1)
    if "T" in text:
        try:
            return datetime.fromisoformat(text)
        except ValueError as exc:
            raise ValueError("claim_date must be an ISO-8601 date or datetime") from exc
    try:
        return date.fromisoformat(text)
    except ValueError as exc:
        raise ValueError("claim_date must be an ISO-8601 date or datetime") from exc


class _Loaded:
    def __init__(
        self,
        path: str,
        image: Image.Image | None,
        facts: ExifFacts | None,
        error: str | None,
    ) -> None:
        self.path = path
        self.image = image
        self.facts = facts
        self.error = error


def _coerce_paths(paths: str | Path | Sequence[str | Path]) -> list[str]:
    if isinstance(paths, (str, Path)):
        return [str(paths)]
    return [str(path) for path in paths]


def _validate_options(threshold: float, distance: int, hash_size: int) -> None:
    if isinstance(threshold, bool) or not isinstance(threshold, (int, float)):
        raise TypeError("integrity_threshold must be a number")
    if threshold != threshold or not 0.0 <= float(threshold) <= 1.0:
        raise ValueError("integrity_threshold must be between 0 and 1")
    if isinstance(distance, bool) or not isinstance(distance, int) or distance < 0:
        raise ValueError("duplicate_distance must be a non-negative integer")
    if isinstance(hash_size, bool) or not isinstance(hash_size, int) or hash_size < 2:
        raise ValueError("hash_size must be an integer >= 2")


def _validate_backend(backend: DamageBackend) -> None:
    if not isinstance(backend, damage_check.DamageBackend):
        raise TypeError("damage_backend must provide name and assess()")
    if not isinstance(backend.name, str) or not backend.name.strip():
        raise ValueError("damage_backend.name must be a non-empty string")


def _load(path: str) -> _Loaded:
    try:
        with Image.open(path) as image:
            image.load()
            facts = metadata_check.read_exif(image)
            rgb = image.convert("RGB")
            rgb.load()
    except (OSError, ValueError, Image.DecompressionBombError) as exc:
        return _Loaded(path, None, None, f"cannot read image: {exc}")
    return _Loaded(path, rgb, facts, None)


def _integrity_results(
    loaded: Sequence[_Loaded],
    *,
    model: IntegrityModel | None,
    threshold: float,
    enabled: bool,
) -> list[IntegrityResult]:
    if not enabled:
        skipped = integrity_check.skipped_integrity("integrity check skipped")
        return [
            integrity_check.error_integrity(item.error) if item.error else skipped
            for item in loaded
        ]
    resolved = model
    load_error: str | None = None
    if resolved is None and any(item.image is not None for item in loaded):
        try:
            resolved = integrity_check.load_integrity_model()
        except IntegrityModelUnavailable as exc:
            load_error = str(exc)
    results: list[IntegrityResult] = []
    for item in loaded:
        if item.error or item.image is None:
            results.append(integrity_check.error_integrity(item.error or "cannot read image"))
            continue
        if load_error is not None or resolved is None:
            detail = load_error or "integrity model is not loaded"
            results.append(integrity_check.unavailable_integrity(detail))
            continue
        try:
            results.append(
                integrity_check.classify_image(item.image.copy(), resolved, threshold=threshold)
            )
        except Exception as exc:
            results.append(
                integrity_check.error_integrity(
                    f"integrity model failed: {exc}",
                    threshold=threshold,
                )
            )
    return results


def _damage_results(loaded: Sequence[_Loaded], backend: DamageBackend) -> list[DamageResult]:
    results: list[DamageResult] = []
    for item in loaded:
        if item.error or item.image is None:
            detail = item.error or "cannot read image"
            results.append(damage_check.error_damage(detail, backend=None))
            continue
        try:
            assessment = backend.assess(item.image.copy(), Path(item.path))
        except DamageModelUnavailable as exc:
            results.append(damage_check.unavailable_damage(str(exc)))
            continue
        except Exception as exc:
            results.append(damage_check.error_damage(str(exc), backend=backend.name))
            continue
        if not isinstance(assessment, damage_check.DamageAssessment):
            results.append(
                damage_check.error_damage(
                    "damage backend must return a DamageAssessment",
                    backend=backend.name,
                )
            )
            continue
        results.append(damage_check.result_from_assessment(backend.name, assessment))
    return results


def _duplicate_section(
    loaded: Sequence[_Loaded],
    *,
    distance: int,
    hash_size: int,
) -> tuple[list[DuplicateResult], tuple[DuplicateGroup, ...]]:
    digests: list[str | None] = []
    hash_errors: list[str | None] = []
    hashed: list[tuple[str, str]] = []
    hash_indexes: list[int] = []
    for index, item in enumerate(loaded):
        if item.error or item.image is None:
            digests.append(None)
            hash_errors.append(item.error or "cannot read image")
            continue
        try:
            digest = duplicate_check.perceptual_hash(item.image, hash_size=hash_size)
        except duplicate_check.HashError as exc:
            digests.append(None)
            hash_errors.append(f"cannot hash image: {exc}")
            continue
        digests.append(digest)
        hash_errors.append(None)
        hashed.append((item.path, digest))
        hash_indexes.append(index)
    matched, groups = duplicate_check.match_hashes(hashed, distance_threshold=distance)
    by_index = dict(zip(hash_indexes, matched, strict=True))
    results: list[DuplicateResult] = []
    for index, image_hash in enumerate(digests):
        if image_hash is None:
            detail = hash_errors[index] or "cannot hash image"
            results.append(duplicate_check.error_duplicates(detail, distance_threshold=distance))
            continue
        results.append(
            duplicate_check.ok_duplicates(
                image_hash=image_hash,
                distance_threshold=distance,
                matches=by_index.get(index, ()),
            )
        )
    return results, groups


def _version() -> str:
    from claim_photo_toolkit import __version__

    return __version__


def _metadata_results(
    loaded: Sequence[_Loaded],
    *,
    claim_date: date | datetime | None,
) -> list[MetadataResult]:
    results: list[MetadataResult] = []
    for item in loaded:
        if item.facts is None:
            results.append(metadata_check.error_metadata(item.error or "cannot read image"))
            continue
        results.append(metadata_check.assess_metadata(item.facts, claim_date=claim_date))
    return results


def _photo(
    item: _Loaded,
    integrity: IntegrityResult,
    damage: DamageResult,
    duplicates: DuplicateResult,
    metadata: MetadataResult,
) -> PhotoReport:
    flags: list[str] = [f"integrity:{flag}" for flag in integrity.flags]
    if duplicates.flagged:
        flags.append("duplicate")
    flags.extend(f"metadata:{flag}" for flag in metadata.flags)
    return PhotoReport(
        path=item.path,
        flagged=bool(flags),
        flags=tuple(flags),
        error=item.error,
        integrity=integrity,
        damage=damage,
        duplicates=duplicates,
        metadata=metadata,
    )
