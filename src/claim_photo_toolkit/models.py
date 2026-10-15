"""Report records returned by :func:`claim_photo_toolkit.inspect`."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from typing import Literal

CheckStatus = Literal["ok", "flagged", "skipped", "unavailable", "error"]
Severity = Literal["none", "minor", "moderate", "severe"]
SEVERITIES: frozenset[str] = frozenset({"none", "minor", "moderate", "severe"})


@dataclass(frozen=True, slots=True)
class IntegrityResult:
    """Image-classifier scores for one photo."""

    status: CheckStatus
    flagged: bool
    label: str | None
    scores: dict[str, float]
    threshold: float | None
    model_id: str
    flags: tuple[str, ...] = ()
    detail: str | None = None


@dataclass(frozen=True, slots=True)
class PartFinding:
    """One vehicle part and how badly it is damaged."""

    part: str
    severity: Severity
    confidence: float | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.part, str):
            raise TypeError("part must be a string")
        part = self.part.strip()
        if not part:
            raise ValueError("part must be a non-empty string")
        if part != self.part:
            object.__setattr__(self, "part", part)
        if self.severity not in SEVERITIES:
            names = ", ".join(sorted(SEVERITIES))
            raise ValueError(f"severity must be one of: {names}")
        confidence = self.confidence
        if confidence is None:
            return
        if (
            isinstance(confidence, bool)
            or not isinstance(confidence, (int, float))
            or confidence != confidence
            or not 0.0 <= float(confidence) <= 1.0
        ):
            raise ValueError("confidence must be between 0 and 1")
        if not isinstance(confidence, float):
            object.__setattr__(self, "confidence", float(confidence))


@dataclass(frozen=True, slots=True)
class DamageAssessment:
    """Findings a damage backend returns for one photo."""

    parts: tuple[PartFinding, ...] = ()
    summary: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.parts, tuple) or any(
            not isinstance(part, PartFinding) for part in self.parts
        ):
            raise TypeError("parts must be a tuple of PartFinding")
        if self.summary is not None and not isinstance(self.summary, str):
            raise TypeError("summary must be a string or None")


@dataclass(frozen=True, slots=True)
class DamageResult:
    """Damage section of a photo report."""

    status: CheckStatus
    backend: str | None
    parts: tuple[PartFinding, ...]
    summary: str | None
    detail: str | None = None


@dataclass(frozen=True, slots=True)
class DuplicateMatch:
    """Another photo in the same claim whose perceptual hash is close."""

    path: str
    distance: int


@dataclass(frozen=True, slots=True)
class DuplicateResult:
    """Duplicate section of a photo report."""

    status: CheckStatus
    flagged: bool
    algorithm: str
    image_hash: str | None
    distance_threshold: int | None
    matches: tuple[DuplicateMatch, ...] = ()
    detail: str | None = None


@dataclass(frozen=True, slots=True)
class DuplicateGroup:
    """Photos in one claim that share the same or a near-same perceptual hash."""

    paths: tuple[str, ...]
    max_distance: int


@dataclass(frozen=True, slots=True)
class GpsLocation:
    """WGS84 point read from EXIF."""

    latitude: float
    longitude: float


@dataclass(frozen=True, slots=True)
class MetadataResult:
    """EXIF section of a photo report."""

    status: CheckStatus
    flagged: bool
    timestamp: datetime | None
    gps: GpsLocation | None
    device: str | None
    software: str | None
    image_format: str | None
    flags: tuple[str, ...] = ()
    detail: str | None = None


@dataclass(frozen=True, slots=True)
class PhotoReport:
    """Checks for a single claim photo."""

    path: str
    flagged: bool
    flags: tuple[str, ...]
    error: str | None
    integrity: IntegrityResult
    damage: DamageResult
    duplicates: DuplicateResult
    metadata: MetadataResult


@dataclass(frozen=True, slots=True)
class ClaimReport:
    """Per-photo checks plus claim-level duplicate groups."""

    version: str
    claim_date: date | datetime | None
    photos: tuple[PhotoReport, ...]
    duplicate_groups: tuple[DuplicateGroup, ...]
    flagged_count: int
    error_count: int

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-ready copy of the report."""
        return {
            "version": self.version,
            "claim_date": _json_when(self.claim_date),
            "photo_count": len(self.photos),
            "flagged_count": self.flagged_count,
            "error_count": self.error_count,
            "duplicate_groups": [_group_dict(group) for group in self.duplicate_groups],
            "photos": [_photo_dict(photo) for photo in self.photos],
        }


def _json_when(value: date | datetime | None) -> str | None:
    if value is None:
        return None
    return value.isoformat()


def _group_dict(group: DuplicateGroup) -> dict[str, object]:
    return {"paths": list(group.paths), "max_distance": group.max_distance}


def _photo_dict(photo: PhotoReport) -> dict[str, object]:
    return {
        "path": photo.path,
        "flagged": photo.flagged,
        "flags": list(photo.flags),
        "error": photo.error,
        "integrity": _integrity_dict(photo.integrity),
        "damage": _damage_dict(photo.damage),
        "duplicates": _duplicate_dict(photo.duplicates),
        "metadata": _metadata_dict(photo.metadata),
    }


def _integrity_dict(result: IntegrityResult) -> dict[str, object]:
    return {
        "status": result.status,
        "flagged": result.flagged,
        "label": result.label,
        "scores": dict(result.scores),
        "threshold": result.threshold,
        "model_id": result.model_id,
        "flags": list(result.flags),
        "detail": result.detail,
    }


def _damage_dict(result: DamageResult) -> dict[str, object]:
    return {
        "status": result.status,
        "backend": result.backend,
        "parts": [
            {"part": part.part, "severity": part.severity, "confidence": part.confidence}
            for part in result.parts
        ],
        "summary": result.summary,
        "detail": result.detail,
    }


def _duplicate_dict(result: DuplicateResult) -> dict[str, object]:
    return {
        "status": result.status,
        "flagged": result.flagged,
        "algorithm": result.algorithm,
        "hash": result.image_hash,
        "distance_threshold": result.distance_threshold,
        "matches": [{"path": match.path, "distance": match.distance} for match in result.matches],
        "detail": result.detail,
    }


def _metadata_dict(result: MetadataResult) -> dict[str, object]:
    gps = (
        None
        if result.gps is None
        else {
            "latitude": result.gps.latitude,
            "longitude": result.gps.longitude,
        }
    )
    timestamp = None if result.timestamp is None else result.timestamp.isoformat()
    return {
        "status": result.status,
        "flagged": result.flagged,
        "timestamp": timestamp,
        "gps": gps,
        "device": result.device,
        "software": result.software,
        "image_format": result.image_format,
        "flags": list(result.flags),
        "detail": result.detail,
    }
