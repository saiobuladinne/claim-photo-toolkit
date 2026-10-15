"""Pluggable damage assessment. No damage model ships with the toolkit."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable

from PIL import Image

from claim_photo_toolkit.errors import DamageModelUnavailable
from claim_photo_toolkit.models import DamageAssessment, DamageResult, PartFinding

__all__ = [
    "DamageAssessment",
    "DamageBackend",
    "PartFinding",
    "UnavailableDamageBackend",
]


@runtime_checkable
class DamageBackend(Protocol):
    """Scores damaged parts on one car photo.

    A backend names itself and maps one RGB image to part-level findings.
    Part names belong to the backend; this toolkit does not define a part list.
    """

    name: str

    def assess(self, image: Image.Image, path: Path) -> DamageAssessment:
        """Return findings for one photo."""


class UnavailableDamageBackend:
    """Placeholder used until a damage model is passed to ``inspect``."""

    name = "unavailable"

    def assess(self, image: Image.Image, path: Path) -> DamageAssessment:
        """Refuse to invent damage findings."""
        del image, path
        raise DamageModelUnavailable(
            "No damage-assessment backend is configured. Pass damage_backend= to inspect()."
        )


def unavailable_damage(detail: str) -> DamageResult:
    return DamageResult(
        status="unavailable",
        backend="unavailable",
        parts=(),
        summary=None,
        detail=detail,
    )


def error_damage(detail: str, *, backend: str | None) -> DamageResult:
    return DamageResult(
        status="error",
        backend=backend,
        parts=(),
        summary=None,
        detail=detail,
    )


def result_from_assessment(backend_name: str, assessment: DamageAssessment) -> DamageResult:
    return DamageResult(
        status="ok",
        backend=backend_name,
        parts=assessment.parts,
        summary=assessment.summary,
        detail=None,
    )
