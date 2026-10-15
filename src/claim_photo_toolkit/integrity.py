"""Lazy loader for the public claim-photo integrity classifier."""

from __future__ import annotations

import threading
from typing import Protocol, cast

from PIL import Image

from claim_photo_toolkit.errors import ClaimPhotoError, IntegrityModelUnavailable
from claim_photo_toolkit.models import CheckStatus, IntegrityResult

MODEL_ID = "obuladinnesai/claim-photo-integrity-detector-v5"
AI_GENERATED = "ai_generated"
REAL = "real"

_lock = threading.Lock()
_model: IntegrityModel | None = None


class IntegrityModel(Protocol):
    """Callable that scores one image the way ``transformers.pipeline`` does."""

    def __call__(self, image: Image.Image) -> object:
        """Return label/score pairs for an RGB image."""


class IntegrityOutputError(ClaimPhotoError):
    """The classifier returned a result this toolkit cannot read."""


def load_integrity_model() -> IntegrityModel:
    """Load the Hugging Face classifier on first use and reuse it."""
    global _model
    if _model is not None:
        return _model
    with _lock:
        if _model is None:
            _model = _build_pipeline()
        return _model


def reset_integrity_model() -> None:
    """Drop the cached classifier so the next check loads it again."""
    global _model
    with _lock:
        _model = None


def classify_image(
    image: Image.Image,
    model: IntegrityModel,
    *,
    threshold: float,
) -> IntegrityResult:
    """Score one RGB image and flag it when ``ai_generated`` clears the threshold."""
    try:
        scores = _parse_scores(model(image))
    except IntegrityOutputError as exc:
        return error_integrity(str(exc), threshold=threshold)
    label = min(scores, key=lambda name: (-scores[name], name))
    flagged = scores.get(AI_GENERATED, 0.0) >= threshold
    return _result(
        status="flagged" if flagged else "ok",
        flagged=flagged,
        label=label,
        scores=scores,
        threshold=threshold,
        flags=(AI_GENERATED,) if flagged else (),
    )


def skipped_integrity(detail: str) -> IntegrityResult:
    return _result(status="skipped", detail=detail)


def unavailable_integrity(detail: str) -> IntegrityResult:
    return _result(status="unavailable", detail=detail)


def error_integrity(detail: str, *, threshold: float | None = None) -> IntegrityResult:
    return _result(status="error", threshold=threshold, detail=detail)


def _build_pipeline() -> IntegrityModel:
    try:
        from transformers import pipeline
    except ImportError as exc:
        raise IntegrityModelUnavailable(
            "Integrity checks need the ml extra: pip install 'claim-photo-toolkit[ml]'"
        ) from exc
    loaded = cast(IntegrityModel, pipeline("image-classification", model=MODEL_ID))
    if not callable(loaded):
        raise IntegrityModelUnavailable("transformers pipeline was not callable")
    return loaded


def _parse_scores(raw: object) -> dict[str, float]:
    if not isinstance(raw, list) or not raw:
        raise IntegrityOutputError("integrity model returned no label scores")
    scores: dict[str, float] = {}
    for item in raw:
        if not isinstance(item, dict):
            raise IntegrityOutputError("integrity model returned no label scores")
        label = item.get("label")
        score = item.get("score")
        if not isinstance(label, str) or not label.strip():
            raise IntegrityOutputError("integrity model returned a score without a label")
        scores[label.strip().casefold()] = _as_score(score)
    if AI_GENERATED not in scores and REAL not in scores:
        raise IntegrityOutputError("integrity model did not return real or ai_generated")
    return scores


def _as_score(value: object) -> float:
    if isinstance(value, bool):
        raise IntegrityOutputError("integrity model returned a non-numeric score")
    if isinstance(value, (int, float)):
        return float(value)
    item = getattr(value, "item", None)
    if callable(item):
        number = item()
        if isinstance(number, (int, float)) and not isinstance(number, bool):
            return float(number)
    raise IntegrityOutputError("integrity model returned a non-numeric score")


def _result(
    *,
    status: CheckStatus,
    flagged: bool = False,
    label: str | None = None,
    scores: dict[str, float] | None = None,
    threshold: float | None = None,
    flags: tuple[str, ...] = (),
    detail: str | None = None,
) -> IntegrityResult:
    return IntegrityResult(
        status=status,
        flagged=flagged,
        label=label,
        scores={} if scores is None else scores,
        threshold=threshold,
        model_id=MODEL_ID,
        flags=flags,
        detail=detail,
    )
