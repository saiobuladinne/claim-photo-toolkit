from __future__ import annotations

import sys
import types
from typing import Any

import pytest
from PIL import Image

from claim_photo_toolkit.errors import IntegrityModelUnavailable
from claim_photo_toolkit.integrity import (
    MODEL_ID,
    classify_image,
    load_integrity_model,
    reset_integrity_model,
)
from tests.helpers import scores


def test_model_id_is_the_public_checkpoint() -> None:
    assert MODEL_ID == "obuladinnesai/claim-photo-integrity-detector-v5"


def test_flags_ai_generated_at_threshold() -> None:
    image = Image.new("RGB", (8, 8))
    flagged = classify_image(image, scores(0.5), threshold=0.5)
    assert flagged.flagged
    assert flagged.status == "flagged"
    assert flagged.label == "ai_generated"
    assert flagged.flags == ("ai_generated",)
    assert flagged.scores["ai_generated"] == 0.5
    assert flagged.model_id == MODEL_ID

    clear = classify_image(image, scores(0.49), threshold=0.5)
    assert not clear.flagged
    assert clear.status == "ok"
    assert clear.label == "real"
    assert clear.flags == ()


def test_numpy_style_score_is_accepted() -> None:
    class Scalar:
        def item(self) -> float:
            return 0.91

    def model(_image: Image.Image) -> list[dict[str, Any]]:
        return [
            {"label": " AI_Generated ", "score": Scalar()},
            {"label": "real", "score": 0.09},
        ]

    result = classify_image(Image.new("RGB", (8, 8)), model, threshold=0.5)
    assert result.flagged
    assert result.scores["ai_generated"] == pytest.approx(0.91)


def test_unexpected_output_is_an_error() -> None:
    def model(_image: Image.Image) -> str:
        return "nope"

    result = classify_image(Image.new("RGB", (8, 8)), model, threshold=0.5)
    assert result.status == "error"
    assert not result.flagged
    assert result.detail is not None
    assert "no label scores" in result.detail


def test_unknown_labels_are_an_error() -> None:
    def model(_image: Image.Image) -> list[dict[str, float | str]]:
        return [{"label": "cat", "score": 0.9}]

    result = classify_image(Image.new("RGB", (8, 8)), model, threshold=0.5)
    assert result.status == "error"
    assert result.detail is not None
    assert "ai_generated" in result.detail


def test_loader_is_lazy_cached_and_targets_the_checkpoint(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[object, ...]] = []

    def fake_pipeline(*args: object, **kwargs: object) -> object:
        calls.append((args, kwargs))
        return scores(0.1)

    module = types.ModuleType("transformers")
    module.pipeline = fake_pipeline  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "transformers", module)

    built = {"imports": 0}
    real_import = __import__

    def counting_import(name: str, *args: object, **kwargs: object) -> object:
        if name == "transformers":
            built["imports"] += 1
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", counting_import)
    reset_integrity_model()
    assert built["imports"] == 0

    first = load_integrity_model()
    second = load_integrity_model()
    assert first is second
    assert built["imports"] == 1
    assert calls == [(("image-classification",), {"model": MODEL_ID})]

    reset_integrity_model()
    load_integrity_model()
    assert len(calls) == 2


def test_missing_transformers_explains_the_extra(monkeypatch: pytest.MonkeyPatch) -> None:
    real_import = __import__

    def blocked(name: str, *args: object, **kwargs: object) -> object:
        if name == "transformers" or name.startswith("transformers."):
            raise ImportError("blocked")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", blocked)
    monkeypatch.delitem(sys.modules, "transformers", raising=False)
    reset_integrity_model()
    with pytest.raises(IntegrityModelUnavailable, match=r"claim-photo-toolkit\[ml\]"):
        load_integrity_model()
