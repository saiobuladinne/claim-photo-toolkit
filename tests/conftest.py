from __future__ import annotations

from collections.abc import Iterator

import pytest

from claim_photo_toolkit.integrity import reset_integrity_model


@pytest.fixture(autouse=True)
def _reset_integrity_cache() -> Iterator[None]:
    reset_integrity_model()
    yield
    reset_integrity_model()
