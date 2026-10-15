from __future__ import annotations

import claim_photo_toolkit as toolkit


def test_public_surface_is_importable() -> None:
    for name in toolkit.__all__:
        assert hasattr(toolkit, name)
    assert toolkit.MODEL_ID == "obuladinnesai/claim-photo-integrity-detector-v5"
    assert toolkit.__version__ == "0.1.0"
