"""Errors raised by the public API."""

from __future__ import annotations


class ClaimPhotoError(Exception):
    """Base error for claim-photo checks."""


class IntegrityModelUnavailable(ClaimPhotoError):
    """The integrity classifier could not be loaded."""


class DamageModelUnavailable(ClaimPhotoError):
    """No damage-assessment backend is configured."""
