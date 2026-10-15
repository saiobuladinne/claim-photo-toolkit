"""Checks for car-damage photos submitted with insurance claims."""

from claim_photo_toolkit.api import inspect
from claim_photo_toolkit.damage import (
    DamageAssessment,
    DamageBackend,
    PartFinding,
    UnavailableDamageBackend,
)
from claim_photo_toolkit.errors import (
    ClaimPhotoError,
    DamageModelUnavailable,
    IntegrityModelUnavailable,
)
from claim_photo_toolkit.integrity import MODEL_ID, load_integrity_model, reset_integrity_model
from claim_photo_toolkit.models import (
    ClaimReport,
    DamageResult,
    DuplicateGroup,
    DuplicateMatch,
    DuplicateResult,
    GpsLocation,
    IntegrityResult,
    MetadataResult,
    PhotoReport,
    Severity,
)

__version__ = "0.1.0"

__all__ = [
    "MODEL_ID",
    "ClaimPhotoError",
    "ClaimReport",
    "DamageAssessment",
    "DamageBackend",
    "DamageModelUnavailable",
    "DamageResult",
    "DuplicateGroup",
    "DuplicateMatch",
    "DuplicateResult",
    "GpsLocation",
    "IntegrityModelUnavailable",
    "IntegrityResult",
    "MetadataResult",
    "PartFinding",
    "PhotoReport",
    "Severity",
    "UnavailableDamageBackend",
    "__version__",
    "inspect",
    "load_integrity_model",
    "reset_integrity_model",
]
