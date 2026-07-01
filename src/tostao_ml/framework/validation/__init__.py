"""Contratos de datos reutilizables (Pandera) e integridad referencial."""

from __future__ import annotations

from .checks import check_referential_integrity, validate_dataframe
from .result import ValidationResult

__all__ = ["ValidationResult", "check_referential_integrity", "validate_dataframe"]
