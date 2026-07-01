"""Transformadores de features reutilizables (estilo scikit-learn).

Componibles en ``Pipeline``/``ColumnTransformer`` y compartidos por A/B/C:
temporales, cíclicos y rezagos (A/C), RFM (C), codificadores (todos),
winsorización (C) y selección por VIF/correlación (todos).
"""

from __future__ import annotations

from .assembly import build_column_transformer
from .base import PandasTransformer
from .encoders import FrequencyEncoder
from .outliers import Winsorizer
from .rfm import RFMTransformer
from .selection import CorrelationSelector, VIFSelector
from .temporal import CyclicalEncoder, DateTimeFeatures, GroupLagFeatures

__all__ = [
    "CorrelationSelector",
    "CyclicalEncoder",
    "DateTimeFeatures",
    "FrequencyEncoder",
    "GroupLagFeatures",
    "PandasTransformer",
    "RFMTransformer",
    "VIFSelector",
    "Winsorizer",
    "build_column_transformer",
]
