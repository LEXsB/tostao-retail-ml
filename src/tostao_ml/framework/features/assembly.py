"""Ensamblado de preprocesadores a partir del tipado de variables.

Construye un ``ColumnTransformer`` estándar (imputación + escalado para
numéricas; imputación + one-hot para categóricas) derivado del mapa de
:class:`VariableKind` del EDA, de modo que el preprocesamiento es coherente con
el análisis y no se especifica a mano por caso.
"""

from __future__ import annotations

from collections.abc import Iterable

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from tostao_ml.framework.types import VariableKind


def build_column_transformer(
    types: dict[str, VariableKind],
    *,
    exclude: Iterable[str] = (),
    scale: bool = True,
    one_hot_max_categories: int | None = 20,
) -> ColumnTransformer:
    """Crea un ``ColumnTransformer`` a partir del tipo de cada columna.

    Args:
        types: Mapa columna → :class:`VariableKind` (del EDA).
        exclude: Columnas a ignorar (p. ej. target, ids).
        scale: Si estandariza las numéricas.
        one_hot_max_categories: Límite de categorías del one-hot (agrupa el resto
            como infrecuentes); ``None`` desactiva el límite.

    Returns:
        ``ColumnTransformer`` con ramas numérica y categórica.
    """
    exclude = set(exclude)
    numeric = [c for c, k in types.items() if k.is_numeric and c not in exclude]
    categorical = [
        c
        for c, k in types.items()
        if k
        in (VariableKind.CATEGORICAL_NOMINAL, VariableKind.CATEGORICAL_ORDINAL, VariableKind.BINARY)
        and c not in exclude
    ]

    numeric_steps: list[tuple[str, object]] = [("imputer", SimpleImputer(strategy="median"))]
    if scale:
        numeric_steps.append(("scaler", StandardScaler()))

    ohe_kwargs: dict[str, object] = {
        "handle_unknown": "infrequent_if_exist",
        "sparse_output": False,
    }
    if one_hot_max_categories is not None:
        ohe_kwargs["max_categories"] = one_hot_max_categories
    categorical_steps = [
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(**ohe_kwargs)),
    ]

    return ColumnTransformer(
        transformers=[
            ("num", Pipeline(numeric_steps), numeric),
            ("cat", Pipeline(categorical_steps), categorical),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
