"""Motor de EDA: orquesta tipado + univariado + multivariado + bivariado.

Ejecuta el mismo análisis para A/B/C parametrizado por la variable objetivo y
produce un :class:`DatasetProfile` con tablas, asociaciones y una
:class:`Narrative` autogenerada a partir de las cifras del run.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from tostao_ml.framework.narrate import Insight, NarrationThresholds, Narrative, Severity, rules
from tostao_ml.framework.types import VariableKind

from . import associations, bivariate, univariate
from .typing import TypingConfig, group_by_kind, infer_variable_types


@dataclass(slots=True)
class DatasetProfile:
    """Resultado estructurado del perfilado de un dataset.

    Attributes:
        name: Nombre del dataset.
        n_rows / n_cols: Dimensiones.
        types: Tipo inferido por columna.
        univariate: Tabla de estadística univariada (una fila por variable).
        correlations: Matrices de correlación por método (numéricas).
        vif: VIF por variable numérica.
        condition_number: Número de condición de la matriz de diseño.
        mutual_information: MI feature→target (si hay target).
        bivariate: Tabla de tests feature–target (si hay target).
        target: Nombre de la variable objetivo (o ``None``).
        narrative: Conclusiones autogeneradas del perfilado.
    """

    name: str
    n_rows: int
    n_cols: int
    types: dict[str, VariableKind]
    univariate: pd.DataFrame
    correlations: dict[str, pd.DataFrame]
    vif: pd.Series
    condition_number: float
    mutual_information: pd.Series
    bivariate: pd.DataFrame
    target: str | None = None
    narrative: Narrative = field(default_factory=Narrative)

    def redundant_pairs(self, threshold: float = 0.9) -> list[tuple[str, str, float]]:
        """Pares de variables con |Pearson| ≥ ``threshold`` (candidatas a drop)."""
        pearson = self.correlations.get("pearson")
        if pearson is None or pearson.empty:
            return []
        pairs = []
        cols = list(pearson.columns)
        for i in range(len(cols)):
            for j in range(i + 1, len(cols)):
                corr = pearson.iloc[i, j]
                if pd.notna(corr) and abs(corr) >= threshold:
                    pairs.append((cols[i], cols[j], float(corr)))
        return sorted(pairs, key=lambda t: -abs(t[2]))


def profile_dataset(
    frame: pd.DataFrame,
    *,
    name: str = "dataset",
    target: str | None = None,
    type_overrides: dict[str, str | VariableKind] | None = None,
    thresholds: NarrationThresholds | None = None,
    typing_config: TypingConfig | None = None,
    random_state: int = 42,
) -> DatasetProfile:
    """Perfila un DataFrame de punta a punta y redacta sus conclusiones.

    Args:
        frame: Datos a perfilar.
        name: Nombre legible del dataset.
        target: Variable objetivo para el análisis bivariado y MI (opcional).
        type_overrides: Override de tipos por columna.
        thresholds: Umbrales de narración.
        typing_config: Umbrales del tipado automático.
        random_state: Semilla para estimadores estocásticos (MI).

    Returns:
        :class:`DatasetProfile` completo con narrativa.
    """
    thresholds = thresholds or NarrationThresholds()
    types = infer_variable_types(frame, overrides=type_overrides, config=typing_config)
    grouped = group_by_kind(types)
    numeric_cols = [c for c, k in types.items() if k.is_numeric]
    numeric_df = (
        frame[numeric_cols].apply(pd.to_numeric, errors="coerce")
        if numeric_cols
        else pd.DataFrame()
    )

    narrative = Narrative()
    univariate_tbl = _profile_univariate(frame, types, narrative, thresholds)
    correlations, vif, cond = _profile_multivariate(numeric_df, narrative, thresholds)
    mi, bivariate_tbl = _profile_target(frame, types, target, narrative, thresholds, random_state)

    profile = DatasetProfile(
        name=name,
        n_rows=int(frame.shape[0]),
        n_cols=int(frame.shape[1]),
        types=types,
        univariate=univariate_tbl,
        correlations=correlations,
        vif=vif,
        condition_number=cond,
        mutual_information=mi,
        bivariate=bivariate_tbl,
        target=target,
        narrative=narrative,
    )
    _narrate_overview(profile, grouped, narrative)
    return profile


def _profile_univariate(
    frame: pd.DataFrame,
    types: dict[str, VariableKind],
    narrative: Narrative,
    thresholds: NarrationThresholds,
) -> pd.DataFrame:
    """Estadística univariada por columna + insights de outliers/normalidad."""
    rows: list[dict[str, object]] = []
    for col, kind in types.items():
        if kind.is_numeric:
            summary = univariate.numeric_summary(frame[col])
            pct_out = univariate.outlier_fraction_iqr(frame[col])
            norm = univariate.normality_test(frame[col])
            rows.append(
                {
                    "variable": col,
                    "kind": kind.value,
                    **summary,
                    "pct_outliers_iqr": pct_out,
                    "normality_test": norm["test"],
                    "normality_pvalue": norm["pvalue"],
                }
            )
            if pct_out >= 5.0:
                narrative.add(rules.outlier_insight(col, pct_out))
        elif kind.is_categorical or kind is VariableKind.TEXT:
            cat_summary = univariate.categorical_summary(frame[col])
            rows.append({"variable": col, "kind": kind.value, **cat_summary})
    return pd.DataFrame(rows)


def _profile_multivariate(
    numeric_df: pd.DataFrame, narrative: Narrative, thresholds: NarrationThresholds
) -> tuple[dict[str, pd.DataFrame], pd.Series, float]:
    """Correlaciones, VIF y número de condición + insights de redundancia."""
    if numeric_df.shape[1] < 2:
        return {}, pd.Series(dtype=float, name="vif"), float("nan")
    correlations = associations.correlation_matrices(numeric_df)
    vif = associations.vif_scores(numeric_df)
    cond = associations.condition_number(numeric_df)

    for feature, value in vif.items():
        insight = rules.vif_insight(str(feature), float(value), thresholds)
        if insight is not None:
            narrative.add(insight)

    pearson = correlations["pearson"]
    cols = list(pearson.columns)
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            corr = pearson.iloc[i, j]
            if pd.notna(corr) and abs(corr) >= thresholds.corr_strong:
                narrative.add(
                    rules.correlation_insight(cols[i], cols[j], float(corr), "Pearson", thresholds)
                )
    return correlations, vif, cond


def _profile_target(
    frame: pd.DataFrame,
    types: dict[str, VariableKind],
    target: str | None,
    narrative: Narrative,
    thresholds: NarrationThresholds,
    random_state: int,
) -> tuple[pd.Series, pd.DataFrame]:
    """Análisis bivariado vs target + información mutua."""
    empty_mi = pd.Series(dtype=float, name="mutual_information")
    if target is None or target not in frame.columns:
        return empty_mi, pd.DataFrame()

    target_kind = types[target]
    discrete_target = target_kind.is_categorical
    rows: list[dict[str, object]] = []

    for col, kind in types.items():
        if col == target or not kind.is_modelable:
            continue
        result: dict[str, float | str] | None = None
        if kind.is_numeric and discrete_target:
            result = bivariate.test_numeric_vs_categorical(frame[col], frame[target])
        elif kind.is_categorical and discrete_target:
            result = bivariate.test_categorical_vs_categorical(frame[col], frame[target])
        elif kind.is_numeric and not discrete_target:
            corr = (
                frame[[col, target]]
                .apply(pd.to_numeric, errors="coerce")
                .corr(method="spearman")
                .iloc[0, 1]
            )
            result = {
                "test": "Spearman",
                "pvalue": float("nan"),
                "effect_name": "ρ",
                "effect_size": float(corr),
            }
        elif kind.is_categorical and not discrete_target:
            eta = associations.correlation_ratio(frame[col], frame[target])
            result = {
                "test": "η (razón de correlación)",
                "pvalue": float("nan"),
                "effect_name": "η",
                "effect_size": eta,
            }
        if result is None:
            continue
        rows.append({"feature": col, "feature_kind": kind.value, **result})
        _narrate_bivariate(col, target, kind, discrete_target, result, narrative, thresholds)

    mi = _target_mutual_information(
        frame, types, target, discrete_target, narrative, thresholds, random_state
    )
    return mi, pd.DataFrame(rows)


def _narrate_bivariate(
    feature: str,
    target: str,
    kind: VariableKind,
    discrete_target: bool,
    result: dict[str, float | str],
    narrative: Narrative,
    thresholds: NarrationThresholds,
) -> None:
    """Traduce un test bivariado a un insight."""
    effect = float(result.get("effect_size", float("nan")))
    pvalue = result.get("pvalue", float("nan"))
    if kind.is_numeric and discrete_target:
        narrative.add(
            rules.group_difference_insight(
                feature,
                target,
                float(pvalue),
                effect,
                str(result["test"]),
                str(result["effect_name"]),
                thresholds,
            )
        )
    elif kind.is_categorical and discrete_target:
        narrative.add(rules.association_insight(feature, target, effect, float(pvalue), thresholds))
    elif (
        kind.is_numeric
        and not discrete_target
        and pd.notna(effect)
        and abs(effect) >= thresholds.corr_moderate
    ):
        narrative.add(rules.correlation_insight(feature, target, effect, "Spearman", thresholds))


def _target_mutual_information(
    frame: pd.DataFrame,
    types: dict[str, VariableKind],
    target: str,
    discrete_target: bool,
    narrative: Narrative,
    thresholds: NarrationThresholds,
    random_state: int,
) -> pd.Series:
    """Codifica features y calcula MI feature→target, narrando las relevantes."""
    feature_cols = [c for c, k in types.items() if c != target and k.is_modelable]
    if not feature_cols:
        return pd.Series(dtype=float, name="mutual_information")
    encoded = pd.DataFrame(index=frame.index)
    for col in feature_cols:
        if types[col].is_numeric:
            encoded[col] = pd.to_numeric(frame[col], errors="coerce")
        else:
            encoded[col] = frame[col].astype("category").cat.codes.replace(-1, pd.NA)
    y = frame[target]
    if not discrete_target:
        y = pd.to_numeric(y, errors="coerce")
    mi = associations.mutual_information(
        encoded, y, discrete_target=discrete_target, random_state=random_state
    )
    for feature, value in mi.head(10).items():
        insight = rules.mutual_information_insight(str(feature), target, float(value), thresholds)
        if insight is not None:
            narrative.add(insight)
    return mi


def _narrate_overview(
    profile: DatasetProfile, grouped: dict[VariableKind, list[str]], narrative: Narrative
) -> None:
    """Insight de resumen con la composición de tipos y la dimensión."""
    composition = ", ".join(f"{len(cols)} {kind.value}" for kind, cols in grouped.items())
    narrative.add(
        Insight(
            text=f"«{profile.name}»: {profile.n_rows:,} filas × {profile.n_cols} columnas ({composition}).",
            severity=Severity.INFO,
            metrics={"n_rows": profile.n_rows, "n_cols": profile.n_cols},
            tags=("eda", "overview"),
            title="Resumen",
        )
    )
