"""Asociaciones, correlaciones y redundancia (§4.4–4.5).

Cubre correlaciones lineales/monótonas, asociación no lineal (información mutua),
asociación categórica (Cramér's V, Theil's U), numérica–categórica (razón de
correlación η), multicolinealidad (VIF, número de condición) y estructura latente
compartida entre bloques (correlaciones canónicas).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.cross_decomposition import CCA
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

_CORR_METHODS = ("pearson", "spearman", "kendall")


def correlation_matrices(
    numeric: pd.DataFrame, methods: tuple[str, ...] = _CORR_METHODS
) -> dict[str, pd.DataFrame]:
    """Matrices de correlación por método (lineal, monótona, tau)."""
    return {m: numeric.corr(method=m) for m in methods}


def vif_scores(numeric: pd.DataFrame) -> pd.Series:
    """Factor de inflación de varianza (VIF) por variable.

    VIF_i = 1 / (1 - R²_i), con R²_i de regredir la variable i sobre el resto.
    Valores altos ⇒ multicolinealidad. Devuelve ``inf`` si R²≈1.
    """
    cols = list(numeric.columns)
    clean = numeric.dropna()
    vifs: dict[str, float] = {}
    if clean.shape[0] <= len(cols) or len(cols) < 2:
        return pd.Series(dict.fromkeys(cols, np.nan), name="vif")
    for col in cols:
        y = clean[col].to_numpy(dtype=float)
        x = clean.drop(columns=[col]).to_numpy(dtype=float)
        r2 = LinearRegression().fit(x, y).score(x, y)
        vifs[col] = float("inf") if r2 >= 1.0 else float(1.0 / (1.0 - r2))
    return pd.Series(vifs, name="vif").sort_values(ascending=False)


def condition_number(numeric: pd.DataFrame) -> float:
    """Número de condición de la matriz de diseño estandarizada.

    >30 sugiere multicolinealidad; >100, severa.
    """
    clean = numeric.dropna()
    if clean.shape[0] < 2 or clean.shape[1] < 2:
        return float("nan")
    scaled = StandardScaler().fit_transform(clean)
    return float(np.linalg.cond(scaled))


def mutual_information(
    features: pd.DataFrame, target: pd.Series, *, discrete_target: bool, random_state: int = 42
) -> pd.Series:
    """Información mutua feature→target (capta relaciones no lineales).

    Args:
        features: Matriz de features numéricas (categóricas ya codificadas).
        target: Variable objetivo.
        discrete_target: ``True`` si el target es categórico/clase.
        random_state: Semilla del estimador de MI.

    Returns:
        Serie de MI por feature, ordenada de mayor a menor.
    """
    clean = features.dropna()
    y = target.loc[clean.index]
    mask = y.notna()
    x_arr = clean.loc[mask].to_numpy(dtype=float)
    y_arr = y.loc[mask].to_numpy()
    if x_arr.shape[0] < 3 or x_arr.shape[1] == 0:
        return pd.Series(dtype=float, name="mutual_information")
    estimator = mutual_info_classif if discrete_target else mutual_info_regression
    scores = estimator(x_arr, y_arr, random_state=random_state)
    return pd.Series(scores, index=clean.columns, name="mutual_information").sort_values(
        ascending=False
    )


def cramers_v(x: pd.Series, y: pd.Series) -> tuple[float, float]:
    """Cramér's V con corrección de sesgo (Bergsma) + p-valor del chi².

    Returns:
        ``(cramers_v, pvalue)`` para dos variables categóricas.
    """
    table = pd.crosstab(x, y)
    if table.shape[0] < 2 or table.shape[1] < 2:
        return 0.0, 1.0
    chi2, pvalue, _, _ = stats.chi2_contingency(table)
    n = table.to_numpy().sum()
    phi2 = chi2 / n
    r, k = table.shape
    phi2corr = max(0.0, phi2 - (k - 1) * (r - 1) / (n - 1))
    rcorr = r - (r - 1) ** 2 / (n - 1)
    kcorr = k - (k - 1) ** 2 / (n - 1)
    denom = min(kcorr - 1, rcorr - 1)
    v = float(np.sqrt(phi2corr / denom)) if denom > 0 else 0.0
    return v, float(pvalue)


def theils_u(x: pd.Series, y: pd.Series) -> float:
    """Coeficiente de incertidumbre U(x|y) ∈ [0,1] (asimétrico).

    Fracción de la entropía de ``x`` explicada por conocer ``y``.
    """
    df = pd.DataFrame({"x": x, "y": y}).dropna()
    if df.empty:
        return 0.0
    px = df["x"].value_counts(normalize=True)
    hx = -np.sum(px * np.log(px))
    if hx == 0:
        return 1.0
    # Entropía condicional H(x|y).
    hxy = 0.0
    py = df["y"].value_counts(normalize=True)
    for level, p_level in py.items():
        sub = df.loc[df["y"] == level, "x"].value_counts(normalize=True)
        hxy += p_level * (-np.sum(sub * np.log(sub)))
    return float((hx - hxy) / hx)


def correlation_ratio(categories: pd.Series, values: pd.Series) -> float:
    """Razón de correlación η (numérica–categórica) ∈ [0,1].

    η² = varianza entre-grupos / varianza total. η=0 sin relación, η=1 relación
    determinista.
    """
    df = pd.DataFrame({"cat": categories, "val": values}).dropna()
    if df.empty or df["cat"].nunique() < 2:
        return 0.0
    grand_mean = df["val"].mean()
    ss_total = float(((df["val"] - grand_mean) ** 2).sum())
    if ss_total == 0:
        return 0.0
    ss_between = 0.0
    for _, group in df.groupby("cat", observed=True):
        ss_between += len(group) * (group["val"].mean() - grand_mean) ** 2
    return float(np.sqrt(ss_between / ss_total))


def canonical_correlations(
    block_x: pd.DataFrame, block_y: pd.DataFrame, n_components: int = 2, random_state: int = 42
) -> list[float]:
    """Correlaciones canónicas entre dos bloques de variables numéricas.

    Detecta estructura latente compartida (p. ej. bloque producto/tienda vs
    comportamiento). Devuelve la correlación de cada par de variates canónicas.
    """
    joined = pd.concat([block_x, block_y], axis=1).dropna()
    if joined.shape[0] < 3:
        return []
    x = joined[block_x.columns].to_numpy(dtype=float)
    y = joined[block_y.columns].to_numpy(dtype=float)
    k = min(n_components, x.shape[1], y.shape[1])
    if k < 1:
        return []
    cca = CCA(n_components=k)
    try:
        x_c, y_c = cca.fit_transform(x, y)
    except (ValueError, np.linalg.LinAlgError):  # pragma: no cover - datos degenerados
        return []
    corrs = []
    for i in range(k):
        xc, yc = x_c[:, i], y_c[:, i]
        if np.std(xc) == 0 or np.std(yc) == 0:
            corrs.append(0.0)
        else:
            corrs.append(float(abs(np.corrcoef(xc, yc)[0, 1])))
    return corrs
