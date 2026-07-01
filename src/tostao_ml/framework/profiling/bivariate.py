"""Relación bivariada con la variable objetivo (§4.3).

Selecciona el test según los tipos: numérica vs target categórico (t/ANOVA +
Mann-Whitney/Kruskal + tamaño de efecto), categórica vs categórica (chi² +
Cramér's V) y comparación de distribuciones (KS). Cada función devuelve métricas
crudas; la narración las interpreta.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from .associations import cramers_v


def cohens_d(group_a: np.ndarray, group_b: np.ndarray) -> float:
    """Tamaño de efecto d de Cohen para dos grupos (pooled std)."""
    na, nb = group_a.size, group_b.size
    if na < 2 or nb < 2:
        return 0.0
    sa, sb = np.var(group_a, ddof=1), np.var(group_b, ddof=1)
    pooled = np.sqrt(((na - 1) * sa + (nb - 1) * sb) / (na + nb - 2))
    if pooled == 0:
        return 0.0
    return float((np.mean(group_a) - np.mean(group_b)) / pooled)


def eta_squared(groups: list[np.ndarray]) -> float:
    """η² (tamaño de efecto de un ANOVA) a partir de los grupos."""
    all_values = np.concatenate(groups)
    grand_mean = all_values.mean()
    ss_total = float(((all_values - grand_mean) ** 2).sum())
    if ss_total == 0:
        return 0.0
    ss_between = float(sum(g.size * (g.mean() - grand_mean) ** 2 for g in groups))
    return ss_between / ss_total


def test_numeric_vs_categorical(values: pd.Series, groups: pd.Series) -> dict[str, float | str]:
    """Compara una numérica entre los niveles de una categórica.

    Elige t-test/ANOVA (paramétrico), Mann-Whitney/Kruskal (no paramétrico) y el
    tamaño de efecto adecuado (Cohen's d con 2 grupos, η² con más).
    """
    df = pd.DataFrame({"v": values, "g": groups}).dropna()
    levels = [g["v"].to_numpy(dtype=float) for _, g in df.groupby("g", observed=True)]
    levels = [arr for arr in levels if arr.size >= 2]
    if len(levels) < 2:
        return {
            "test": "n/a",
            "pvalue": float("nan"),
            "effect_name": "n/a",
            "effect_size": float("nan"),
        }

    if len(levels) == 2:
        a, b = levels
        _, p_param = stats.ttest_ind(a, b, equal_var=False)
        _, p_nonparam = stats.mannwhitneyu(a, b, alternative="two-sided")
        return {
            "test": "t-test (Welch)",
            "pvalue": float(p_param),
            "test_nonparam": "Mann-Whitney U",
            "pvalue_nonparam": float(p_nonparam),
            "effect_name": "Cohen's d",
            "effect_size": cohens_d(a, b),
            "n_groups": 2,
        }

    _, p_param = stats.f_oneway(*levels)
    _, p_nonparam = stats.kruskal(*levels)
    return {
        "test": "ANOVA",
        "pvalue": float(p_param),
        "test_nonparam": "Kruskal-Wallis",
        "pvalue_nonparam": float(p_nonparam),
        "effect_name": "eta²",
        "effect_size": eta_squared(levels),
        "n_groups": len(levels),
    }


def test_categorical_vs_categorical(x: pd.Series, y: pd.Series) -> dict[str, float | str]:
    """Chi-cuadrado + Cramér's V entre dos categóricas."""
    v, pvalue = cramers_v(x, y)
    return {"test": "chi²", "pvalue": pvalue, "effect_name": "Cramér's V", "effect_size": v}


def ks_two_sample(group_a: pd.Series, group_b: pd.Series) -> dict[str, float]:
    """Test KS de dos muestras (diferencia global de distribuciones)."""
    a = group_a.dropna().to_numpy(dtype=float)
    b = group_b.dropna().to_numpy(dtype=float)
    if a.size < 2 or b.size < 2:
        return {"statistic": float("nan"), "pvalue": float("nan")}
    stat, pvalue = stats.ks_2samp(a, b)
    return {"statistic": float(stat), "pvalue": float(pvalue)}
