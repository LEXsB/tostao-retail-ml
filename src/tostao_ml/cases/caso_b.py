"""Caso B — Combos: clustering de tiendas + reglas de asociación (FP-Growth).

Segmenta tiendas por su perfil de compra (reutilizando el K-Means del framework),
descubre reglas de co-compra por cluster (support/confidence/lift/conviction) y
propone los Top-N combos por cluster con precio sugerido y lift esperado.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd
from mlxtend.frequent_patterns import association_rules, fpgrowth

from tostao_ml.framework.models import KMeansModel
from tostao_ml.framework.narrate import Insight, Narrative, Severity


@dataclass(slots=True)
class CaseBResult:
    """Resultado del Caso B: perfiles, clusters, reglas y combos propuestos."""

    store_profiles: pd.DataFrame
    store_clusters: pd.Series
    rules: pd.DataFrame
    combos: pd.DataFrame
    silhouette: float
    narrative: Narrative = field(default_factory=Narrative)
    k_selection: pd.DataFrame | None = None


def build_store_profiles(master_b: pd.DataFrame) -> pd.DataFrame:
    """Perfil de cada tienda: mezcla de categorías + estadísticos de cesta."""
    cat_qty = master_b.pivot_table(
        index="id_tienda", columns="categoria", values="cantidad", aggfunc="sum", fill_value=0.0
    )
    cat_share = cat_qty.div(cat_qty.sum(axis=1).replace(0, 1), axis=0)
    cat_share.columns = [f"share_{c}" for c in cat_share.columns]
    basket_stats = master_b.groupby("id_tienda", observed=True).agg(
        n_tickets=("id_ticket", "nunique"),
        items_por_ticket=("cantidad", "sum"),
        ticket_medio=("importe_linea", "sum"),
    )
    basket_stats["items_por_ticket"] /= basket_stats["n_tickets"]
    basket_stats["ticket_medio"] /= basket_stats["n_tickets"]
    return cat_share.join(basket_stats).fillna(0.0)


def cluster_stores(
    profiles: pd.DataFrame, n_clusters: int = 3, seed: int = 42
) -> tuple[pd.Series, float]:
    """Agrupa tiendas por perfil con K-Means; devuelve etiquetas y silhouette."""
    from sklearn.preprocessing import StandardScaler

    x = pd.DataFrame(
        StandardScaler().fit_transform(profiles), index=profiles.index, columns=profiles.columns
    )
    k = min(n_clusters, max(2, len(profiles) - 1))
    model = KMeansModel(n_clusters=k, random_state=seed).fit(x)
    labels = pd.Series(model.predict(x), index=profiles.index, name="cluster")
    return labels, float(model.metadata.extra.get("silhouette", 0.0))


def select_n_clusters(
    profiles: pd.DataFrame, k_range: tuple[int, ...] = (2, 3, 4, 5, 6), seed: int = 42
) -> tuple[int, pd.DataFrame]:
    """Elige k por máxima silhouette barriendo un rango; devuelve (k*, tabla de barrido)."""
    rows = []
    for k in k_range:
        if k >= len(profiles):
            continue
        _, sil = cluster_stores(profiles, n_clusters=k, seed=seed)
        rows.append({"k": k, "silhouette": round(sil, 4)})
    sweep = pd.DataFrame(rows)
    best_k = int(sweep.loc[sweep["silhouette"].idxmax(), "k"]) if not sweep.empty else 3
    return best_k, sweep


def association_rules_for(
    baskets: pd.DataFrame, min_support: float = 0.02, min_lift: float = 1.0
) -> pd.DataFrame:
    """Reglas de asociación (FP-Growth) sobre una matriz cesta×producto booleana."""
    if baskets.shape[0] < 5 or baskets.shape[1] < 2:
        return pd.DataFrame()
    itemsets = fpgrowth(baskets, min_support=min_support, use_colnames=True)
    if itemsets.empty:
        return pd.DataFrame()
    rules = association_rules(itemsets, metric="lift", min_threshold=min_lift)
    rules = rules[(rules["antecedents"].map(len) == 1) & (rules["consequents"].map(len) == 1)]
    return rules.sort_values("lift", ascending=False).reset_index(drop=True)


def top_combos_per_cluster(
    master_b: pd.DataFrame,
    baskets: pd.DataFrame,
    store_clusters: pd.Series,
    *,
    top_n: int = 5,
    combo_discount: float = 0.1,
    min_support: float = 0.02,
) -> pd.DataFrame:
    """Top-N combos (pares) por cluster con precio propuesto y lift esperado."""
    price = master_b.groupby("id_producto", observed=True)["precio_unitario"].mean()
    name = master_b.groupby("id_producto", observed=True)["nombre"].first()
    ticket_cluster = (
        master_b[["id_ticket", "id_tienda"]]
        .drop_duplicates()
        .assign(cluster=lambda d: d["id_tienda"].map(store_clusters))
    )
    records = []
    for cluster_id, group in ticket_cluster.groupby("cluster", observed=True):
        sub = baskets.loc[baskets.index.isin(set(group["id_ticket"]))]
        rules = association_rules_for(sub, min_support=min_support)
        seen: set[frozenset] = set()
        for _, r in rules.iterrows():
            a = next(iter(r["antecedents"]))
            b = next(iter(r["consequents"]))
            pair = frozenset((a, b))
            if pair in seen:
                continue
            seen.add(pair)
            base_price = float(price.get(a, 0) + price.get(b, 0))
            records.append(
                {
                    "cluster": int(cluster_id),
                    "producto_a": name.get(a, a),
                    "producto_b": name.get(b, b),
                    "support": round(float(r["support"]), 4),
                    "confidence": round(float(r["confidence"]), 4),
                    "lift": round(float(r["lift"]), 3),
                    "conviction": round(float(r.get("conviction", float("nan"))), 3),
                    "precio_combo": round(base_price * (1 - combo_discount), 2),
                    "precio_lista": round(base_price, 2),
                }
            )
            if len([x for x in records if x["cluster"] == cluster_id]) >= top_n:
                break
    return pd.DataFrame(records)


def run_case_b(
    master_b: pd.DataFrame,
    baskets: pd.DataFrame,
    *,
    n_clusters: int = 3,
    top_n: int = 5,
    tune: bool = False,
) -> CaseBResult:
    """Ejecuta el Caso B completo: perfiles → clusters → reglas → combos.

    Si ``tune`` es ``True``, selecciona el número de clusters ``k`` maximizando la
    silhouette sobre un barrido.
    """
    profiles = build_store_profiles(master_b)
    k_selection = None
    if tune:
        n_clusters, k_selection = select_n_clusters(profiles)
    clusters, silhouette = cluster_stores(profiles, n_clusters=n_clusters)
    rules = association_rules_for(baskets)
    combos = top_combos_per_cluster(master_b, baskets, clusters, top_n=top_n)

    narrative = Narrative()
    narrative.add(
        Insight(
            text=(
                f"{len(profiles)} tiendas agrupadas en {clusters.nunique()} clusters por perfil de compra "
                f"(silhouette = {silhouette:.2f})."
            ),
            severity=Severity.GOOD if silhouette > 0.25 else Severity.INFO,
            metrics={
                "n_stores": len(profiles),
                "n_clusters": int(clusters.nunique()),
                "silhouette": round(silhouette, 3),
            },
            tags=("caso_b", "clustering"),
            title="Segmentación",
        )
    )
    if not rules.empty:
        best = rules.iloc[0]
        a = next(iter(best["antecedents"]))
        b = next(iter(best["consequents"]))
        narrative.add(
            Insight(
                text=(
                    f"Regla de co-compra más fuerte: «{a}» → «{b}» con lift {best['lift']:.2f} "
                    f"(confianza {best['confidence']:.0%}, support {best['support']:.1%})."
                ),
                severity=Severity.GOOD,
                metrics={"lift": round(float(best["lift"]), 3)},
                tags=("caso_b", "rules"),
                title="Afinidad",
            )
        )
    narrative.add(
        Insight(
            text=f"Se proponen {len(combos)} combos (Top {top_n} por cluster) con precio y lift esperado.",
            severity=Severity.INFO,
            metrics={"n_combos": len(combos)},
            tags=("caso_b", "combos"),
        )
    )
    return CaseBResult(
        profiles, clusters, rules, combos, silhouette, narrative, k_selection=k_selection
    )
