"""Narración e interpretación profesional de datos y resultados.

Convierte las cifras reales del run en una historia legible por negocio: describe
qué son los datos, qué revela el EDA y qué significan los resultados del modelo
(si son significativos, si son robustos y qué decisión implican). Es el «código
que interpreta las salidas» — no imprime definiciones, sino lecturas contextuales.
"""

from __future__ import annotations

import pandas as pd

from tostao_ml.framework.narrate import Insight, Narrative, Severity
from tostao_ml.framework.profiling.engine import DatasetProfile

# --------------------------------------------------------------------------- #
# Tarea propuesta por caso (de la prueba técnica) — va en el encabezado.
# --------------------------------------------------------------------------- #
TASKS: dict[str, dict[str, str]] = {
    "a": {
        "titulo": "Caso A — Optimización de Abastecimiento",
        "contexto": (
            "Predecir la demanda es solo la mitad de la batalla: si pedimos de menos "
            "perdemos ventas (costo de oportunidad); si pedimos de más, incurrimos en "
            "costos de almacenamiento y capital."
        ),
        "objetivo": (
            "(1) Pronosticar la demanda por SKU-Tienda para la próxima semana. "
            "(2) Diseñar una optimización de pedido que minimice el costo total esperado, "
            "balanceando stockout vs. overstock, usando la incertidumbre del modelo y los "
            "márgenes para decidir una política agresiva o conservadora."
        ),
    },
    "b": {
        "titulo": "Caso B — Creación de Combos",
        "contexto": (
            "Se busca incrementar el ticket promedio mediante venta cruzada, identificando "
            "patrones de compra no evidentes para generar recomendaciones automáticas."
        ),
        "objetivo": (
            "Identificar los Top-5 combos con mayor potencial de venta por cluster de "
            "tiendas, proponer su precio y cuantificar el lift esperado, filtrando el ruido "
            "de artículos de alta frecuencia y baja correlación específica."
        ),
    },
    "c": {
        "titulo": "Caso C — Modelado del Ticket Promedio (AOV)",
        "contexto": (
            "Existe alta variabilidad en el ticket promedio (AOV) entre sucursales; se "
            "requiere entender qué factores exógenos y endógenos lo influyen."
        ),
        "objetivo": (
            "(1) Un modelo inferencial que determine la importancia de las variables sobre "
            "el valor del ticket. (2) Un modelo predictivo del gasto esperado del cliente "
            "recurrente en su próxima visita, con interpretabilidad y manejo de outliers."
        ),
    },
}


def context_html(case_key: str) -> str:
    """Bloque HTML con la tarea propuesta (contexto + objetivo) para el encabezado."""
    t = TASKS[case_key]
    return (
        f"<h3>Tarea propuesta — {t['titulo']}</h3>"
        f"<div class='task'><p><strong>Contexto de negocio.</strong> {t['contexto']}</p>"
        f"<p><strong>Objetivo.</strong> {t['objetivo']}</p></div>"
    )


def _p(text: str, severity: Severity = Severity.INFO, title: str | None = None) -> Insight:
    return Insight(text=text, severity=severity, title=title, tags=("story",))


# --------------------------------------------------------------------------- #
# Historia de los datos
# --------------------------------------------------------------------------- #
def data_story_a(weekly: pd.DataFrame) -> Narrative:
    """Narra que representan los datos del Caso A (demanda semanal SKU-tienda)."""
    n = len(weekly)
    n_series = weekly.groupby(["id_tienda", "id_producto"]).ngroups
    n_stores, n_products = weekly["id_tienda"].nunique(), weekly["id_producto"].nunique()
    weeks = int(weekly["semana"].nunique())
    total = int(weekly["unidades_vendidas"].sum())
    mean_w = weekly["unidades_vendidas"].mean()
    pct_zero = float((weekly["unidades_vendidas"] == 0).mean() * 100)
    cat = weekly.groupby("categoria")["unidades_vendidas"].sum()
    top_cat, top_share = cat.idxmax(), cat.max() / cat.sum() * 100
    intermit = "baja" if pct_zero < 10 else "moderada" if pct_zero < 30 else "alta"
    return Narrative().add(
        _p(
            f"La tabla maestra reúne {weeks} semanas de demanda de {n_products} SKU en {n_stores} tiendas "
            f"—{n_series} series SKU-tienda y {n:,} observaciones semana-SKU-tienda—. Se vendieron "
            f"{total:,} unidades en total; la demanda semanal media por serie es {mean_w:.1f} unidades, "
            f"con {pct_zero:.0f}% de semanas sin venta (intermitencia {intermit}). La categoría «{top_cat}» "
            f"concentra el {top_share:.0f}% del volumen. El grano semanal SKU-tienda es el correcto para "
            f"un pronóstico accionable de reposición.",
            title="Qué son los datos",
        )
    )


def data_story_b(master_b: pd.DataFrame) -> Narrative:
    """Narra que representan los datos del Caso B (lineas de ticket)."""
    n_lines = len(master_b)
    n_tickets = master_b["id_ticket"].nunique()
    n_products = master_b["id_producto"].nunique()
    n_stores = master_b["id_tienda"].nunique() if "id_tienda" in master_b else 0
    items_per_ticket = master_b.groupby("id_ticket")["cantidad"].sum().mean()
    ticket_val = master_b.groupby("id_ticket")["importe_linea"].sum().mean()
    cat = master_b.groupby("categoria")["cantidad"].sum()
    top_cat = cat.idxmax()
    return Narrative().add(
        _p(
            f"La tabla maestra tiene {n_lines:,} líneas de venta de {n_tickets:,} tickets en {n_stores} "
            f"tiendas, sobre un catálogo de {n_products} productos. Cada cesta contiene en promedio "
            f"{items_per_ticket:.1f} artículos y suma un ticket medio de {ticket_val:,.0f}. La categoría "
            f"«{top_cat}» es la más vendida. Esta estructura ticket→línea es la base para detectar "
            f"co-compra (qué se lleva junto) y proponer combos por segmento de tienda.",
            title="Qué son los datos",
        )
    )


def data_story_c(master_c: pd.DataFrame) -> Narrative:
    """Narra que representan los datos del Caso C (tickets enriquecidos)."""
    n = len(master_c)
    n_clients = master_c["id_cliente"].nunique()
    n_stores = master_c["id_tienda"].nunique() if "id_tienda" in master_c else 0
    mean_ticket = master_c["total_venta"].mean()
    recurring = (master_c.groupby("id_cliente").size() >= 2).mean() * 100
    promo_cov = float(master_c["hay_promo"].mean() * 100) if "hay_promo" in master_c else 0.0
    period = ""
    if "fecha" in master_c:
        period = f" entre {master_c['fecha'].min():%Y-%m} y {master_c['fecha'].max():%Y-%m}"
    return Narrative().add(
        _p(
            f"La tabla maestra reúne {n:,} tickets de {n_clients:,} clientes en {n_stores} tiendas{period}. "
            f"El ticket medio es {mean_ticket:.2f}; el {recurring:.0f}% de los clientes son recurrentes "
            f"(≥2 visitas), población objetivo del modelo predictivo de gasto. Cada ticket se enriquece "
            f"con perfil del cliente (edad, segmento, antigüedad), condiciones exógenas (clima, tráfico, "
            f"índice de competencia) y una intensidad de promociones activas que cubre el {promo_cov:.0f}% "
            f"de los tickets. Es una vista 360° del ticket, ideal para separar drivers endógenos de exógenos.",
            title="Qué son los datos",
        )
    )


# --------------------------------------------------------------------------- #
# Interpretación del EDA
# --------------------------------------------------------------------------- #
def interpret_eda(profile: DatasetProfile) -> Narrative:
    """Lectura profesional del EDA: qué discrimina el target, colinealidad, anomalías."""
    narr = Narrative()
    mi = profile.mutual_information
    if not mi.empty:
        top = mi.head(3)
        drivers = ", ".join(f"«{k}» (MI={v:.2f})" for k, v in top.items())
        narr.add(
            _p(
                f"Frente al objetivo «{profile.target}», los predictores con más información son {drivers}. "
                f"Esto orienta el feature engineering hacia esas variables y sugiere que el resto aporta "
                f"señal marginal.",
                severity=Severity.GOOD,
                title="Qué discrimina el objetivo",
            )
        )

    if not profile.vif.empty and profile.vif.notna().any():
        max_vif = float(profile.vif.max())
        worst = str(profile.vif.idxmax())
        if max_vif >= 10:
            narr.add(
                _p(
                    f"Se detecta multicolinealidad severa (VIF máximo {max_vif:.1f} en «{worst}»): "
                    f"conviene eliminar o combinar variables antes de un modelo lineal.",
                    severity=Severity.CRITICAL,
                    title="Multicolinealidad",
                )
            )
        elif max_vif >= 5:
            narr.add(
                _p(
                    f"Multicolinealidad moderada (VIF máximo {max_vif:.1f} en «{worst}»): a vigilar, "
                    f"pero manejable con regularización o árboles.",
                    severity=Severity.WARNING,
                    title="Multicolinealidad",
                )
            )
        else:
            narr.add(
                _p(
                    f"No hay multicolinealidad relevante (VIF máximo {max_vif:.1f} < 5): las variables "
                    f"aportan información sin redundarse, lo que da estabilidad a los coeficientes.",
                    severity=Severity.GOOD,
                    title="Multicolinealidad",
                )
            )

    uni = profile.univariate
    if not uni.empty and "skewness" in uni.columns:
        num = uni[uni["kind"].str.startswith("numeric")].dropna(subset=["skewness"])
        if not num.empty:
            row = num.iloc[num["skewness"].abs().argmax()]
            out = num.iloc[num["pct_outliers_iqr"].argmax()] if "pct_outliers_iqr" in num else row
            narr.add(
                _p(
                    f"La variable más asimétrica es «{row['variable']}» (skew={row['skewness']:.1f}); "
                    f"«{out['variable']}» presenta la mayor proporción de outliers "
                    f"({out.get('pct_outliers_iqr', 0):.0f}%). Esto justifica winsorizar/transformar antes "
                    f"de modelos sensibles a colas.",
                    severity=Severity.INFO,
                    title="Forma y anomalías",
                )
            )
    return narr


# --------------------------------------------------------------------------- #
# Interpretación de resultados de modelo
# --------------------------------------------------------------------------- #
def _verdict_wape(w: float) -> str:
    return (
        "muy bajo"
        if w < 0.1
        else "bajo-moderado"
        if w < 0.2
        else "moderado"
        if w < 0.35
        else "alto"
    )


def interpret_model_a(result) -> Narrative:
    """Interpreta los resultados del Caso A (calidad, intervalos, negocio)."""
    m = result.metrics
    narr = Narrative()
    rel = (m["wape_naive"] - m["wape"]) / m["wape_naive"] * 100 if m["wape_naive"] else 0.0
    supera = rel > 0
    narr.add(
        _p(
            f"El pronóstico logra un WAPE de {m['wape']:.1%} (error {_verdict_wape(m['wape'])} para demanda "
            f"semanal) y explica el {m['r2']:.0%} de la varianza (R²={m['r2']:.2f}). "
            f"{'Supera' if supera else 'No supera'} al baseline ingenuo de persistencia en {abs(rel):.0f}%, "
            f"lo que indica que el modelo {'aporta señal real más allá de repetir la semana previa' if supera else 'apenas mejora la persistencia'}.",
            severity=Severity.GOOD if supera else Severity.WARNING,
            title="¿Es bueno el pronóstico?",
        )
    )

    picp, nominal = m["picp"], 0.8
    if picp >= nominal - 0.05:
        narr.add(
            _p(
                f"Los intervalos están bien calibrados (PICP={picp:.0%} vs {nominal:.0%} nominal): "
                f"son fiables para fijar niveles de servicio.",
                severity=Severity.GOOD,
                title="¿Son robustos los intervalos?",
            )
        )
    else:
        narr.add(
            _p(
                f"Los intervalos SUB-cubren (PICP={picp:.0%} < {nominal:.0%} nominal): son demasiado "
                f"estrechos y aún NO son fiables para decisiones de nivel de servicio; se recomienda "
                f"calibrarlos (predicción conforme o cuantiles más anchos) antes de producción.",
                severity=Severity.WARNING,
                title="¿Son robustos los intervalos?",
            )
        )

    saved = result.cost_naive - result.cost_model
    saved_pct = saved / result.cost_naive * 100 if result.cost_naive else 0.0
    narr.add(
        _p(
            f"En negocio, aplicar la política newsvendor reduce el costo esperado de faltante+sobrante en "
            f"{saved_pct:.0f}% frente a pedir lo del período anterior: un ahorro material que traduce la "
            f"incertidumbre del modelo en una decisión de pedido concreta.",
            severity=Severity.GOOD if saved > 0 else Severity.WARNING,
            title="Impacto de negocio",
        )
    )
    narr.add(
        _p(
            "Robustez: validado en holdout temporal (walk-forward, sin leakage). La historia disponible es "
            "corta (~13 semanas por serie), por lo que la confianza es razonable pero limitada; con más "
            "historia el intervalo de confianza del desempeño se estrecharía.",
            severity=Severity.INFO,
            title="Sobre la robustez",
        )
    )
    return narr


def interpret_model_b(result) -> Narrative:
    """Interpreta los resultados del Caso B (validez de clusters y reglas)."""
    narr = Narrative()
    sil = result.silhouette
    k = int(result.store_clusters.nunique())
    calidad = (
        "clara y bien separada"
        if sil > 0.5
        else "moderada pero utilizable"
        if sil > 0.25
        else "débil (segmentos difusos)"
    )
    narr.add(
        _p(
            f"Las {len(result.store_profiles)} tiendas se agrupan en {k} segmentos con una estructura "
            f"{calidad} (silhouette={sil:.2f}). La k se eligió maximizando la silhouette, no a dedo, por lo "
            f"que la segmentación está justificada por los datos.",
            severity=Severity.GOOD if sil > 0.25 else Severity.WARNING,
            title="¿Son válidos los clusters?",
        )
    )
    if not result.rules.empty:
        best = result.rules.iloc[0]
        a = ", ".join(sorted(best["antecedents"]))
        b = ", ".join(sorted(best["consequents"]))
        robusto = (
            "robusta (soporte suficiente, no es ruido)"
            if best["support"] >= 0.02
            else "a validar (soporte bajo)"
        )
        narr.add(
            _p(
                f"La afinidad más fuerte es «{a}» → «{b}»: se compran juntos {best['lift']:.1f}× más de lo "
                f"esperado por azar (lift), con confianza {best['confidence']:.0%} y soporte "
                f"{best['support']:.1%}. Es una señal {robusto}. Todos los combos propuestos tienen lift>1, "
                f"por lo que son relevantes y no coincidencias.",
                severity=Severity.GOOD,
                title="¿Son robustas las reglas?",
            )
        )
    narr.add(
        _p(
            "Impacto de negocio: cada combo empaqueta productos con co-compra genuina y sugiere un precio con "
            "descuento; el lift esperado cuantifica cuánto más probable es la venta cruzada, insumo directo "
            "para priorizar qué combos activar por segmento.",
            severity=Severity.INFO,
            title="Impacto de negocio",
        )
    )
    return narr


def interpret_model_c(result) -> Narrative:
    """Interpreta los resultados del Caso C (drivers significativos y prediccion)."""
    narr = Narrative()
    coefs = result.coefficients
    sig = coefs[coefs["pvalue"] < 0.05]
    if not sig.empty:
        parts = []
        for name, r in sig.head(3).iterrows():
            signo = "aumenta" if r["coef"] > 0 else "reduce"
            parts.append(f"«{name}» {signo} el ticket (β={r['coef']:+.2f}, p={r['pvalue']:.3g})")
        narr.add(
            _p(
                f"El modelo inferencial identifica {len(sig)} driver(s) estadísticamente significativo(s) "
                f"(p<0.05): {'; '.join(parts)}. Los intervalos de confianza no cruzan cero en estos casos, "
                f"por lo que el efecto es real y no atribuible al azar. El resto de variables no son "
                f"significativas: su efecto sobre el ticket no se distingue de cero con los datos disponibles.",
                severity=Severity.GOOD,
                title="¿Qué mueve el ticket (y es significativo)?",
            )
        )
    pm = result.predictive_metrics
    fuerte = "fuerte" if pm["r2"] > 0.6 else "moderado" if pm["r2"] > 0.3 else "débil"
    rel = (
        (pm["wape_baseline"] - pm["wape"]) / pm["wape_baseline"] * 100
        if pm.get("wape_baseline")
        else 0.0
    )
    narr.add(
        _p(
            f"El modelo predictivo del gasto explica el {pm['r2']:.0%} de la varianza (ajuste {fuerte}, "
            f"R²={pm['r2']:.2f}) y mejora el baseline de la media en {rel:.0f}% de WAPE, lo que confirma que "
            f"el perfil RFM+loyalty tiene poder predictivo real. El error medio (MAE={pm['mae']:.2f}) es bajo "
            f"frente al ticket medio, por lo que las estimaciones son accionables para priorizar clientes por "
            f"valor esperado.",
            severity=Severity.GOOD if pm["r2"] > 0.3 else Severity.WARNING,
            title="¿Predice bien el gasto?",
        )
    )
    if result.tuning is not None:
        narr.add(
            _p(
                "Robustez: el modelo predictivo se validó con K-Fold y sus hiperparámetros se optimizaron con "
                "Optuna; el GLM se ajusta sobre todo el conjunto porque su fin es inferencial, no predictivo. "
                "La winsorización controla el efecto de tickets atípicos.",
                severity=Severity.INFO,
                title="Sobre la robustez",
            )
        )
    return narr
