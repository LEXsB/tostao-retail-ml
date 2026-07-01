"""Pruebas del motor de narración."""

from __future__ import annotations

import pytest

from tostao_ml.framework.narrate import Insight, NarrationThresholds, Narrative, Severity, rules


@pytest.mark.unit
def test_narrative_sorts_by_severity_and_reports_worst() -> None:
    narrative = Narrative()
    narrative.add(Insight("ok", Severity.GOOD))
    narrative.add(Insight("peligro", Severity.CRITICAL))
    narrative.add(Insight("aviso", Severity.WARNING))
    narrative.add(None)  # se ignora

    assert len(narrative) == 3
    assert narrative.worst_severity is Severity.CRITICAL
    assert next(iter(narrative.sorted_by_severity())).severity is Severity.CRITICAL


@pytest.mark.unit
def test_thresholds_labels() -> None:
    th = NarrationThresholds()
    assert th.effect_label(0.9) == "grande"
    assert th.effect_label(0.05) == "insignificante"
    assert th.corr_label(0.8) == "fuerte"
    assert th.corr_label(0.1) == "débil"


@pytest.mark.unit
def test_thresholds_from_dict_ignores_unknown_keys() -> None:
    th = NarrationThresholds.from_dict({"vif_severe": 12.0, "no_existe": 1})
    assert th.vif_severe == 12.0


@pytest.mark.unit
def test_vif_rule_severity_levels() -> None:
    assert rules.vif_insight("x", 3.0) is None  # bajo → sin insight
    assert rules.vif_insight("x", 6.0).severity is Severity.WARNING
    assert rules.vif_insight("x", 11.0).severity is Severity.CRITICAL
    assert rules.vif_insight("x", float("inf")).severity is Severity.CRITICAL


@pytest.mark.unit
def test_correlation_rule_direction_and_metrics() -> None:
    ins = rules.correlation_insight("a", "b", -0.85, method="Spearman")
    assert "negativa" in ins.text and "fuerte" in ins.text
    assert ins.metrics["Spearman"] == -0.85
    assert ins.severity is Severity.WARNING


@pytest.mark.unit
def test_metric_vs_baseline_improvement() -> None:
    ins = rules.metric_vs_baseline_insight(
        "WAPE", model_value=0.20, baseline_value=0.40, higher_is_better=False
    )
    assert ins.severity is Severity.GOOD
    assert ins.metrics["rel_change_pct"] == -50.0
