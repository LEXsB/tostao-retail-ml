"""Prueba de integración de la API de inferencia (FastAPI TestClient)."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from serving.api.main import app


@pytest.fixture(scope="module")
def client() -> TestClient:
    # El context manager dispara el lifespan (warmup: masters + modelos).
    with TestClient(app) as c:
        yield c


@pytest.mark.integration
def test_health(client: TestClient) -> None:
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["models_ready"] is True


@pytest.mark.integration
def test_caso_a_order(client: TestClient) -> None:
    payload = {
        "quantiles": {"0.1": 2, "0.5": 5, "0.9": 10},
        "precio_venta": 2500,
        "costo_unitario": 800,
        "costo_almacenamiento_semanal": 10,
        "stock_actual": 1,
    }
    resp = client.post("/v1/caso_a/order", json=payload)
    assert resp.status_code == 200
    body = resp.json()
    assert body["order_qty"] >= 0
    assert 0 <= body["critical_fractile"] <= 1


@pytest.mark.integration
def test_caso_b_combos(client: TestClient) -> None:
    resp = client.get("/v1/caso_b/combos")
    assert resp.status_code == 200
    combos = resp.json()
    assert isinstance(combos, list) and len(combos) > 0
    assert {"cluster", "producto_a", "producto_b", "lift"}.issubset(combos[0])


@pytest.mark.integration
def test_caso_c_spend(client: TestClient) -> None:
    payload = {"recency": 10, "frequency": 5, "edad": 35, "antiguedad": 300, "segmento": "Regular"}
    resp = client.post("/v1/caso_c/spend", json=payload)
    assert resp.status_code == 200
    assert resp.json()["gasto_esperado"] > 0
