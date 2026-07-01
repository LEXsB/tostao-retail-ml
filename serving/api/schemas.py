"""Esquemas Pydantic de la API de inferencia (validación de contratos)."""

from __future__ import annotations

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    """Estado del servicio."""

    status: str
    version: str
    models_ready: bool


class OrderRequest(BaseModel):
    """Solicitud de recomendación de pedido (Caso A, newsvendor)."""

    quantiles: dict[float, float] = Field(
        ..., description="Mapa cuantil→demanda pronosticada, p. ej. {0.1: 3, 0.5: 5, 0.9: 9}"
    )
    precio_venta: float = Field(..., gt=0)
    costo_unitario: float = Field(..., ge=0)
    costo_almacenamiento_semanal: float = Field(..., ge=0)
    stock_actual: float = Field(0.0, ge=0)
    safety_factor: float = Field(1.0, gt=0)


class OrderResponse(BaseModel):
    """Recomendación de pedido."""

    critical_fractile: float
    order_up_to: float
    order_qty: float


class SpendRequest(BaseModel):
    """Solicitud de predicción de gasto del cliente recurrente (Caso C)."""

    recency: float = Field(..., ge=0)
    frequency: int = Field(..., ge=1)
    edad: float = Field(..., gt=0)
    antiguedad: float = Field(..., ge=0)
    segmento: str = Field("Regular")


class SpendResponse(BaseModel):
    """Gasto esperado predicho."""

    gasto_esperado: float


class Combo(BaseModel):
    """Combo propuesto (Caso B)."""

    cluster: int
    producto_a: str
    producto_b: str
    lift: float
    precio_combo: float
    precio_lista: float
