"""Modelo lineal generalizado (GLM) inferencial vía statsmodels.

Pensado para el Caso C: cuantificar drivers del ticket con coeficientes,
errores estándar e intervalos de confianza (interpretabilidad de primer orden),
complementando a SHAP.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm

from tostao_ml.framework.types import TaskType

from .base import BaseModel
from .registry import register_model

_FAMILIES = {
    "gaussian": sm.families.Gaussian,
    "gamma": sm.families.Gamma,
    "poisson": sm.families.Poisson,
}


@register_model("glm")
class GLMModel(BaseModel):
    """GLM con familia configurable y salida inferencial completa.

    Args:
        family: Familia del GLM (``gaussian``/``gamma``/``poisson``).
        add_constant: Si añade intercepto.
    """

    task = TaskType.REGRESSION

    def __init__(
        self, family: str = "gaussian", add_constant: bool = True, **kwargs: object
    ) -> None:
        """Inicializa la instancia con sus parametros de configuracion."""
        super().__init__(name="glm")
        if family not in _FAMILIES:
            raise ValueError(f"Familia GLM desconocida: {family}. Opciones: {sorted(_FAMILIES)}")
        self.family = family
        self.add_constant = add_constant
        self._kwargs = kwargs
        self.metadata.params = {"family": family, "add_constant": add_constant}

    def _design(self, X: pd.DataFrame) -> pd.DataFrame:
        return sm.add_constant(X, has_constant="add") if self.add_constant else X

    def fit(self, X: pd.DataFrame, y: pd.Series | np.ndarray | None = None) -> GLMModel:
        """Ajusta el estimador con los datos y devuelve la propia instancia."""
        design = self._design(X)
        family = _FAMILIES[self.family]()
        self._result = sm.GLM(np.asarray(y, dtype=float), design, family=family).fit()
        self.metadata.feature_names = list(X.columns)
        self.metadata.extra["aic"] = float(self._result.aic)
        self.metadata.extra["deviance"] = float(self._result.deviance)
        self.is_fitted_ = True
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Devuelve las predicciones para las observaciones de entrada."""
        self._check_fitted()
        return np.asarray(self._result.predict(self._design(X)))

    def coefficients_frame(self, alpha: float = 0.05) -> pd.DataFrame:
        """Tabla de coeficientes con error estándar, p-valor e IC.

        Args:
            alpha: Nivel para el intervalo de confianza (0.05 → IC 95%).

        Returns:
            DataFrame indexado por variable con ``coef``, ``std_err``,
            ``pvalue``, ``ci_lower``, ``ci_upper``.
        """
        self._check_fitted()
        conf = self._result.conf_int(alpha=alpha)
        return pd.DataFrame(
            {
                "coef": self._result.params,
                "std_err": self._result.bse,
                "pvalue": self._result.pvalues,
                "ci_lower": conf[0],
                "ci_upper": conf[1],
            }
        )
