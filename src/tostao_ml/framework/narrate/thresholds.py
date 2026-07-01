"""Umbrales configurables que gobiernan las reglas de narración.

Los valores por defecto replican convenciones estadísticas habituales, pero se
sobreescriben desde ``conf/base/parameters.yml`` (clave ``narrate``) para no
hardcodear criterios en el código.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class NarrationThresholds:
    """Cortes numéricos usados al redactar mini-conclusiones.

    Attributes:
        vif_moderate: VIF a partir del cual la multicolinealidad es moderada.
        vif_severe: VIF a partir del cual la multicolinealidad es severa.
        corr_strong: |correlación| considerada fuerte.
        corr_moderate: |correlación| considerada moderada.
        pvalue_alpha: Nivel de significancia para tests de hipótesis.
        effect_small/medium/large: Cortes de tamaño de efecto (Cohen).
        mi_relevant: Información mutua mínima para considerar relevante un feature.
    """

    vif_moderate: float = 5.0
    vif_severe: float = 10.0
    corr_strong: float = 0.7
    corr_moderate: float = 0.4
    pvalue_alpha: float = 0.05
    effect_small: float = 0.2
    effect_medium: float = 0.5
    effect_large: float = 0.8
    mi_relevant: float = 0.05

    @classmethod
    def from_dict(cls, params: dict | None) -> NarrationThresholds:
        """Crea umbrales desde un diccionario de parámetros (ignora claves extra)."""
        if not params:
            return cls()
        known = set(cls.__dataclass_fields__)
        return cls(**{k: v for k, v in params.items() if k in known})

    def effect_label(self, effect_size: float) -> str:
        """Etiqueta cualitativa de un tamaño de efecto (valor absoluto)."""
        magnitude = abs(effect_size)
        if magnitude >= self.effect_large:
            return "grande"
        if magnitude >= self.effect_medium:
            return "mediano"
        if magnitude >= self.effect_small:
            return "pequeño"
        return "insignificante"

    def corr_label(self, corr: float) -> str:
        """Etiqueta cualitativa de una correlación (valor absoluto)."""
        magnitude = abs(corr)
        if magnitude >= self.corr_strong:
            return "fuerte"
        if magnitude >= self.corr_moderate:
            return "moderada"
        return "débil"
