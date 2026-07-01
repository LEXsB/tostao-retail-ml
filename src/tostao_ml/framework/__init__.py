"""Framework reutilizable y agnóstico al caso.

Núcleo del producto: io, validation, profiling, features, models, tuning,
evaluation, optimization, interpret, viz, narrate y reporting. Los pipelines
Kedro (``tostao_ml.pipelines``) solo orquestan estas piezas; los tres casos
(A/B/C) las reutilizan y extienden sin duplicar lógica.
"""
