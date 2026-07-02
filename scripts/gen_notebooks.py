"""Genera los notebooks de vitrina (EDA por caso) con nbformat.

Ejecutar con: ``uv run python scripts/gen_notebooks.py``

Los notebooks importan y reutilizan el framework (``tostao_ml``) y cargan datos
por el catálogo de Kedro: construyen la tabla maestra cruzada del caso y hacen el
EDA sobre ella, alternando código, figuras Plotly y mini-conclusiones en celdas
markdown. No duplican lógica: orquestan y narran.
"""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

NB_DIR = Path("notebooks")


def _save(nb: nbf.NotebookNode, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    nb.metadata["kernelspec"] = {
        "name": "python3",
        "display_name": "Python 3",
        "language": "python",
    }
    nbf.write(nb, str(path))


def setup_notebook() -> nbf.NotebookNode:
    return new_notebook(
        cells=[
            new_markdown_cell(
                "# 00 - Setup del entorno\n\n"
                "> **Objetivo.** Verificar el entorno reproducible, abrir la sesión de Kedro y "
                "confirmar que el catálogo expone las 12 fuentes crudas por caso.\n\n"
                "> **Salidas.** La lista de datasets del catálogo; punto de partida común de "
                "todos los notebooks.\n\n"
                "> **Cómo ejecutar.** `Restart & Run All`; determinista."
            ),
            new_code_cell(
                "from pathlib import Path\n"
                "from kedro.framework.session import KedroSession\n"
                "from kedro.framework.startup import bootstrap_project\n"
                "import tostao_ml\n\n"
                "PROJECT = Path.cwd().parent if Path.cwd().name == 'notebooks' else Path.cwd()\n"
                "bootstrap_project(PROJECT)\n"
                "print('tostao_ml', tostao_ml.__version__)"
            ),
            new_code_cell(
                "with KedroSession.create(project_path=PROJECT) as session:\n"
                "    catalog = session.load_context().catalog\n"
                "    datasets = [d for d in catalog.list() if d.startswith(('a_', 'b_', 'c_'))]\n"
                "datasets"
            ),
            new_markdown_cell(
                "**Conclusión.** El entorno reproducible (Python 3.13 + uv) está listo y el "
                "catálogo tipa las 12 fuentes crudas por caso. A partir de aquí, cada caso "
                "construye su **tabla maestra** cruzando sus fuentes y hace el EDA sobre ella."
            ),
        ]
    )


def eda_notebook(
    case: str, builder_call: str, target: str, intro: str, joins_md: str
) -> nbf.NotebookNode:
    load = (
        "from pathlib import Path\n"
        "from kedro.framework.session import KedroSession\n"
        "from kedro.framework.startup import bootstrap_project\n"
        "from tostao_ml.cases import masters\n"
        "from tostao_ml.framework.profiling import profile_dataset, build_eda_figures\n\n"
        "PROJECT = Path.cwd().parents[1] if Path.cwd().name.startswith('caso') else Path.cwd()\n"
        "bootstrap_project(PROJECT)\n"
        "with KedroSession.create(project_path=PROJECT) as session:\n"
        "    catalog = session.load_context().catalog\n"
        f"{builder_call}"
    )
    docstring = (
        f"# {case} - EDA sobre la tabla maestra\n\n"
        f"> **Objetivo.** {intro}\n\n"
        f"> **Entradas.** Fuentes crudas del caso cargadas por el catálogo de Kedro "
        f"(`data/01_raw`), sin `pd.read_csv` sueltos.\n\n"
        f"> **Salidas.** La tabla maestra cruzada, su diagnóstico de cobertura y las "
        f"conclusiones del EDA (tipado, VIF, correlaciones, información mutua, tests).\n\n"
        f"> **Cómo ejecutar.** Reinicia el kernel y ejecuta todo de arriba abajo "
        f"(`Restart & Run All`); es determinista. Reutiliza `tostao_ml` (no reimplementa "
        f"lógica): el notebook orquesta y narra."
    )
    return new_notebook(
        cells=[
            new_markdown_cell(docstring),
            new_markdown_cell(
                f"## 1. Construcción de la tabla maestra (cruce de fuentes)\n\n{joins_md}"
            ),
            new_code_cell(load),
            new_code_cell(
                "print('Master:', master.shape)\nprint('Cobertura de cruces:', join_report)\nmaster.head()"
            ),
            new_markdown_cell(
                "La **cobertura de cruces** confirma la integridad referencial: la proporción "
                "de filas del hecho que encontró match en cada fuente unida."
            ),
            new_markdown_cell("## 2. Perfilado estadístico (motor de EDA reutilizable)"),
            new_code_cell(
                f"profile = profile_dataset(master, name='{case}', target='{target}')\n"
                "print('Tipos:')\n"
                "for c, k in profile.types.items():\n    print(f'  {c:28s} {k.value}')\n"
                "profile.univariate.round(3)"
            ),
            new_markdown_cell("### Mini-conclusiones autogeneradas (cifras reales del run)"),
            new_code_cell(
                "from IPython.display import Markdown\nMarkdown(profile.narrative.to_markdown())"
            ),
            new_markdown_cell("## 3. Multicolinealidad y correlaciones"),
            new_code_cell(
                "figs = build_eda_figures(master, profile)\n"
                "display(profile.vif.round(3).to_frame('VIF'))\n"
                "figs.get('correlation_heatmap')"
            ),
            new_markdown_cell("## 4. Relación con el target e información mutua"),
            new_code_cell(
                "display(profile.mutual_information.round(4).to_frame('MI'))\n"
                "profile.bivariate.round(4)"
            ),
            new_code_cell(
                "for name, fig in figs.items():\n"
                "    if name.startswith(('dist__', 'target__')):\n"
                "        fig.show()"
            ),
            new_markdown_cell(
                "## 5. Conclusión\n\n"
                "El EDA sobre la **tabla maestra cruzada** (no fuente por fuente) revela el tipado "
                "de cada variable, la multicolinealidad (VIF), las correlaciones y qué features "
                "discriminan el target (tests + información mutua). Estas conclusiones guían el "
                "feature engineering y la elección de modelo del caso."
            ),
        ]
    )


A_BUILDER = (
    "    ventas = catalog.load('a_ventas_historicas'); cat = catalog.load('a_catalogo_productos')\n"
    "    tiendas = catalog.load('a_maestro_tiendas'); inv = catalog.load('a_inventario_actual')\n"
    "    trends = catalog.load('a_ground_truth_trends')\n"
    "master_d, join_report = masters.build_master_a(ventas, cat, tiendas, inv, trends)\n"
    "master = masters.aggregate_weekly_a(master_d)  # grano de forecast: semana x SKU x tienda\n"
)
B_BUILDER = (
    "    tickets = catalog.load('b_tickets'); detalle = catalog.load('b_detalle_tickets')\n"
    "    cat = catalog.load('b_catalogo_productos')\n"
    "master, join_report = masters.build_master_b(tickets, detalle, cat)\n"
)
C_BUILDER = (
    "    trans = catalog.load('c_transacciones_resumen'); loy = catalog.load('c_clientes_loyalty')\n"
    "    exo = catalog.load('c_variables_exogenas'); promo = catalog.load('c_promociones_activas')\n"
    "master, join_report = masters.build_master_c(trans, loy, exo, promo)\n"
)


def main() -> None:
    _save(setup_notebook(), NB_DIR / "00_setup.ipynb")
    _save(
        eda_notebook(
            "Caso A — Abastecimiento",
            A_BUILDER,
            "unidades_vendidas",
            "Forecast de demanda semanal por SKU-tienda. Cruzo ventas con catálogo, "
            "maestro de tiendas, inventario y tendencia de referencia.",
            "`ventas ⨝ catálogo(producto) ⨝ maestro_tiendas(tienda) ⨝ inventario(tienda+producto) "
            "⨝ ground_truth(tienda+producto)`, luego agrego a grano **semanal**.",
        ),
        NB_DIR / "caso_a" / "01_eda.ipynb",
    )
    _save(
        eda_notebook(
            "Caso B — Combos",
            B_BUILDER,
            "importe_linea",
            "Creación de combos por afinidad de co-compra. Cruzo el detalle de tickets con "
            "la cabecera y el catálogo de productos.",
            "`detalle_tickets ⨝ tickets(id_ticket) ⨝ catálogo(id_producto)` — una fila por "
            "línea de ticket enriquecida.",
        ),
        NB_DIR / "caso_b" / "01_eda.ipynb",
    )
    _save(
        eda_notebook(
            "Caso C — AOV",
            C_BUILDER,
            "total_venta",
            "Drivers y predicción del ticket promedio. Cruzo transacciones con loyalty, "
            "variables exógenas e intensidad de promociones.",
            "`transacciones ⨝ loyalty(cliente) ⨝ exógenas(fecha+tienda) ⨝ intensidad_promos"
            "(fecha+tienda)` — una fila por ticket enriquecida.",
        ),
        NB_DIR / "caso_c" / "01_eda.ipynb",
    )
    print("Notebooks generados:")
    for p in sorted(NB_DIR.rglob("*.ipynb")):
        print("  ", p)


if __name__ == "__main__":
    main()
