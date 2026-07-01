"""Punto de entrada de línea de comandos del proyecto Kedro ``tostao_ml``.

Permite ejecutar ``python -m tostao_ml`` o el script ``tostao-ml`` como envoltorio
del CLI de Kedro, respetando el directorio de proyecto y los metadatos.
"""

from __future__ import annotations

from pathlib import Path


def main(*args, **kwargs) -> None:
    """Lanza el CLI de Kedro con la configuración del proyecto."""
    from kedro.framework.cli.utils import find_run_command
    from kedro.framework.project import configure_project

    package_name = Path(__file__).parent.name
    configure_project(package_name)
    run = find_run_command(package_name)
    run(*args, **kwargs)


if __name__ == "__main__":
    main()
