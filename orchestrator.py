"""Orquestador de la pipeline MLOps.

Lee config.yaml y ejecuta, en orden, cada componente como un MLflow Project
(arquitectura component-artifact). Cada paso recibe la misma configuracion y
produce artefactos que consume el siguiente. El semaforo de colores indica el
estado: verde = ok, amarillo = advertencia/omitido, rojo = fallo.

Uso:
    python orchestrator.py                 # corre los pasos de config.yaml
    python orchestrator.py --only data_load model_competition
    python orchestrator.py --from model_competition
"""

from __future__ import annotations

import os
from pathlib import Path

import click
import mlflow

from src.common.config import load_config
from src.common.logging_utils import get_logger, success
from src.common.mlflow_utils import setup_mlflow
from src.common.paths import CONFIG_PATH, SRC_DIR, ensure_dirs

logger = get_logger("orchestrator")

# Mapea nombre de paso -> (directorio del componente, entry point del MLproject).
STEP_COMPONENTS: dict[str, tuple[Path, str]] = {
    "data_load": (SRC_DIR / "data_load", "main"),
    "data_validation": (SRC_DIR / "data_validation", "main"),
    "data_preprocessing": (SRC_DIR / "data_preprocessing", "main"),
    "model_competition": (SRC_DIR / "model_competition", "main"),
    "sweep": (SRC_DIR / "sweep", "main"),
    "register": (SRC_DIR / "register", "main"),
}


def run_step(step: str, env_manager: str) -> None:
    component_dir, entry_point = STEP_COMPONENTS[step]
    logger.info("──> Ejecutando paso: %s (%s)", step, component_dir.name)
    submitted = mlflow.projects.run(
        uri=str(component_dir),
        entry_point=entry_point,
        parameters={"config": str(CONFIG_PATH)},
        env_manager=env_manager,
        synchronous=True,
    )
    if submitted.get_status() != "FINISHED":
        raise RuntimeError(f"El paso '{step}' termino con estado {submitted.get_status()}")


@click.command()
@click.option("--config", "config_path", default=str(CONFIG_PATH))
@click.option("--only", multiple=True, help="Corre solo estos pasos.")
@click.option("--from", "from_step", default=None, help="Empieza desde este paso.")
@click.option("--enable-sweep", is_flag=True, help="Fuerza la ejecucion del sweep de W&B.")
def main(config_path: str, only: tuple[str, ...], from_step: str | None, enable_sweep: bool) -> None:
    ensure_dirs()
    config = load_config(config_path)
    if enable_sweep:
        config.sweep.enabled = True
        # Se propaga al subproceso del componente sweep (que relee config.yaml).
        os.environ["EDGE_FORCE_SWEEP"] = "1"
        logger.info("Sweep forzado por --enable-sweep.")
    setup_mlflow(config)

    steps = list(config.orchestrator.steps)
    if only:
        steps = [s for s in steps if s in only]
    if from_step:
        if from_step not in steps:
            logger.error("El paso '%s' no esta en la lista. Pasos: %s", from_step, steps)
            raise SystemExit(1)
        steps = steps[steps.index(from_step):]

    # Omite el sweep si no esta habilitado, para no gastar corridas.
    if "sweep" in steps and not config.sweep.enabled:
        logger.warning("Paso 'sweep' omitido (sweep.enabled=false en config.yaml).")
        steps = [s for s in steps if s != "sweep"]

    logger.info("Pipeline: %s", " → ".join(steps))
    for step in steps:
        try:
            run_step(step, config.orchestrator.env_manager)
            success(logger, f"Paso '{step}' completado.")
        except Exception as exc:  # noqa: BLE001
            logger.error("Paso '%s' FALLO: %s", step, exc)
            raise SystemExit(1) from exc

    success(logger, "Pipeline completa. Revisa MLflow y artifacts/.")


if __name__ == "__main__":
    main()
