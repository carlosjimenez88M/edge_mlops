# Cómo continuar con otros capítulos

La plataforma vive en `master`. Cada capítulo del libro se trabaja en su propia rama
`chapter-XX`, que reutiliza toda la infraestructura (logger, orquestador, monitoreo,
CI/CD) y solo cambia lo específico del capítulo: los componentes de `src/` y `config.yaml`.

## Receta paso a paso

### 1. Crea la rama del capítulo

```bash
git checkout master
git pull
git checkout -b chapter-03
```

### 2. Ajusta `config.yaml`

Es la única fuente de verdad. Cambia, según el capítulo:

- `project.chapter`, `project.experiment_name`, `project.task` (`regression`/`classification`).
- `data.*` (fuente, target, split).
- `competition.metric` y `competition.models`.
- `sweep.*` (si vas a tunear) y `register.min_r2` (gate de calidad).

### 3. Adapta o añade componentes

La mayoría de capítulos reutilizan la estructura. Toca lo mínimo:

- **Dataset nuevo** → ajusta `src/data_load/loader.py` (y `data.source` en config / el `Literal` de `DataConfig`).
- **Features nuevas** → `src/data_preprocessing/transforms.py:engineer_features`.
- **Modelos nuevos** → añade al `MODEL_REGISTRY` de `src/model_competition/models.py`.
  Para clasificación, crea métricas equivalentes (accuracy/f1/roc_auc) y su dirección en `METRIC_LOWER_IS_BETTER`.
- **Espacio de búsqueda** → `src/sweep/search_space.py:SEARCH_SPACES`.
- **Esquema de la API** → `src/api/schemas.py` (features de entrada del nuevo problema).

### 4. Crear un componente desde cero (plantilla)

```
src/<nuevo_componente>/
├── MLproject          # name + entry point 'main' que llama a python main.py --config {config}
├── pyproject.toml     # dependencias propias del componente
├── main.py            # carga config, setup_mlflow, abre run, escribe artefactos
└── <funciones>.py     # lógica pura, con type hints
```

`main.py` mínimo:

```python
import sys; from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import click, mlflow
from src.common.config import load_config
from src.common.logging_utils import get_logger, success
from src.common.mlflow_utils import setup_mlflow
from src.common.paths import PROJECT_ROOT

logger = get_logger("<nuevo_componente>")

@click.command()
@click.option("--config", "config_path", default=str(PROJECT_ROOT / "config.yaml"))
def main(config_path: str) -> None:
    config = load_config(config_path)
    setup_mlflow(config)
    with mlflow.start_run(run_name="<nuevo_componente>"):
        ...  # lee artefactos del paso anterior, produce los tuyos
        success(logger, "listo")

if __name__ == "__main__":
    main()
```

Registra el paso en `orchestrator.py` → `STEP_COMPONENTS` y añádelo a
`orchestrator.steps` en `config.yaml`.

### 5. Prueba en local, luego haz push

```bash
uv sync --all-groups
uv run ruff check src orchestrator.py
uv run pytest -q
uv run python orchestrator.py            # corre la pipeline del capítulo
git add . && git commit -m "chapter 03: ..."
git push -u origin chapter-03            # ← dispara el Action en la Raspberry
```

### 6. Cuando funcione, mézclalo

Si el capítulo aporta mejoras reutilizables a la plataforma (no específicas del capítulo),
abre un PR de `chapter-XX` → `master`.

## Checklist por capítulo

- [ ] Rama `chapter-XX` creada desde `master`.
- [ ] `config.yaml` ajustado (chapter, experiment_name, task, data, metric, models).
- [ ] Componentes adaptados (loader / transforms / models / search_space / schema).
- [ ] `ruff` + `pytest` en verde.
- [ ] `orchestrator.py` corre la pipeline completa en local.
- [ ] Push → Action verde en la Pi (test → train → deploy → monitor).
