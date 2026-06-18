# Arquitectura

## Principio: component–artifact

Cada paso de la pipeline es un **componente autónomo** ubicado en `src/<nombre>/`.
Un componente:

1. Es un **MLflow Project** (tiene un archivo `MLproject` con sus *entry points*).
2. Declara sus **propias dependencias** (`pyproject.toml`) y puede instalarse aislado con `uv`.
3. Tiene un `main.py` (punto de entrada) y módulos con las funciones puras.
4. **Lee y escribe artefactos** en disco (`data/`, `artifacts/`) y en MLflow.

El acoplamiento entre componentes es solo a través de **artefactos**, no de imports
cruzados de lógica de negocio (sí comparten utilidades transversales en `src/common`).
Esto permite ejecutar, cachear y razonar cada paso por separado.

```
config.yaml ──► orchestrator.py ──► mlflow.projects.run(<componente>) ──► artefactos ──► siguiente paso
```

## El orquestador

`orchestrator.py`:

- Carga y valida `config.yaml` con Pydantic.
- Resuelve la lista de pasos (`config.orchestrator.steps`), con filtros `--only` / `--from`.
- Ejecuta cada componente con `mlflow.projects.run(..., env_manager=local)`.
- Aplica el **semáforo de colores**: 🟢 paso ok · 🟡 omitido/advertencia · 🔴 fallo (y aborta).

> **Decisión — un único venv raíz.** Por defecto los componentes corren con
> `env_manager=local` sobre el `.venv` raíz (`uv sync`), rápido y liviano para la Pi.
> Cada componente conserva su `MLproject` + `pyproject.toml`, así que también puede
> correrse aislado: `cd src/<comp> && uv run python main.py`.

## Flujo de artefactos (Capítulo 2)

| Paso | Lee | Escribe |
|------|-----|---------|
| `data_load`          | —                              | `data/raw/housing.parquet` |
| `data_validation`    | `housing.parquet`              | `artifacts/validation_report.json`, `histograms.png` |
| `data_preprocessing` | `housing.parquet`              | `data/processed/{train,test}.parquet`, `feature_names.json` |
| `model_competition`  | splits                         | `artifacts/best_model.json`, `leaderboard.json`, modelo en MLflow |
| `sweep`              | `best_model.json` + splits     | `artifacts/tuned_model.joblib`, actualiza `best_model.json` |
| `register`           | `best_model.json` + test split | versión en MLflow Model Registry |
| `api`                | modelo registrado              | servicio HTTP |

## Tracking y registro (MLflow)

- **Tracking**: `sqlite:///mlflow.db` (el Model Registry necesita backend con BD; `file://` no lo soporta).
- La URI se **absolutiza en tiempo de ejecución** (`src/common/mlflow_utils.resolve_tracking_uri`)
  para que todos los componentes —cada uno corre en su propio directorio— compartan la misma BD.
- **Registro**: `register` aplica un gate de calidad (`min_r2`) y promueve por
  *champion-challenger* (solo asciende si mejora al modelo en el stage objetivo).

## Decisiones clave

- **Pydantic** valida toda la configuración: si falta una clave o tiene mal tipo, falla temprano.
- **Preprocesamiento dentro del Pipeline** de sklearn → sin fuga de datos en validación cruzada.
- **Serialización `cloudpickle`** en `log_model` (evita el rechazo de `numpy.dtype` del backend skops).
- **Logger a color sin dependencias** (ANSI), que se desactiva solo si la salida no es TTY (CI).
