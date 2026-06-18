# edge_mlops 🍓

**Plataforma de prácticas MLOps de borde sobre Raspberry Pi**, construida capítulo
a capítulo siguiendo *Hands-On Machine Learning with Scikit-Learn & PyTorch*
(Aurélien Géron).

El objetivo es vivir el ciclo completo en hardware real:

```
Escribo código → hago push → se prueba → se construye → se despliega en la Raspberry → se monitorea
```

Cada capítulo se implementa como una **pipeline modular de componentes** (arquitectura
*component–artifact*): cada pieza es un *MLflow Project* independiente, con su propio
`MLproject` y dependencias gestionadas con `uv`, que produce **artefactos** consumidos
por el siguiente paso. Un **orquestador** encadena los pasos según `config.yaml`.

> **Capítulo actual: 02 — Proyecto end-to-end (California Housing, regresión).**
> Resultado de referencia: `RandomForest` gana la competencia con **R² ≈ 0.81 / RMSE ≈ 0.50** en test.

---

## 🏗️ Arquitectura

```
                         config.yaml  (Pydantic, única fuente de verdad)
                              │
                              ▼
                       orchestrator.py  ── ejecuta cada componente como MLflow Project
                              │
   ┌──────────┬──────────────┼───────────────┬──────────────┬───────────┐
   ▼          ▼              ▼                ▼              ▼           ▼
data_load  data_validation  data_preprocessing  model_competition  sweep   register
 (raw)      (EDA + drift)     (split + features)   (elige el mejor)  (W&B)  (MLflow Registry)
   │          │                │                   │               │         │
   └──────────┴────────────────┴───── artefactos ──┴───────────────┴─────────┘
                                          │
                                          ▼
                                  src/api (FastAPI + Docker)
                                          │
                              /predict  /health  /metrics
                                          │
                        Prometheus  ──scrape──►  Grafana (dashboards)
```

Todo está escrito con **type hints**, validado con **Pydantic**, registrado con un
**logger a color** (🟢 ok · 🟡 warning · 🔴 fallo) y probado con **pytest**.

---

## 📁 Estructura

```
edge_mlops/
├── config.yaml               # configuración de la corrida (validada por Pydantic)
├── orchestrator.py           # encadena los componentes según config.yaml
├── pyproject.toml            # entorno raíz (uv), un grupo de deps por componente
├── Dockerfile                # imagen de la API
├── docker-compose.yml        # api + prometheus + grafana
├── src/
│   ├── common/               # logger a color, config Pydantic, rutas, utils MLflow
│   ├── data_load/            # descarga y persiste el dataset crudo
│   ├── data_validation/      # EDA + detección de data drift (KS)
│   ├── data_preprocessing/   # feature engineering + split + preprocesador
│   ├── model_competition/    # entrena varios modelos y elige el campeón
│   ├── sweep/                # W&B sweep (60 intentos) sobre el modelo ganador
│   ├── register/             # registro + promoción (champion-challenger)
│   └── api/                  # FastAPI + métricas Prometheus
├── monitoring/               # prometheus.yml + dashboards/datasource de Grafana
├── tests/                    # pruebas unitarias y de la API
├── docs/                     # guías (arquitectura, Pi, CI/CD, nuevos capítulos)
└── .github/workflows/        # CI/CD para el runner self-hosted en la Pi
```

Cada `src/<componente>/` contiene: `MLproject`, `pyproject.toml`, `main.py`
(punto de entrada) y módulos con las funciones.

---

## 🚀 Quickstart

Requisitos: Python ≥ 3.11, [`uv`](https://docs.astral.sh/uv/), Docker (opcional, para servir/monitorear).

```bash
# 1. Entorno (un solo venv raíz con todas las dependencias)
uv sync --all-groups

# 2. (opcional) W&B sweep: copia el ejemplo y pon tu API key
cp .env.example .env        # edita wandb_api_key=...

# 3. Ejecuta la pipeline completa (data → validación → competencia → registro)
uv run python orchestrator.py

# 4. Inspecciona los experimentos
uv run mlflow ui --backend-store-uri sqlite:///mlflow.db   # http://localhost:5000

# 5. Sirve el modelo + monitoreo
docker compose up -d --build
#   API        → http://localhost:8000/docs
#   Prometheus → http://localhost:9090
#   Grafana    → http://localhost:3000  (admin/admin)
```

Probar la API:

```bash
curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d '{
  "MedInc": 8.3, "HouseAge": 41, "AveRooms": 6.98, "AveBedrms": 1.02,
  "Population": 322, "AveOccup": 2.55, "Latitude": 37.88, "Longitude": -122.23
}'
```

### Ejecutar pasos sueltos

```bash
uv run python orchestrator.py --only data_load data_validation
uv run python orchestrator.py --from model_competition

# Un componente AISLADO con su propio venv uv (component-artifact puro):
cd src/data_load && uv run python main.py --config ../../config.yaml
```

---

## 🧩 Componentes

| Componente          | Entrada                | Salida (artefactos)                              | Qué hace |
|---------------------|------------------------|--------------------------------------------------|----------|
| `data_load`         | —                      | `data/raw/housing.parquet`                       | Descarga California Housing y lo registra en MLflow |
| `data_validation`   | dataset crudo          | `validation_report.json`, `histograms.png`       | EDA + drift (test KS, semáforo de warnings) |
| `data_preprocessing`| dataset crudo          | `train.parquet`, `test.parquet`, `feature_names` | Feature engineering + split |
| `model_competition` | splits                 | `best_model.json`, `leaderboard.json`, modelo    | Entrena N modelos, valida (CV + holdout), elige el campeón |
| `sweep`             | `best_model.json`      | `tuned_model`, `best_params`                      | W&B sweep (60 intentos) sobre la familia ganadora |
| `register`          | mejor modelo           | versión en MLflow Registry                        | Gate de calidad + promoción champion-challenger |
| `api`               | modelo registrado      | servicio HTTP                                     | FastAPI `/predict` `/health` `/metrics` |

---

## 🏆 Competencia de modelos

`model_competition` entrena cada modelo de `config.competition.models` con el **mismo
preprocesamiento dentro del Pipeline** (sin fuga de datos), los compara por validación
cruzada y holdout, y selecciona el campeón según `config.competition.metric`. Cada
candidato queda como *run anidado* en MLflow y el ganador se persiste en `best_model.json`.

## 🎛️ Sweep de Weights & Biases

Cuando `sweep.enabled: true`, el componente `sweep` lee el modelo ganador y lanza un
**sweep bayesiano de exactamente 60 intentos** (`sweep.count`) para afinar sus
hiperparámetros, minimizando `val_rmse`. Reentrena con la mejor combinación, lo registra
en MLflow y actualiza `best_model.json` para que `register` use el modelo afinado.
La API key se lee de `.env` (`wandb_api_key`). Espacios de búsqueda en
`src/sweep/search_space.py`.

## 📈 Monitoreo

La API expone métricas Prometheus en `/metrics` (`edge_predictions_total`,
`edge_prediction_errors_total`, `edge_prediction_latency_seconds`,
`edge_last_prediction_value`). Prometheus las recolecta y Grafana las grafica con un
dashboard ya provisionado (*Inference Monitoring*).

---

## 🔁 CI/CD en la Raspberry Pi

El workflow `.github/workflows/cicd.yml` corre en un **runner self-hosted** en la Pi y
ejecuta el ciclo completo en cada push: **test → train (pipeline) → build → deploy → monitor**.
Configuración del runner en [`docs/RASPBERRY_PI_SETUP.md`](docs/RASPBERRY_PI_SETUP.md).

## 📚 Trabajar todos los capítulos

- `master` → **plataforma compartida** (logger, monitoreo, CI, orquestador, docs) + Cap. 2 de referencia.
- `chapter-XX` → una rama por capítulo que solo añade/ajusta componentes en `src/` y su `config.yaml`.

Haces `push` de tu rama → el Action corre en la Pi → ves la arquitectura component-artifact
funcionando de punta a punta. Guía paso a paso en [`docs/ADD_A_CHAPTER.md`](docs/ADD_A_CHAPTER.md).

---

## 📖 Documentación

| Documento | Contenido |
|-----------|-----------|
| [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md)          | Diseño, flujo de artefactos y decisiones |
| [`docs/ADD_A_CHAPTER.md`](docs/ADD_A_CHAPTER.md)        | Cómo continuar con otros capítulos por tu cuenta |
| [`docs/RASPBERRY_PI_SETUP.md`](docs/RASPBERRY_PI_SETUP.md) | Runner self-hosted, uv y Docker en la Pi |
| [`docs/CICD.md`](docs/CICD.md)                          | El pipeline de CI/CD explicado |

## 🧪 Tests

```bash
uv run pytest -q
uv run ruff check src orchestrator.py
```

## 📝 Licencia

MIT.
