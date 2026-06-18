# Preparar la Raspberry Pi

Guía para dejar la Pi lista como entorno de ejecución y como runner self-hosted de
GitHub Actions. Probado en Raspberry Pi OS (64-bit, `aarch64`).

## 1. Dependencias base

```bash
sudo apt update && sudo apt install -y git curl

# uv (gestor de entornos/paquetes)
curl -LsSf https://astral.sh/uv/install.sh | sh
exec $SHELL    # recarga PATH

# Docker + compose plugin
curl -fsSL https://get.docker.com | sh
sudo usermod -aG docker "$USER"     # re-loguéate para aplicar el grupo
docker compose version
```

## 2. Clonar y validar

```bash
git clone git@github.com:carlosjimenez88M/edge_mlops.git
cd edge_mlops
uv sync --all-groups
uv run pytest -q
uv run python orchestrator.py        # primera corrida completa
```

> En la Pi, `RandomForest` (200 árboles) + `GradientBoosting` son lo más lento.
> Si necesitas que la pipeline sea más liviana, reduce `competition.models` o
> `competition.cv_folds` en `config.yaml`.

## 3. Runner self-hosted de GitHub Actions

En GitHub: **Settings → Actions → Runners → New self-hosted runner** (Linux / ARM64) y
sigue las instrucciones. Al configurarlo, añade las **labels** `rpi` (el workflow usa
`runs-on: [self-hosted, rpi]`):

```bash
# dentro de la carpeta del runner descargado:
./config.sh --url https://github.com/carlosjimenez88M/edge_mlops \
            --token <TOKEN> --labels rpi
sudo ./svc.sh install     # instala el runner como servicio
sudo ./svc.sh start
```

Verifica que `uv` y `docker` estén en el PATH del servicio del runner (el runner hereda
el entorno del usuario que lo instaló).

## 4. Secret de Weights & Biases (para el sweep en CI)

En GitHub: **Settings → Secrets and variables → Actions → New repository secret**

- Nombre: `WANDB_API_KEY`
- Valor: tu API key de W&B

En local, en cambio, la key se lee de `.env` (`wandb_api_key=...`). **Nunca** comitees `.env`.

## 5. Puertos del stack

| Servicio   | Puerto | URL |
|------------|--------|-----|
| API        | 8000   | http://raspberrypi.local:8000/docs |
| Prometheus | 9090   | http://raspberrypi.local:9090 |
| Grafana    | 3000   | http://raspberrypi.local:3000 (admin/admin) |
| MLflow UI  | 5000   | `uv run mlflow ui --backend-store-uri sqlite:///mlflow.db` |

## Notas de borde

- Usa siempre imágenes/wheels `arm64`. `uv` resuelve wheels `aarch64` para numpy, scipy,
  scikit-learn, pandas y pyarrow (descarga, no compila).
- Vigila la temperatura/throttling en entrenamientos largos (`vcgencmd measure_temp`).
- El stack de Docker monta `mlflow.db`, `mlruns/` y `artifacts/`: corre la pipeline
  **antes** de levantar la API para que haya un modelo que servir.
