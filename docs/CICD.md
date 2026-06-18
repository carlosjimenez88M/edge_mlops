# CI/CD en la Raspberry Pi

Hay **un workflow por capítulo** en la pestaña Actions (entradas separadas), pero **una sola
definición** del pipeline para no duplicar nada:

- `.github/workflows/chapter-02.yml` → "Capítulo 2 · MLOps (California Housing)"
- `.github/workflows/chapter-03.yml` → "Capítulo 3 · MLOps (MNIST)"
- `.github/workflows/_pipeline.yml` → pipeline **reutilizable** (`workflow_call`) con las 9 cajas.

Cada workflow de capítulo es fino: invoca al pipeline reutilizable pasándole su rama
(`chapter_ref`). El pipeline hace `checkout` de esa rama, así corre el código y el
`config.yaml` del capítulo. Para añadir el capítulo N, copia un `chapter-0X.yml` y cambia
`chapter_ref` (ver `docs/ADD_A_CHAPTER.md`).

El ciclo es un **grafo de pasos**: cada componente es un job independiente (una caja),
encadenados con `needs:`.

```
test → data_load → data_validation → data_preprocessing →
       model_competition → sweep → register → build_deploy → monitor
```

Todo corre en el **runner self-hosted** de la Pi (`runs-on: [self-hosted, rpi]`), así que
el modelo se entrena y se sirve en el mismo hardware de borde.

> **Por qué se ven como cajas separadas:** GitHub dibuja un nodo por job y los conecta
> según `needs:`. Como cada paso es su propio job, ves el pipeline avanzar caja por caja,
> desde "1 · Cargar datos" hasta "8 · Monitorear".

> **Cómo comparten datos entre cajas:** están pensadas para **un único runner en la Pi**.
> Cada job hace `checkout` con `clean: false`, de modo que los artefactos generados
> (`data/`, `artifacts/`, `mlflow.db`, `mlruns/`) permanecen en el workspace del runner y
> el siguiente paso los encuentra — sin subir/bajar gigas por la red. Con varios runners
> habría que pasar el estado con `upload-artifact`/`download-artifact`.

## Disparadores

- **Solo manual** (`workflow_dispatch`): en la pestaña **Actions** eliges el workflow del
  capítulo ("Capítulo 2…" o "Capítulo 3…") y pulsas **"Run workflow"**. **No corre con push.**
- El input `run_sweep` decide si se ejecuta el sweep de W&B en esa corrida.
- No necesitas elegir rama: cada workflow ya apunta a la suya (`chapter_ref`).

> Los tres archivos de workflow viven en la rama por defecto (`master`) para que aparezcan
> en el sidebar y sean ejecutables. El reutilizable (`_pipeline.yml`) no tiene botón propio
> (solo `workflow_call`).

## Jobs (una caja cada uno)

| Job                  | Depende de           | Qué hace |
|----------------------|----------------------|----------|
| `test`               | —                    | `uv sync` · `ruff check` · `pytest` |
| `data_load`          | `test`               | `orchestrator.py --only data_load` (descarga dataset) |
| `data_validation`    | `data_load`          | EDA + drift; publica `validation_report.json` |
| `data_preprocessing` | `data_validation`    | feature engineering + split |
| `model_competition`  | `data_preprocessing` | entrena, compara y elige el campeón; publica `leaderboard.json` |
| `sweep`              | `model_competition`  | sweep de W&B (si `run_sweep`); si no, se omite solo |
| `register`           | `sweep`              | gate de calidad + registro + promoción en MLflow |
| `build_deploy`       | `register`           | `docker compose up -d --build` y espera a `/health` |
| `monitor`            | `build_deploy`       | smoke `/predict`, `/metrics`, scrape de Prometheus y salud de Grafana |

El job `sweep` **siempre aparece** en el grafo: corre el componente, que se omite a sí
mismo (caja verde, sin trabajo) salvo que marques el input **`run_sweep`** al lanzar el
workflow (necesita el secret `WANDB_API_KEY`).

## Probar el flujo completo

```bash
git checkout -b chapter-02-experimento
# ... cambios ...
git push -u origin chapter-02-experimento
```

Abre la pestaña **Actions** del repo y observa cómo los cuatro jobs pasan en cadena
sobre la Pi. Cuando `monitor` queda en verde, tu modelo está entrenado, registrado,
desplegado y monitoreado.

## Depuración

- **El runner no toma el job** → revisa que el servicio esté `active` (`sudo ./svc.sh status`)
  y que tenga la label `rpi`.
- **`uv`/`docker` no encontrado** → el servicio del runner no tiene el PATH correcto;
  reinstálalo desde un shell donde `uv` y `docker` funcionen.
- **La API no levanta** → el job `deploy` imprime `docker compose logs api`. Causa típica:
  no hay modelo registrado (corre la pipeline antes) o falta montar `mlflow.db`.
- **Sweep falla** → falta el secret `WANDB_API_KEY` o `sweep.enabled` está en `true` sin red.
