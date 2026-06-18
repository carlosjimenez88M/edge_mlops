# CI/CD en la Raspberry Pi

El workflow `.github/workflows/cicd.yml` materializa el ciclo como un **grafo de pasos**:
cada componente es un job independiente (una caja en el grafo de Actions), encadenados con
`needs:`.

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

- **Solo manual** (`workflow_dispatch`): se ejecuta cuando le das **"Run workflow"** en la
  pestaña **Actions** de GitHub. **No corre con push.**
- Al lanzarlo eliges la **rama** (por ejemplo `chapter-03`) y, con el input `run_sweep`,
  decides si se ejecuta el sweep de W&B en esa corrida.
- `concurrency` cancela corridas anteriores de la misma rama.

> Para que el botón "Run workflow" aparezca, el workflow debe existir en la rama por
> defecto (`master`); luego puedes ejecutarlo apuntando a cualquier rama.

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
