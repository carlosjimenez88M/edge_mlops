# CI/CD en la Raspberry Pi

El workflow `.github/workflows/cicd.yml` materializa el ciclo:

```
Run workflow (manual) → test → train → deploy → monitor
```

Todo corre en el **runner self-hosted** de la Pi (`runs-on: [self-hosted, rpi]`), así que
el modelo se entrena y se sirve en el mismo hardware de borde.

## Disparadores

- **Solo manual** (`workflow_dispatch`): se ejecuta cuando le das **"Run workflow"** en la
  pestaña **Actions** de GitHub. **No corre con push.**
- Al lanzarlo eliges la **rama** (por ejemplo `chapter-03`) y, con el input `run_sweep`,
  decides si se ejecuta el sweep de W&B en esa corrida.
- `concurrency` cancela corridas anteriores de la misma rama.

> Para que el botón "Run workflow" aparezca, el workflow debe existir en la rama por
> defecto (`master`); luego puedes ejecutarlo apuntando a cualquier rama.

## Jobs

| Job       | Depende de | Qué hace |
|-----------|------------|----------|
| `test`    | —          | `uv sync` · `ruff check` · `pytest` |
| `train`   | `test`     | corre `orchestrator.py` (pipeline completa) y sube `artifacts/` + `mlflow.db` |
| `deploy`  | `train`    | `docker compose up -d --build` y espera a que `/health` responda |
| `monitor` | `deploy`   | smoke test de `/predict`, verifica `/metrics` y la salud de Grafana |

El **sweep** de W&B se ejecuta dentro de `train` solo si `sweep.enabled: true` en
`config.yaml` (y existe el secret `WANDB_API_KEY`). Por defecto está apagado para que
el CI sea rápido y determinista.

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
