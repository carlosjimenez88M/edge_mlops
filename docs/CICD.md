# CI/CD en la Raspberry Pi

El workflow `.github/workflows/cicd.yml` materializa el ciclo:

```
push → test → train → deploy → monitor
```

Todo corre en el **runner self-hosted** de la Pi (`runs-on: [self-hosted, rpi]`), así que
el modelo se entrena y se sirve en el mismo hardware de borde.

## Disparadores

- `push` a `master`, `main` o cualquier rama `chapter-*`.
- `workflow_dispatch` (ejecución manual desde la pestaña Actions).
- `concurrency` cancela corridas anteriores de la misma rama.

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
