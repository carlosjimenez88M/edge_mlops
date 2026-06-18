"""Smoke test de la API: toma una fila real del test set y pide una prediccion.

Agnostico a la tarea: arma el payload segun project.task (clasificacion ->
vector de pixeles; regresion -> features nombradas). Usa solo la stdlib para
no depender de paquetes extra en el job de monitoreo.
"""

from __future__ import annotations

import json
import urllib.request


import pandas as pd

from src.common.config import load_config
from src.common.paths import PROJECT_ROOT

URL = "http://localhost:8000/predict"


def main() -> None:
    config = load_config()
    test_path = PROJECT_ROOT / config.preprocessing.processed_dir / "test.parquet"
    row = pd.read_parquet(test_path).drop(columns=[config.data.target]).iloc[0]

    if config.project.task == "classification":
        payload = {"pixels": [float(x) for x in row.tolist()]}
    else:
        payload = {k: float(v) for k, v in row.items()}

    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(URL, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as resp:  # noqa: S310 (URL local controlada)
        result = json.load(resp)
    print("Prediccion OK:", result)


if __name__ == "__main__":
    main()
