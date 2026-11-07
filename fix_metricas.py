#!/usr/bin/env python3
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
from pathlib import Path
import pandas as pd
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from ts_utils import carregar_dados, calcular_metricas

RAIZ = Path(__file__).resolve().parent
treino, val = carregar_dados(RAIZ)
prev = pd.read_csv(RAIZ / "previsoes_validacao.csv", parse_dates=["date"])
val = val.copy(); val["date"] = pd.to_datetime(val["date"])

lista = []
for serie in ["store_total", "FOODS", "HOBBIES"]:
    for modelo in ["boosting", "arvore"]:
        p = prev[(prev["series"] == serie) & (prev["modelo"] == modelo)]
        if p.empty:
            continue
        j = val[["date", serie]].merge(p[["date", "yhat"]], on="date", how="left")
        mae, rmse, mase = calcular_metricas(treino[serie].values, j[serie].values, j["yhat"].values)
        lista.append({"series": serie, "modelo": modelo, "mae": mae, "rmse": rmse, "mase": mase})

df_new = pd.DataFrame(lista)
met = pd.read_csv(RAIZ / "metricas.csv")
met = met[~met["modelo"].isin(["boosting", "arvore"])]
met = pd.concat([met, df_new], ignore_index=True)
met.to_csv(RAIZ / "metricas.csv", index=False)
# também saidas/
met.to_csv(RAIZ / "saidas" / "metricas.csv", index=False)
print(met.to_string(index=False))
print("\nOK")
