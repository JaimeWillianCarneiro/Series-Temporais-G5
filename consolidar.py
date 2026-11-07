#!/usr/bin/env python3
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
from pathlib import Path
import sys, subprocess, pandas as pd, numpy as np, joblib
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from ts_utils import carregar_dados, calcular_metricas

RAIZ = Path(__file__).resolve().parent
SERIES = ["store_total", "FOODS", "HOBBIES"]
LAGS = [1, 7, 14, 28]

# 1. Regenera baselines + sarima via run.py (sobrescreve raiz/previsoes e metricas)
subprocess.run([str(RAIZ / "venv" / "bin" / "python"), str(RAIZ / "run.py")], cwd=RAIZ, check=False)

treino, val = carregar_dados(RAIZ)
val["date"] = pd.to_datetime(val["date"])
treino_hist = pd.read_csv(RAIZ / "dados" / "treino.csv", parse_dates=["date"]).sort_values("date")

def prever_recursivo(modelo, y_hist, cov_fut):
    feats = list(modelo.feature_names_in_) if hasattr(modelo, "feature_names_in_") else None
    if feats is None:
        feats = list(modelo.feature_names_in_)
    base = cov_fut.reset_index(drop=True).reindex(columns=feats, fill_value=0).astype(float)
    ys = list(map(float, y_hist)); preds = []
    for i in range(len(base)):
        linha = base.iloc[[i]].copy()
        for l in LAGS:
            linha[f"lag_{l}"] = ys[-l]
        p = float(modelo.predict(linha[feats])[0])
        preds.append(p); ys.append(p)
    return np.array(preds)

# covariaveis preparadas
cov = pd.read_csv(RAIZ / "dados" / "covariaveis.csv", parse_dates=["date"])
cov["dia_semana"] = cov["date"].dt.dayofweek
cov = pd.get_dummies(cov, columns=["dia_semana", "event_name_1", "event_type_1", "event_name_2", "event_type_2"])
cov_val = cov[cov["date"].isin(val["date"])].sort_values("date")

prev = pd.read_csv(RAIZ / "previsoes_validacao.csv")
met = pd.read_csv(RAIZ / "metricas.csv")

# 2. Linear a partir dos .pkl
lista_lin = []
for serie in SERIES:
    pacote = joblib.load(RAIZ / f"modelo_{serie}.pkl")
    m = pacote["modelo"]; stats = pacote["stats_precos"]
    cov2 = cov.copy()
    for s in SERIES:
        cov2[f"preco_{s}_padronizada"] = (cov2[f"preco_{s}"] - stats[s][0]) / stats[s][1]
    cov_v = cov2[cov2["date"].isin(val["date"])].sort_values("date")
    yhat = prever_recursivo(m, treino_hist[serie].values, cov_v)
    for d, y in zip(cov_v["date"], yhat):
        lista_lin.append({"date": d.strftime("%Y-%m-%d"), "series": serie, "modelo": "linear", "yhat": y})
df_lin = pd.DataFrame(lista_lin)
prev = pd.concat([prev[prev["modelo"] != "linear"], df_lin], ignore_index=True)

# 3. Boosting a partir dos .pkl de árvore
lista_boost = []
for serie in SERIES:
    pacote = joblib.load(RAIZ / f"modelo_arvore_{serie}.pkl")
    m = pacote["modelo"]; stats = pacote["stats_precos"]
    cov2 = cov.copy()
    for s in SERIES:
        cov2[f"preco_{s}_padronizada"] = (cov2[f"preco_{s}"] - stats[s][0]) / stats[s][1]
    cov_v = cov2[cov2["date"].isin(val["date"])].sort_values("date")
    yhat = prever_recursivo(m, treino_hist[serie].values, cov_v)
    for d, y in zip(cov_v["date"], yhat):
        lista_boost.append({"date": d.strftime("%Y-%m-%d"), "series": serie, "modelo": "boosting", "yhat": y})
df_boost = pd.DataFrame(lista_boost)
prev = pd.concat([prev[~prev["modelo"].isin(["boosting", "arvore"])], df_boost], ignore_index=True)
prev.to_csv(RAIZ / "previsoes_validacao.csv", index=False)
prev.to_csv(RAIZ / "saidas" / "previsoes_validacao.csv", index=False)

# 4. Métricas para linear e boosting
for modelo in ["linear", "boosting"]:
    linhas = []
    for serie in SERIES:
        p = prev[(prev["series"] == serie) & (prev["modelo"] == modelo)].copy()
        p["date"] = pd.to_datetime(p["date"])
        j = val[["date", serie]].merge(p[["date", "yhat"]], on="date", how="left")
        mae, rmse, mase = calcular_metricas(treino[serie].values, j[serie].values, j["yhat"].values)
        linhas.append({"series": serie, "modelo": modelo, "mae": mae, "rmse": rmse, "mase": mase})
    dfm = pd.DataFrame(linhas)
    met = met[met["modelo"] != modelo]
    met = pd.concat([met, dfm], ignore_index=True)
met.to_csv(RAIZ / "metricas.csv", index=False)
met.to_csv(RAIZ / "saidas" / "metricas.csv", index=False)
print(met.to_string(index=False))
print("\nOK")
