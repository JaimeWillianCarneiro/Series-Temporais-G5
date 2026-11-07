#!/usr/bin/env python3
"""Treina árvore/boosting (RandomForest, GradientBoosting, HistGradientBoosting) nas 3 séries,
escolhe o melhor por validação interna (corte 2016-02-28) e gera previsões + métricas."""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
import sys, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from ts_utils import carregar_dados, calcular_metricas
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error

RAIZ = Path(__file__).resolve().parent
SERIES = ["store_total", "FOODS", "HOBBIES"]
LAGS = [1, 7, 14, 28]

df_treino, df_val = carregar_dados(RAIZ)
df_cov = pd.read_csv(RAIZ / "dados" / "covariaveis.csv", parse_dates=["date"])
df_cov["dia_semana"] = df_cov["date"].dt.dayofweek
df_cov = pd.get_dummies(df_cov, columns=["dia_semana", "event_name_1", "event_type_1", "event_name_2", "event_type_2"])

def padronizar(df, mask):
    df = df.copy(); stats = {}
    for s in SERIES:
        m, d = df.loc[mask, f"preco_{s}"].mean(), df.loc[mask, f"preco_{s}"].std()
        stats[s] = (m, d)
        df[f"preco_{s}_padronizada"] = (df[f"preco_{s}"] - m) / d
    return df, stats

def montar_treino(df_y, cov, serie):
    d = df_y[["date", serie]].sort_values("date").copy()
    for l in LAGS:
        d[f"lag_{l}"] = d[serie].shift(l)
    d = d.dropna().merge(cov, on="date", how="inner")
    cols_drop = ["store_total", "FOODS", "HOBBIES", "date", "preco_store_total", "preco_FOODS", "preco_HOBBIES"]
    return d.drop(columns=cols_drop, errors="ignore"), d[serie]

def prever_recursivo(modelo, y_hist, cov_fut):
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

# validação interna
cov_int, _ = padronizar(df_cov, df_cov["date"] <= "2016-02-28")
cov_int_val = cov_int[(cov_int["date"] > "2016-02-28") & (cov_int["date"] <= "2016-03-27")]
hist_int = df_treino[df_treino["date"] <= "2016-02-28"]
val_int = df_treino[df_treino["date"] > "2016-02-28"]

candidatos = {
    "RandomForest": lambda: RandomForestRegressor(n_estimators=300, max_depth=8, random_state=42, n_jobs=1),
    "GradientBoosting": lambda: GradientBoostingRegressor(n_estimators=300, max_depth=3, learning_rate=0.05, random_state=42),
    "HistGradientBoosting": lambda: HistGradientBoostingRegressor(max_iter=300, max_depth=6, learning_rate=0.05, random_state=42),
}

melhor_por_serie = {}
for serie in SERIES:
    X_tr, y_tr = montar_treino(hist_int, cov_int, serie)
    y_vl = val_int.sort_values("date")[serie].values
    erros = {}
    for nome, ctor in candidatos.items():
        m = ctor().fit(X_tr, y_tr)
        yhat = prever_recursivo(m, hist_int.sort_values("date")[serie].values, cov_int_val)
        erros[nome] = mean_absolute_error(y_vl, yhat)
        print(f"{serie} | {nome}: MAE interno = {erros[nome]:.2f}")
    vencedor = min(erros, key=erros.get)
    melhor_por_serie[serie] = vencedor
    print(f"  -> vencedor {serie}: {vencedor}\n")

# treino definitivo
cov_final, stats_final = padronizar(df_cov, df_cov["date"] <= "2016-03-27")
modelos = {}
for serie in SERIES:
    X_t, y_t = montar_treino(df_treino, cov_final, serie)
    ctor = candidatos[melhor_por_serie[serie]]
    m = ctor().fit(X_t, y_t)
    modelos[serie] = (m, melhor_por_serie[serie])
    joblib.dump({"modelo": m, "nome": melhor_por_serie[serie], "lags": LAGS, "stats_precos": stats_final}, f"modelo_arvore_{serie}.pkl")
    print(f"salvo modelo_arvore_{serie}.pkl ({melhor_por_serie[serie]})")

# previsões validação
cov_val = cov_final[cov_final["date"].isin(df_val["date"])].sort_values("date")
lista = []
for serie in SERIES:
    m, nome = modelos[serie]
    nome_csv = "boosting" if "Gradient" in nome else "arvore"
    yhat = prever_recursivo(m, df_treino.sort_values("date")[serie].values, cov_val)
    for d, y in zip(cov_val["date"], yhat):
        lista.append({"date": d.strftime("%Y-%m-%d"), "series": serie, "modelo": nome_csv, "yhat": y})
df_prev_arvore = pd.DataFrame(lista)

# consolidar com saidas/previsoes_validacao.csv
arq_prev = RAIZ / "saidas" / "previsoes_validacao.csv"
if arq_prev.exists():
    antigo = pd.read_csv(arq_prev)
    antigo = antigo[~antigo["modelo"].isin(["boosting", "arvore"])]
    df_prev = pd.concat([antigo, df_prev_arvore], ignore_index=True)
else:
    df_prev = df_prev_arvore
df_prev.to_csv(arq_prev, index=False)
# também na raiz
df_prev.to_csv(RAIZ / "previsoes_validacao.csv", index=False)
print("\nprevisoes salvas")

# métricas
lista_m = []
for serie in SERIES:
    p = df_prev[(df_prev["series"] == serie) & (df_prev["modelo"].isin(["boosting", "arvore"]))].copy()
    p["date"] = pd.to_datetime(p["date"])
    j = df_val[["date", serie]].merge(p[["date", "yhat"]], on="date", how="left")
    mae, rmse, mase = calcular_metricas(df_treino[serie].values, j[serie].values, j["yhat"].values)
    nome_csv = p["modelo"].iloc[0]
    lista_m.append({"series": serie, "modelo": nome_csv, "mae": mae, "rmse": rmse, "mase": mase})
df_m_arvore = pd.DataFrame(lista_m)

arq_met = RAIZ / "saidas" / "metricas.csv"
if arq_met.exists():
    antigo = pd.read_csv(arq_met)
    antigo = antigo[~antigo["modelo"].isin(["boosting", "arvore"])]
    df_met = pd.concat([antigo, df_m_arvore], ignore_index=True)
else:
    df_met = df_m_arvore
df_met.to_csv(arq_met, index=False)
df_met.to_csv(RAIZ / "metricas.csv", index=False)
print(df_met.to_string(index=False))
print("\nOK concluído")
