#!/usr/bin/env python3
"""Task 2 — linear (Lasso) e árvore (boosting), recursivos, a partir de dados/.

Roda depois de `run.py` (Task 1), que grava as linhas de baselines e SARIMA em
metricas.csv / previsoes_validacao.csv. Este script acrescenta `linear` e `boosting`
sem tocar nas linhas da Task 1 e copia as duas tabelas para saidas/.

Mesma lógica de analise_task2.ipynb:
  - features: lags 1, 7, 14, 28 do próprio y; dia da semana; snap_CA; dummies de
    event_name_1/2 e event_type_1/2; os três preços semanais padronizados com
    média/desvio do treino (na validação interna, do treino interno);
  - estratégia recursiva: a previsão de t+1 volta como lag; o y da validação nunca entra;
  - hiperparâmetros: alpha do Lasso e escolha entre RandomForest / GradientBoosting /
    HistGradientBoosting por MAE numa validação interna com origem 2016-02-28
    (prevê 2016-02-29..2016-03-27, dentro do treino). Os 28 dias de validação só publicam.
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_absolute_error

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from ts_utils import RAIZ, SERIES, calcular_metricas, carregar_dados  # noqa: E402

ORIGEM = pd.Timestamp("2016-03-27")
ORIGEM_INTERNA = pd.Timestamp("2016-02-28")
LAGS = [1, 7, 14, 28]
ALPHAS = [0.001, 0.01, 0.1, 1, 10, 100]
COLS_NAO_FEATURE = ["store_total", "FOODS", "HOBBIES", "date", "preco_store_total", "preco_FOODS", "preco_HOBBIES"]
CANDIDATOS_ARVORE = {
    "RandomForest": lambda: RandomForestRegressor(n_estimators=300, max_depth=8, random_state=42, n_jobs=1),
    "GradientBoosting": lambda: GradientBoostingRegressor(
        n_estimators=300, max_depth=3, learning_rate=0.05, random_state=42
    ),
    "HistGradientBoosting": lambda: HistGradientBoostingRegressor(
        max_iter=300, max_depth=6, learning_rate=0.05, random_state=42
    ),
}


def carregar_covariaveis():
    cov = pd.read_csv(RAIZ / "dados" / "covariaveis.csv", parse_dates=["date"])
    cov["dia_semana"] = cov["date"].dt.dayofweek
    return pd.get_dummies(cov, columns=["dia_semana", "event_name_1", "event_type_1", "event_name_2", "event_type_2"])


def padronizar_precos(cov, ate):
    """Padroniza preco_<serie> com média/desvio só das datas <= ate."""
    cov = cov.copy()
    stats = {}
    ref = cov["date"] <= ate
    for s in SERIES:
        m, d = cov.loc[ref, f"preco_{s}"].mean(), cov.loc[ref, f"preco_{s}"].std()
        stats[s] = (m, d)
        cov[f"preco_{s}_padronizada"] = (cov[f"preco_{s}"] - m) / d
    return cov, stats


def montar_treino(df_y, cov, serie):
    """Lags reais de y (só no ajuste) + covariáveis do dia."""
    d = df_y[["date", serie]].sort_values("date").copy()
    for lag in LAGS:
        d[f"lag_{lag}"] = d[serie].shift(lag)
    d = d.dropna().merge(cov, on="date", how="inner")
    return d.drop(columns=COLS_NAO_FEATURE, errors="ignore"), d[serie]


def prever_recursivo(modelo, y_hist, cov_fut):
    """A previsão de t+1 entra como lag nas seguintes; só o histórico do treino é real."""
    feats = list(modelo.feature_names_in_)
    base = cov_fut.reset_index(drop=True).reindex(columns=feats, fill_value=0).astype(float)
    ys = list(map(float, y_hist))
    preds = []
    for i in range(len(base)):
        linha = base.iloc[[i]].copy()
        for lag in LAGS:
            linha[f"lag_{lag}"] = ys[-lag]
        p = float(modelo.predict(linha[feats])[0])
        preds.append(p)
        ys.append(p)
    return np.array(preds)


def main():
    treino, val = carregar_dados()
    treino = treino.sort_values("date").reset_index(drop=True)
    assert treino["date"].max() == ORIGEM and len(val) == 28
    cov = carregar_covariaveis()

    # validação interna (dentro do treino)
    cov_int, _ = padronizar_precos(cov, ORIGEM_INTERNA)
    hist_int = treino[treino["date"] <= ORIGEM_INTERNA]
    alvo_int = treino[treino["date"] > ORIGEM_INTERNA]
    cov_int_fut = cov_int[cov_int["date"].isin(alvo_int["date"])].sort_values("date")

    cov_fin, stats = padronizar_precos(cov, ORIGEM)
    cov_val = cov_fin[cov_fin["date"].isin(val["date"])].sort_values("date")
    assert (cov_val["date"].to_numpy() == val["date"].to_numpy()).all()

    escolhas, linhas = [], []
    for s in SERIES:
        X_int, y_int = montar_treino(hist_int, cov_int, s)
        y_alvo = alvo_int[s].to_numpy()
        h_int = hist_int[s].to_numpy()

        mae_alpha = {
            a: mean_absolute_error(
                y_alvo, prever_recursivo(Lasso(alpha=a, max_iter=100000).fit(X_int, y_int), h_int, cov_int_fut)
            )
            for a in ALPHAS
        }
        alpha = min(mae_alpha, key=mae_alpha.get)
        mae_arv = {
            n: mean_absolute_error(y_alvo, prever_recursivo(c().fit(X_int, y_int), h_int, cov_int_fut))
            for n, c in CANDIDATOS_ARVORE.items()
        }
        arvore = min(mae_arv, key=mae_arv.get)
        escolhas.append({"series": s, "alpha_lasso": alpha, "arvore": arvore,
                         **{f"mae_int_{n}": v for n, v in mae_arv.items()}})
        print(f"{s}: alpha={alpha} | árvore={arvore} | MAE interno árvores "
              f"{ {n: round(v, 1) for n, v in mae_arv.items()} }")

        # ajuste definitivo em todo o treino e previsão recursiva da validação
        X, y = montar_treino(treino, cov_fin, s)
        modelos = {
            "linear": Lasso(alpha=alpha, max_iter=100000).fit(X, y),
            "boosting": CANDIDATOS_ARVORE[arvore]().fit(X, y),
        }
        joblib.dump({"modelo": modelos["linear"], "alpha": alpha, "lags": LAGS, "stats_precos": stats},
                    RAIZ / f"modelo_{s}.pkl")
        joblib.dump({"modelo": modelos["boosting"], "nome": arvore, "lags": LAGS, "stats_precos": stats},
                    RAIZ / f"modelo_arvore_{s}.pkl")
        for nome, m in modelos.items():
            yhat = prever_recursivo(m, treino[s].to_numpy(), cov_val)
            linhas += [{"date": d.strftime("%Y-%m-%d"), "series": s, "modelo": nome, "yhat": v}
                       for d, v in zip(cov_val["date"], yhat)]

    novos = pd.DataFrame(linhas)
    (RAIZ / "saidas").mkdir(exist_ok=True)
    pd.DataFrame(escolhas).to_csv(RAIZ / "saidas" / "escolhas_task2.csv", index=False)

    # acrescenta às tabelas da Task 1 (gravadas por run.py) sem mexer nas linhas dela
    prev = pd.read_csv(RAIZ / "previsoes_validacao.csv")
    prev = prev[~prev["modelo"].isin(novos["modelo"].unique())]
    assert {"naive_sazonal", "sarima"} <= set(prev["modelo"]), "rode python run.py antes"
    prev = pd.concat([prev, novos], ignore_index=True)

    met = pd.read_csv(RAIZ / "metricas.csv")
    met = met[~met["modelo"].isin(novos["modelo"].unique())]
    novas_met = []
    for s in SERIES:
        for nome in novos["modelo"].unique():
            p = novos[(novos.series == s) & (novos.modelo == nome)]
            j = val[["date", s]].assign(date=val["date"].dt.strftime("%Y-%m-%d")).merge(p, on="date")
            assert len(j) == 28
            mae, rmse, mase = calcular_metricas(treino[s].to_numpy(), j[s].to_numpy(), j["yhat"].to_numpy())
            novas_met.append({"series": s, "modelo": nome, "mae": mae, "rmse": rmse, "mase": mase})
    met = pd.concat([met, pd.DataFrame(novas_met)], ignore_index=True)

    for pasta in (RAIZ, RAIZ / "saidas"):
        prev.to_csv(pasta / "previsoes_validacao.csv", index=False)
        met.to_csv(pasta / "metricas.csv", index=False)
    print(met.round(3).to_string(index=False))
    print("\nConcluído: metricas.csv, previsoes_validacao.csv (raiz e saidas/), saidas/escolhas_task2.csv")


if __name__ == "__main__":
    main()
