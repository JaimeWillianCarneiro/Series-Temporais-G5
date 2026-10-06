#!/usr/bin/env python3
"""Task 1 — diagnóstico, baselines e SARIMA.

Gera, a partir de dados/:
  - figuras/            série + ACF/PACF no treino (e das séries diferenciadas), previsões
  - previsoes_validacao.csv
  - metricas.csv
  - saidas/ordens_sarima.csv e saidas/diagnostico_testes.csv
Tudo é ajustado só no treino (até 2016-03-27); a validação serve apenas para avaliar.
"""
import os

# um ajuste por processo: evita sobre-assinatura de threads do BLAS no grid paralelo
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import itertools
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tools.sm_exceptions import ConvergenceWarning, EstimationWarning
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller, kpss

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from ts_utils import BASELINES, M, RAIZ, SERIES, calcular_baselines, calcular_metricas, carregar_dados  # noqa: E402

FIG = RAIZ / "figuras"
SAIDAS = RAIZ / "saidas"  # tabelas auxiliares (metricas.csv e previsoes_validacao.csv ficam na raiz)
LAGS = 3 * M + 7  # 28 lags: 4 ciclos semanais

# Grade guiada pelo diagnóstico: D=1 (sazonalidade semanal forte no ACF), d in {0,1}.
GRADE = list(itertools.product(range(3), (0, 1), range(3), range(2), (1,), range(2)))


def diagnostico(treino):
    """Figuras de série + ACF/PACF no treino, e ACF/PACF após diff(1) e diff sazonal(7)."""
    FIG.mkdir(exist_ok=True)
    linhas = []
    for s in SERIES:
        y = treino.set_index("date")[s]
        fig, ax = plt.subplots(3, 1, figsize=(11, 9))
        ax[0].plot(y.index, y.values, lw=0.6)
        ax[0].plot(y.index, y.rolling(365, center=True).mean(), color="red", lw=1.5, label="média móvel 365d")
        ax[0].set(title=f"{s} — treino (2011-01-29 a 2016-03-27)", ylabel="vendas")
        ax[0].legend()
        plot_acf(y.values, lags=LAGS, ax=ax[1], title=f"ACF — {s}")
        plot_pacf(y.values, lags=LAGS, ax=ax[2], method="ywm", title=f"PACF — {s}")
        fig.tight_layout()
        fig.savefig(FIG / f"diagnostico_{s}.png", dpi=110)
        plt.close(fig)

        dif = y.diff().diff(M).dropna()
        fig, ax = plt.subplots(2, 1, figsize=(11, 6))
        plot_acf(dif.values, lags=LAGS, ax=ax[0], title=f"ACF — {s} após diff(1) e diff sazonal(7)")
        plot_pacf(dif.values, lags=LAGS, ax=ax[1], method="ywm", title=f"PACF — {s} após diff(1) e diff sazonal(7)")
        fig.tight_layout()
        fig.savefig(FIG / f"acf_pacf_diferenciada_{s}.png", dpi=110)
        plt.close(fig)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # kpss avisa quando p-valor sai da tabela
            adf = adfuller(y.values, autolag="AIC")[1]
            adf_sd = adfuller(y.diff(M).dropna().values, autolag="AIC")[1]
            kp = kpss(y.values, regression="c", nlags="auto")[1]
        linhas.append({
            "series": s,
            "zeros": int((y == 0).sum()),
            "adf_p_nivel": adf,
            "kpss_p_nivel": kp,
            "adf_p_diff_sazonal": adf_sd,
        })
    return pd.DataFrame(linhas)


def _ajusta(args):
    y, ordem, sazonal = args
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            warnings.simplefilter("ignore", EstimationWarning)  # só indica start_params zerados
            r = SARIMAX(y, order=ordem, seasonal_order=sazonal).fit(disp=False, maxiter=200)
        return ordem, sazonal, r.aic, r.bic, bool(r.mle_retvals.get("converged", True))
    except (np.linalg.LinAlgError, ValueError) as e:
        print(f"    falhou {ordem}{sazonal}: {e}")
        return None


def identificar_sarima(y, nome):
    """Escolhe a ordem por menor AIC dentro de GRADE (ajuste só no treino)."""
    tarefas = [(y, (p, d, q), (P, D, Q, M)) for p, d, q, P, D, Q in GRADE]
    with ProcessPoolExecutor() as ex:
        res = [r for r in ex.map(_ajusta, tarefas) if r is not None and r[4]]
    res.sort(key=lambda r: r[2])
    print(f"  {nome}: {len(res)}/{len(GRADE)} ajustes convergiram. Top 3 por AIC:")
    for o, s, aic, _, _ in res[:3]:
        print(f"    SARIMA{o}{s}  AIC={aic:.2f}")
    return res[0]


def prever_sarima(y, ordem, sazonal, h):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", EstimationWarning)
        r = SARIMAX(y, order=ordem, seasonal_order=sazonal).fit(disp=False, maxiter=200)
    return np.asarray(r.get_forecast(steps=h).predicted_mean)


def grafico_previsoes(treino, val, prev):
    fig, ax = plt.subplots(3, 1, figsize=(11, 10))
    for a, s in zip(ax, SERIES):
        ult = treino.tail(56)
        a.plot(ult["date"], ult[s], color="gray", label="treino (últimos 56d)")
        a.plot(val["date"], val[s], color="black", label="validação (real)")
        for m in ["naive_sazonal", "sarima"]:
            d = prev[(prev.series == s) & (prev.modelo == m)]
            a.plot(d["date"], d["yhat"], label=m)
        a.set_title(s)
        a.legend(ncol=4, fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "previsoes_validacao.png", dpi=110)
    plt.close(fig)


def main():
    treino, val = carregar_dados()
    h = len(val)
    assert treino["date"].max() == pd.Timestamp("2016-03-27"), "treino deve acabar em 2016-03-27"
    assert val["date"].min() == pd.Timestamp("2016-03-28") and h == 28
    print(f"Treino: {len(treino)} dias | Validação: {h} dias")

    print("\n1. Diagnóstico (figuras em figuras/)")
    testes = diagnostico(treino)
    print(testes.to_string(index=False))
    SAIDAS.mkdir(exist_ok=True)
    testes.to_csv(SAIDAS / "diagnostico_testes.csv", index=False)

    print("\n2. Baselines e SARIMA")
    linhas, ordens = [], []
    for s in SERIES:
        y = treino[s].to_numpy(dtype=float)
        previsoes = calcular_baselines(y, h=h, m=M)
        ordem, sazonal, aic, bic, _ = identificar_sarima(y, s)
        previsoes["sarima"] = prever_sarima(y, ordem, sazonal, h)
        ordens.append({"series": s, "order": ordem, "seasonal_order": sazonal, "aic": aic, "bic": bic})
        for modelo, yhat in previsoes.items():
            linhas += [{"date": d, "series": s, "modelo": modelo, "yhat": v} for d, v in zip(val["date"], yhat)]
    prev = pd.DataFrame(linhas)
    prev.to_csv(RAIZ / "previsoes_validacao.csv", index=False)
    pd.DataFrame(ordens).to_csv(SAIDAS / "ordens_sarima.csv", index=False)

    print("\n3. Métricas (só na validação)")
    met = []
    for s in SERIES:
        y_tr, y_va = treino[s].to_numpy(dtype=float), val[s].to_numpy(dtype=float)
        for modelo in BASELINES + ["sarima"]:
            yhat = prev[(prev.series == s) & (prev.modelo == modelo)].sort_values("date")["yhat"].to_numpy()
            mae, rmse, mase = calcular_metricas(y_tr, y_va, yhat)
            met.append({"series": s, "modelo": modelo, "mae": mae, "rmse": rmse, "mase": mase})
    met = pd.DataFrame(met)
    met.to_csv(RAIZ / "metricas.csv", index=False)
    print(met.round(3).to_string(index=False))

    grafico_previsoes(treino, val, prev)
    print("\nOrdens SARIMA:")
    print(pd.DataFrame(ordens).to_string(index=False))
    print("\nConcluído: metricas.csv, previsoes_validacao.csv, saidas/, figuras/")


if __name__ == "__main__":
    main()
