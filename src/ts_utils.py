"""Funções compartilhadas: dados, baselines e métricas (usadas por run.py e pelos notebooks)."""
from pathlib import Path

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parent.parent
SERIES = ["store_total", "FOODS", "HOBBIES"]
BASELINES = ["media", "naive", "naive_sazonal", "drift"]
M = 7


def carregar_dados(raiz=RAIZ):
    treino = pd.read_csv(raiz / "dados" / "treino.csv", parse_dates=["date"])
    val = pd.read_csv(raiz / "dados" / "validacao.csv", parse_dates=["date"])
    return treino, val


def calcular_baselines(y_train, h=28, m=M):
    """Quatro baselines da aula: média, naive, naive sazonal (período m) e drift."""
    y_train = np.asarray(y_train, dtype=float)
    T = len(y_train)
    y_last = y_train[-1]
    drift = (y_last - y_train[0]) / (T - 1)
    return {
        "media": np.full(h, y_train.mean()),
        "naive": np.full(h, y_last),
        # repete o último ciclo sazonal, cobrindo h mesmo se h não for múltiplo de m
        "naive_sazonal": np.tile(y_train[-m:], int(np.ceil(h / m)))[:h],
        "drift": y_last + np.arange(1, h + 1) * drift,
    }


def escala_mase(y_train, m=M):
    """MAE in-sample do naive sazonal de período m no treino."""
    y_train = np.asarray(y_train, dtype=float)
    return np.mean(np.abs(y_train[m:] - y_train[:-m]))


def calcular_metricas(y_train, y_val, yhat, m=M):
    erro = np.asarray(y_val, dtype=float) - np.asarray(yhat, dtype=float)
    mae = np.mean(np.abs(erro))
    rmse = np.sqrt(np.mean(erro**2))
    return mae, rmse, mae / escala_mase(y_train, m)
