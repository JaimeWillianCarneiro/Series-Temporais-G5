#!/usr/bin/env python3
"""
Task 1: ARIMA/SARIMA — Identificação, ajuste e previsão
Gera previsoes_validacao.csv com previsões SARIMA para os 28 dias de validação
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from statsmodels.tsa.statespace.sarimax import SARIMAX
import itertools


def find_best_sarima(serie, serie_name, m=7):
    """Grid search com fallback padrão"""
    best_aic = np.inf
    best_order = (1, 1, 1)
    best_seasonal_order = (1, 0, 1, m)
    found_valid = False

    p_opts = [0, 1, 2]
    d_opts = [0, 1]
    q_opts = [0, 1, 2]
    P_opts = [0, 1]
    D_opts = [0]
    Q_opts = [0, 1]

    total = len(p_opts) * len(d_opts) * len(q_opts) * len(P_opts) * len(D_opts) * len(Q_opts)
    print(f"  {serie_name}: testando {total} combinações...")

    for p, d, q, P, D, Q in itertools.product(p_opts, d_opts, q_opts, P_opts, D_opts, Q_opts):
        try:
            model = SARIMAX(serie, order=(p, d, q), seasonal_order=(P, D, Q, m),
                          enforce_stationarity=False, enforce_invertibility=False)
            results = model.fit(disp=False, maxiter=150)
            aic = results.aic

            if aic < best_aic:
                best_aic = aic
                best_order = (p, d, q)
                best_seasonal_order = (P, D, Q, m)
                found_valid = True
        except:
            continue

    if not found_valid:
        print(f"    ⚠ Usando fallback SARIMA{best_order}{best_seasonal_order}")
    else:
        print(f"    ✓ Melhor: SARIMA{best_order}{best_seasonal_order} (AIC={best_aic:.2f})")

    return best_order, best_seasonal_order


def forecast_sarima(serie, order, seasonal_order, h):
    """Ajusta e prevê"""
    model = SARIMAX(serie, order=order, seasonal_order=seasonal_order,
                   enforce_stationarity=False, enforce_invertibility=False)
    results = model.fit(disp=False, maxiter=150)
    forecast_result = results.get_forecast(steps=h)
    if hasattr(forecast_result.predicted_mean, 'values'):
        return forecast_result.predicted_mean.values
    else:
        return np.asarray(forecast_result.predicted_mean)


def main():
    print("="*70)
    print("Task 1: ARIMA/SARIMA")
    print("="*70)

    print("\n1. Carregando dados...")
    df_train = pd.read_csv("dados/treino.csv")
    df_val = pd.read_csv("dados/validacao.csv")

    df_train["date"] = pd.to_datetime(df_train["date"])
    df_val["date"] = pd.to_datetime(df_val["date"])

    series_names = ["store_total", "FOODS", "HOBBIES"]
    h = len(df_val)

    print(f"   Treino: {len(df_train)} dias | Validação: {h} dias")

    print("\n2. Identificando ordens SARIMA...")
    sarima_orders = {}

    for serie_name in series_names:
        order, seasonal_order = find_best_sarima(df_train[serie_name].values, serie_name, m=7)
        sarima_orders[serie_name] = (order, seasonal_order)

    print("\n3. Gerando previsões SARIMA...")
    predictions = []

    for serie_name in series_names:
        y_train = df_train[serie_name].values
        order, seasonal_order = sarima_orders[serie_name]

        forecast = forecast_sarima(y_train, order, seasonal_order, h)
        for date, yhat in zip(df_val["date"], forecast):
            predictions.append({"date": date, "series": serie_name, "modelo": "sarima", "yhat": yhat})

    print("   ✓ Previsões geradas para 3 séries")

    print("\n4. Salvando previsoes_validacao.csv...")
    df_predictions = pd.DataFrame(predictions)
    df_predictions.to_csv("previsoes_validacao.csv", index=False)
    print(f"   ✓ {len(df_predictions)} previsões salvas")

    print("\n" + "="*70)
    print("ORDENS SARIMA IDENTIFICADAS (via grid search com AIC)")
    print("="*70)
    for serie_name in series_names:
        order, seasonal_order = sarima_orders[serie_name]
        p, d, q = order
        P, D, Q, m = seasonal_order
        print(f"{serie_name:15s}: SARIMA({p},{d},{q})({P},{D},{Q},{m})")

    print("\n✓ Concluído! CSV gerado: previsoes_validacao.csv\n")


if __name__ == "__main__":
    main()
