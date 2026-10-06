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


SERIES_NAMES = ["store_total", "FOODS", "HOBBIES"]
H = 28
M = 7

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
    D_opts = [0, 1]
    Q_opts = [0, 1]

    total = len(p_opts) * len(d_opts) * len(q_opts) * len(P_opts) * len(D_opts) * len(Q_opts)
    print(f"  {serie_name}: testando {total} combinações...")

    for p, d, q, P, D, Q in itertools.product(p_opts, d_opts, q_opts, P_opts, D_opts, Q_opts):
        try:
            model = SARIMAX(serie, order=(p, d, q), seasonal_order=(P, D, Q, m),
                          enforce_stationarity=False, enforce_invertibility=False)
            results = model.fit(disp=False, maxiter=200)
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
    results = model.fit(disp=False, maxiter=200)
    forecast_result = results.get_forecast(steps=h)
    if hasattr(forecast_result.predicted_mean, 'values'):
        return forecast_result.predicted_mean.values
    else:
        return np.asarray(forecast_result.predicted_mean)

def calcular_baselines(y_train, h=28, m=7):
    """Gera os quatro baselines da atividade."""

    # Média
    yhat_media = np.full(h, np.mean(y_train))

    # Naive
    y_last = y_train[-1]
    yhat_naive = np.full(h, y_last)

    # Naive sazonal
    yhat_sazonal = np.tile(y_train[-m:], h // m)

    # Drift
    T = len(y_train)
    y_first = y_train[0]
    C = (y_last - y_first) / (T - 1)

    yhat_drift = y_last + np.arange(1, h + 1) * C

    return {
        "media": yhat_media,
        "naive": yhat_naive,
        "naive_sazonal": yhat_sazonal,
        "drift": yhat_drift,
    }


def calcular_metricas(y_train, y_val, yhat, m=7):
    """Calcula MAE, RMSE e MASE."""

    erro = y_val - yhat

    mae = np.mean(np.abs(erro))
    rmse = np.sqrt(np.mean(erro**2))

    # Escala do MASE:
    # MAE do naive sazonal no conjunto de treino
    escala_mase = np.mean(
        np.abs(y_train[m:] - y_train[:-m])
    )

    mase = mae / escala_mase

    return mae, rmse, mase


def main():
    print("="*70)
    print("Task 1: ARIMA/SARIMA")
    print("="*70)

    print("\n1. Carregando dados...")
    df_train = pd.read_csv("dados/treino.csv")
    df_val = pd.read_csv("dados/validacao.csv")

    df_train["date"] = pd.to_datetime(df_train["date"])
    df_val["date"] = pd.to_datetime(df_val["date"])

    # series_names = ["store_total", "FOODS", "HOBBIES"]
    h = len(df_val)

    print(f"   Treino: {len(df_train)} dias | Validação: {h} dias")

    print("\n2. Gerando baselines...")

    predictions = []
    for serie_name in SERIES_NAMES:

        y_train = df_train[serie_name].to_numpy()

        baselines = calcular_baselines(
            y_train,
            h=h,
            m=M
        )

        for modelo, forecast in baselines.items():

            for date, yhat in zip(
                df_val["date"],
                forecast
            ):
                predictions.append(
                    {
                        "date": date,
                        "series": serie_name,
                        "modelo": modelo,
                        "yhat": yhat,
                    }
                )


    print("   ✓ Média")
    print("   ✓ Naive")
    print("   ✓ Naive sazonal")
    print("   ✓ Drift")


    print("\n3. Identificando ordens SARIMA...")

    sarima_orders = {}

    for serie_name in SERIES_NAMES:

        y_train = df_train[serie_name].to_numpy()

        order, seasonal_order = find_best_sarima(
            y_train,
            serie_name,
            m=M
        )

        sarima_orders[serie_name] = (
            order,
            seasonal_order
        )
    print("\n4. Gerando previsões SARIMA...")

    for serie_name in SERIES_NAMES:

        y_train = df_train[serie_name].to_numpy()

        order, seasonal_order = sarima_orders[
            serie_name
        ]

        forecast = forecast_sarima(
            y_train,
            order,
            seasonal_order,
            h
        )

        for date, yhat in zip(
            df_val["date"],
            forecast
        ):
            predictions.append(
                {
                    "date": date,
                    "series": serie_name,
                    "modelo": "sarima",
                    "yhat": yhat,
                }
            )

    print("   ✓ SARIMA")


    print("\n5. Salvando previsoes_validacao.csv...")
    df_predictions = pd.DataFrame(predictions)
    df_predictions.to_csv("previsoes_validacao.csv", index=False)
    print(f"   ✓ {len(df_predictions)} previsões salvas")

    print("\n6. Calculando métricas...")

    metricas = []

    for serie_name in SERIES_NAMES:

        y_train = df_train[serie_name].to_numpy()
        y_val = df_val[serie_name].to_numpy()

        df_serie = df_predictions[
            df_predictions["series"] == serie_name
        ]

        for modelo in [
            "media",
            "naive",
            "naive_sazonal",
            "drift",
            "sarima",
        ]:

            yhat = (
                df_serie[
                    df_serie["modelo"] == modelo
                ]
                .sort_values("date")["yhat"]
                .to_numpy()
            )

            mae, rmse, mase = calcular_metricas(
                y_train,
                y_val,
                yhat,
                m=M
            )

            metricas.append(
                {
                    "series": serie_name,
                    "modelo": modelo,
                    "mae": mae,
                    "rmse": rmse,
                    "mase": mase,
                }
            )

    df_metricas = pd.DataFrame(metricas)

    df_metricas.to_csv(
        "metricas.csv",
        index=False
    )

    print("   ✓ metricas.csv salvo")


    print("\n" + "=" * 70)
    print("ORDENS SARIMA IDENTIFICADAS")
    print("=" * 70)

    for serie_name in SERIES_NAMES:

        order, seasonal_order = sarima_orders[
            serie_name
        ]

        print(
            f"{serie_name:15s}: "
            f"SARIMA{order}{seasonal_order}"
        )

    print("\n" + "=" * 70)
    print("MELHORES MODELOS POR SÉRIE")
    print("=" * 70)

    melhores = (
        df_metricas
        .loc[
            df_metricas
            .groupby("series")["mae"]
            .idxmin()
        ]
        .sort_values("series")
    )

    print(melhores.to_string(index=False))

    print("\n✓ Concluído!")
    print("  - previsoes_validacao.csv")
    print("  - metricas.csv")


    

if __name__ == "__main__":
    main()
