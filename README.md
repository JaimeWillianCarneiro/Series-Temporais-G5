# Series-Temporais-G5

Task 1 — diagnóstico, baselines e ARIMA/SARIMA (FGV EMAp · Séries Temporais 2026/2).
Séries diárias da loja M5 `CA_1`: `store_total`, `FOODS`, `HOBBIES`.
Enunciado: [`docs/enunciado.md`](docs/enunciado.md) · Análise e conclusões: [`RESULTADOS.md`](RESULTADOS.md)

## Como reproduzir

```bash
python -m venv .venv && source .venv/bin/activate
python -m pip install -r requirements.txt
python run.py                                  # ~1 min (grid SARIMA em paralelo)
python checks/check_task1.py --submission .    # confere os critérios
```

`run.py` ajusta tudo **só no treino** (até 2016-03-27) e avalia nos 28 dias de validação
(2016-03-28 a 2016-04-24). O holdout não é usado.

## Saídas

| Arquivo | Conteúdo |
|---|---|
| `metricas.csv` | MAE, RMSE, MASE por série e modelo (`media`, `naive`, `naive_sazonal`, `drift`, `sarima`) |
| `previsoes_validacao.csv` | `date, series, modelo, yhat` |
| `saidas/ordens_sarima.csv` | ordem SARIMA escolhida por série, com AIC/BIC |
| `saidas/diagnostico_testes.csv` | zeros de calendário, ADF e KPSS |
| `figuras/diagnostico_<serie>.png` | série + média móvel, ACF e PACF no treino |
| `figuras/acf_pacf_diferenciada_<serie>.png` | ACF/PACF após diff(1) e diff sazonal(7) |
| `figuras/previsoes_validacao.png` | previsões vs. real na validação |

## Estrutura

```
dados/            treino, validação e datas do holdout (sem y)
src/ts_utils.py   baselines, métricas e carregamento (usado por run.py)
src/entrega_1/    notebooks exploratórios (01 diagnóstico, 02 baselines, 03 SARIMA)
run.py            pipeline completo, fonte oficial de metricas.csv
checks/           script de verificação do professor
docs/             enunciado
figuras/          figuras geradas por run.py
saidas/           tabelas auxiliares (ordens SARIMA, ADF/KPSS)
```

Os notebooks documentam a exploração; os números oficiais vêm do `run.py`.
