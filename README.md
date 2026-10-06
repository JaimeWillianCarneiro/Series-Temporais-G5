# Series-Temporais-G5

Task 1 — diagnóstico, baselines e ARIMA/SARIMA (FGV EMAp · Séries Temporais 2026/2).
Séries diárias da loja M5 `CA_1`: `store_total`, `FOODS`, `HOBBIES`.
Enunciado: [`docs/enunciado.md`](docs/enunciado.md)

## Como reproduzir

```bash
python -m venv .venv && source .venv/bin/activate
python -m pip install -r requirements.txt
python run.py                                  # ~1 min (grid SARIMA em paralelo)
python checks/check_task1.py --submission .    # confere os critérios
```

`run.py` ajusta tudo **só no treino** (até 2016-03-27) e avalia nos 28 dias de validação
(2016-03-28 a 2016-04-24). O holdout não é usado.

## Fonte da análise

[`analise_task1.ipynb`](analise_task1.ipynb) é o documento principal: diagnóstico (série, tendência,
sazonalidade, zeros, ACF/PACF, ADF/KPSS), baselines e SARIMA, com o texto que justifica cada decisão
e a comparação com os baselines. Execute-o a partir da raiz, depois de rodar `python run.py`
(as células de conferência comparam o notebook com `metricas.csv`).

## Saídas do `run.py`

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
analise_task1.ipynb   notebook principal (diagnóstico, baselines, SARIMA)
run.py                pipeline que gera metricas.csv, previsoes e figuras
src/ts_utils.py       baselines, métricas e carregamento (usado por run.py e pelo notebook)
dados/                treino, validação e datas do holdout (sem y)
figuras/, saidas/     saídas geradas por run.py
checks/               script de verificação do professor
docs/                 enunciado
AI_USAGE.md           uso de IA
```
