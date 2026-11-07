# Series-Temporais-G5

FGV EMAp · Séries Temporais 2026/2. Séries diárias da loja M5 `CA_1`: `store_total`, `FOODS`, `HOBBIES`.

- Task 1: diagnóstico, baselines e ARIMA/SARIMA. Enunciado: [`docs/enunciado-task1.md`](docs/enunciado-task1.md)
- Task 2: features de `covariaveis.csv`, `linear` (Lasso) e `boosting`, previsão recursiva. Enunciado: [`docs/enunciado-task2.md`](docs/enunciado-task2.md)

## Como reproduzir

```bash
python -m venv .venv && source .venv/bin/activate
python -m pip install -r requirements.txt
python run.py && python run_task2.py           # ~2 min: Task 1 (baselines, SARIMA) + Task 2 (linear, boosting)
python checks/check_task2.py --submission . --ate arvore   # confere os critérios
```

`run.py` grava as linhas da Task 1, e `run_task2.py` acrescenta `linear` e `boosting` às mesmas
tabelas, sem tocar nas linhas da Task 1. Rodar só `run.py` deixa `metricas.csv` sem as linhas da Task 2.

`run.py` ajusta tudo **só no treino** (até 2016-03-27) e avalia nos 28 dias de validação
(2016-03-28 a 2016-04-24). O holdout não é usado.

## Fonte da análise

[`analise_task2.ipynb`](analise_task2.ipynb) documenta a Task 2: colunas usadas, estratégia recursiva, escolha de
hiperparâmetros na validação interna (origem 2016-02-28, dentro do treino), visualizações e a comparação de cada
classe com o naive sazonal e o SARIMA.

[`analise_task1.ipynb`](analise_task1.ipynb) é o documento principal da Task 1: diagnóstico (série, tendência,
sazonalidade, zeros, ACF/PACF, ADF/KPSS), baselines e SARIMA, com o texto que justifica cada decisão
e a comparação com os baselines. Execute-o a partir da raiz, depois de rodar `python run.py`
(as células de conferência comparam o notebook com `metricas.csv`).

## Saídas do `run.py`

| Arquivo | Conteúdo |
|---|---|
| `metricas.csv` | MAE, RMSE, MASE por série e modelo (`media`, `naive`, `naive_sazonal`, `drift`, `sarima`, `linear`, `boosting`) |
| `previsoes_validacao.csv` | `date, series, modelo, yhat` |
| `saidas/ordens_sarima.csv` | ordem SARIMA escolhida por série, com AIC/BIC |
| `saidas/diagnostico_testes.csv` | zeros de calendário, ADF e KPSS |
| `figuras/diagnostico_<serie>.png` | série + média móvel, ACF e PACF no treino |
| `figuras/acf_pacf_diferenciada_<serie>.png` | ACF/PACF após diff(1) e diff sazonal(7) |
| `figuras/previsoes_validacao.png` | previsões vs. real na validação |
| `saidas/escolhas_task2.csv` | `run_task2.py`: alpha do Lasso e árvore escolhida por série, com MAE da validação interna |
| `modelo_<serie>.pkl`, `modelo_arvore_<serie>.pkl` | `run_task2.py`: modelos linear e boosting ajustados até 2016-03-27 |

## Estrutura

```
analise_task1.ipynb   notebook da Task 1 (diagnóstico, baselines, SARIMA)
analise_task2.ipynb   notebook da Task 2 (features, linear, boosting)
run.py                pipeline da Task 1: metricas.csv, previsoes e figuras
run_task2.py          pipeline da Task 2: acrescenta linear e boosting
src/ts_utils.py       baselines, métricas e carregamento (usado por run.py e pelo notebook)
dados/                treino, validação e datas do holdout (sem y)
figuras/, saidas/     saídas geradas por run.py
checks/               script de verificação do professor
docs/                 enunciado
AI_USAGE.md           uso de IA
```
