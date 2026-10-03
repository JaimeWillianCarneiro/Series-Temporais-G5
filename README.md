# Series-Temporais-G5

## Task 1 — Diagnóstico, baselines e ARIMA/SARIMA

Pacote da Task 1 (prazo 30/09, 25% da A2). Enunciado, critérios e formato de entrega: [`enunciado.md`](enunciado.md).

| Peça | Arquivo |
|------|---------|
| Enunciado (prazo 30/09, 25% da A2, tabela de 10 pontos) | [`enunciado.md`](enunciado.md) · [`enunciado.pdf`](enunciado.pdf) |
| Treino e validação conhecida | [`dados/`](dados/) |
| Datas do holdout, sem y | [`dados/holdout_datas.csv`](dados/holdout_datas.csv) |
| Script de checks (crédito parcial) | [`checks/check_task1.py`](checks/check_task1.py) |

## Como reproduzir

```bash
# 1. Instalar dependências
python -m pip install -r requirements.txt

# 2. Executar ARIMA/SARIMA e gerar previsões
python run.py

# 3. Validar resultado (após calcular métricas)
python checks/check_task1.py --submission .
```

O script `run.py` gera:
- `previsoes_validacao.csv` — Previsões SARIMA para os 28 dias de validação

Os notebooks em `src/entrega_1/` documentam:
- `01_diagnostico_visual.ipynb` — Análise visual, ACF/PACF das três séries
- `02_baselines.ipynb` — Implementação dos quatro baselines
- `03_arima_sarima.ipynb` — Identificação, ajuste e previsão de SARIMA

O y do holdout não entra nesta entrega. Rodar o check a partir da raiz desta pasta (o default de `--dados` é a pasta `dados/` ao lado de `checks/`):

```bash
python checks/check_task1.py --submission /caminho/do/repo/do/grupo
```
