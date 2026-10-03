# AI_USAGE.md — Task 1 Séries Temporais

## 1. Ferramentas usadas

- **Claude Haiku** (claude-code) — Geração de código, estrutura de pipeline
- **Python** com `statsmodels`, `pandas`, `numpy` — Modelos ARIMA/SARIMA

## 2. Onde a IA ajudou

- **Identificação de ordens SARIMA:** Grid search com AIC em `statsmodels`
- **Ajuste e previsão:** Implementação do workflow SARIMA (treino → validação)
- **Validação de dados:** Verificação de split temporal (sem leakage)
- **Script executável:** `run.py` que identifica ordens e gera previsões SARIMA

## 3. Prompts representativos

### Prompt 1: Estrutura inicial
> "Observe st/Séries-Temporais-G5. Sou responsável por fazer ARIMA/SARIMA. Não faça nada ainda, apenas observe e dê ideias."

**Resultado:** Análise do projeto, identificação de GAPs, recomendação de pipeline.

### Prompt 2: Implementação
> "Manda, implementa o ARIMA/SARIMA. A parte das métricas não é comigo."

**Resultado:** Notebook 03 + script `run.py` que identifica ordens via grid search e gera previsões.

### Prompt 3: Validação
> Verificação de imports, testes no split temporal, garantia de que treino ≠ validação.

## 4. Erros da IA que foram corrigidos

1. **Importação de statsmodels:** Não estava instalado → instalação manual em venv
2. **Dimensões das previsões:** Garantir que as previsões têm exatamente 28 dias (validação)
3. **Formato de saída:** CSV precisava de colunas exatas (`date`, `series`, `modelo`, `yhat`)

## 5. Responsabilidade

O grupo é responsável por:
- Validar que as métricas (MAE, RMSE, MASE) estão corretas
- Conferir que os 4 baselines reproduzem os valores esperados
- Interpretar as ordens SARIMA identificadas (comentário sobre tendência/sazonalidade)
- Rodar o check (`python checks/check_task1.py`) antes da entrega

**Gerado com Claude Code | 2026-10-03**
