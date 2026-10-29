# AI_USAGE.md — Task 1 Séries Temporais (Grupo 5)

## 1. Ferramentas usadas

- **Claude Code** (CLI da Anthropic) em dois momentos: a primeira versão do SARIMA e a revisão final do repo. O modelo usado na primeira versão não foi registrado; a revisão final foi feita com Claude Sonnet 5.5.
- Bibliotecas: `statsmodels`, `pandas`, `numpy`, `matplotlib`, `scipy`.

## 2. Onde a IA ajudou

- Primeira versão de `run.py` e do notebook de SARIMA (grid search).
- Revisão final: refatoração de baselines e métricas para `src/ts_utils.py`, figuras de ACF/PACF em arquivo, grid paralelo, README e `requirements.txt`.
- Unificação dos três notebooks do grupo (diagnóstico, baselines, SARIMA) em `analise_task1.ipynb`, com texto explicativo e células que conferem o resultado contra `metricas.csv`. O diagnóstico visual, os baselines e a primeira versão do SARIMA foram escritos pelos integrantes do grupo.

## 3. Prompts representativos

1. "Observe a pasta do projeto. Sou responsável por fazer ARIMA/SARIMA. Não faça nada ainda, apenas observe e dê ideias."
2. "Implementa o ARIMA/SARIMA. A parte das métricas não é comigo."
3. "Pode revisar o estado desse projeto? Se todas as demandas foram cumpridas, se falta rodar alguma coisa, se falta organizar alguma coisa, quais bibliotecas ele usa, como estão organizadas as pastas."
4. "Pode prosseguir com todas as sugestões de melhoria ou completude?"

## 4. Erros da IA identificados e corrigidos

1. **Ordem "via ACF/PACF" sem ACF/PACF:** a primeira versão do SARIMA escolhia a ordem só por grid search de AIC, e o texto dizia que ela vinha de ACF/PACF. A revisão passou a justificar d, D e a grade com os gráficos da série diferenciada (ver `analise_task1.ipynb`).
2. **`except:` genérico e `enforce_stationarity=False`:** a primeira versão engolia qualquer erro do ajuste e desligava as restrições de estacionariedade e invertibilidade. Agora captura só `LinAlgError`/`ValueError`, usa as restrições padrão e descarta ajustes que não convergem.
3. **Código e métricas duplicados:** baselines e métricas estavam copiados no `run.py` e em dois notebooks, com CSVs divergentes em `src/entrega_1/` (notebooks hoje unificados em `analise_task1.ipynb`). Ficaram numa única implementação (`src/ts_utils.py`), e os CSVs duplicados foram removidos.
4. **Diagnóstico sem arquivos:** as figuras de ACF/PACF só existiam dentro dos notebooks, e o `check_task1.py` dava 0/0,5. Agora `run.py` salva os PNGs em `figuras/`.
5. **Divergência com o enunciado:** o naive sazonal usava `h // m` repetições, o que só cobre h múltiplo de 7. Agora usa `ceil(h/m)` e corta em h.

## 5. Responsabilidade

O grupo responde por tudo o que está no repo, inclusive o código e o texto produzidos com ajuda de IA. As métricas foram conferidas contra a definição do enunciado (MASE com escala do naive sazonal de período 7 no treino), e o `check_task1.py` foi rodado antes desta versão.
