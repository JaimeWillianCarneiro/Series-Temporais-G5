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

---

# AI_USAGE.md — Task 2 Séries Temporais (Grupo 5)

A seção da Task 1 acima fica como foi entregue. Esta seção cobre a Task 2: features, linear (01/11) e árvore/boosting (08/11).

## 1. Ferramentas usadas

* **Gemini** (Google): estruturação do código iterativo do linear, padronização temporal dos preços sem vazamento, laços `for` nas três séries (`store_total`, `FOODS`, `HOBBIES`) e organização do notebook de modelagem linear (`analise_task2.ipynb`).
* **Claude Code** (CLI da Anthropic, Claude Opus 5.5): revisão da entrega da árvore (boosting), unificação do pipeline em `run_task2.py`, visualizações e texto de interpretação da seção de árvore do notebook.
* Bibliotecas: `pandas`, `numpy`, `scikit-learn` (`Lasso`, `RandomForestRegressor`, `GradientBoostingRegressor`, `HistGradientBoostingRegressor`, `permutation_importance`), `joblib`, `matplotlib`.

## 2. Onde a IA ajudou

* **Features e padronização:** padronização dos preços com média e desvio só do treino (do treino interno, na validação interna); dummies de dia da semana e eventos; lags 1/7/14/28.
* **Validação interna temporal:** laço com origem em `2016-02-28` (prevê 2016-02-29 a 2016-03-27, dentro do treino) para escolher o `alpha` do Lasso por série e o modelo de árvore por série.
* **Linear:** ajuste definitivo até `2016-03-27`, previsão recursiva e empacotamento dos modelos com `joblib` (`.pkl`).
* **Árvore (boosting):** RandomForest, GradientBoosting e HistGradientBoosting com hiperparâmetros fixados a priori, comparados por MAE na validação interna. Vencedores: GradientBoosting (`store_total`, `HOBBIES`) e HistGradientBoosting (`FOODS`). Previsão recursiva.
* **Revisão e análise:** script único `run_task2.py` que reproduz as linhas `linear` e `boosting` a partir de `dados/`; gráficos de importância por permutação, erro por horizonte, resíduo por dia da semana e barras de métricas; texto comparando cada classe com o naive sazonal e o SARIMA.

## 3. Prompts representativos

1. "Duvida: para cada serie é uma regressào diferente, logo tem hiperparametros diferentes, certo?"
2. "Eu to com duvida na parte de validcao, verifique se na célula final do notebook eu estou pegando as variaveis corretas na parte de validaçao"
3. "O df_resultad_treino ele não era só com o treino interno? aqui como está ficando o ultimo for..."
4. "precisamos revisar os últimos commits desse repositório e garantir que tudo foi feito adequadamente [...] tudo relacionado a entrega 2 da task 2: 08/11/2026 boosting ou árvore [...] se não houve vazamento de dados, se todas visualizações clássicas foram feitas, se a análise está razoável, se a interpretação ainda está razoável"

## 4. Erros da IA identificados e corrigidos

1. **Vazamento no merge da validação interna (linear):** a IA sugeriu montar a tabela de busca de hiperparâmetros com o `df_treino` completo. Foi corrigido para usar só o treino interno (até `2016-02-28`) na busca e o treino completo só no ajuste definitivo.
2. **Pipeline que apagava as linhas da Task 2 (árvore):** as linhas `boosting` só existiam por causa de scripts auxiliares soltos, e o comando do README (`python run.py`) regravava `metricas.csv` só com os baselines e o SARIMA. Além disso, o notebook não gravava as previsões nem calculava as métricas do boosting. Corrigido com `run_task2.py` (o README agora roda `run.py && run_task2.py`) e com células no notebook que gravam e conferem os números contra `metricas.csv`.
3. **Interpretação sem olhar a figura (árvore):** a primeira versão do texto dizia que, em `HOBBIES`, os preços pesavam mais que nas outras séries. O gráfico de importância por permutação mostrou o contrário: o dia da semana domina e os preços quase não contam. O texto foi reescrito a partir das figuras.

## 5. Responsabilidade

O grupo responde por tudo o que está no repositório, inclusive o código e o texto produzidos com ajuda de IA. A escolha de hiperparâmetros só na validação interna, a padronização só com o treino e a previsão recursiva sem o y da validação foram conferidas pelo grupo. O `checks/check_task2.py --ate arvore` foi rodado antes desta entrega.
