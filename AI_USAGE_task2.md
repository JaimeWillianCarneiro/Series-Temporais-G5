
# AI_USAGE.md — Task 2 Séries Temporais (Grupo 5)

## 1. Ferramentas usadas

* **Gemini** (Google AI collaborator) para estruturação do código iterativo, validação da lógica de padronização temporal para evitar vazamento de dados (*data leakage*), criação de laços dinâmicos `for` para as três séries (`store_total`, `FOODS` e `HOBBIES`), e organização limpa do notebook de modelagem linear (`analise_task2.ipynb`).
* Bibliotecas: `pandas`, `pathlib`, `scikit-learn` (`Lasso`, `Ridge`, `mean_absolute_error`), `joblib`.

## 2. Onde a IA ajudou

* **Pré-processamento e Padronização:** Definição da estratégia de padronização manual de preços utilizando máscaras temporais estritamente restritas ao período de treino (evitando vazamento do conjunto de validação interna e holdout).
* **Validação Interna Temporal:** Construção do laço de validação com corte em `2016-02-28` para simulação preditiva e seleção do melhor hiperparâmetro (`alpha = 10`) de forma independente para cada uma das séries.
* **Treinamento Definitivo e Empacotamento:** Estruturação do fluxo para reter a base completa de treino (até `2016-03-27`), ajuste do modelo definitivo via `.fit()` e empacotamento dos objetos treinados junto com suas estatísticas de média e desvio em dicionários salvos via `joblib.dump()` (`.pkl`).
* **Refatoração de Código em Laços (`for`):** Otimização do código para iterar dinamicamente sobre as listas de séries, selecionando alvos e covariáveis sem duplicação desnecessária de blocos.

## 3. Prompts representativos

1. "Duvida: para cada serie é uma regressào diferente,  logo tem hiperparametros diferentes, certo?"
2. "Na parte quando a gente tava fazendo só pra śerie food: `df_resultado_treino = pd.merge(df_treino_interno,df_covariaveis, on= 'date', how = 'inner' )` na parte do for, eu merjo todo os dataframes ( com as 3 padronizações) e por enquanto o for restá assim..."
3. "Eu to com duvida na parte de validcao, verifique se na célula final do notebook eu estou pegando as variaveis corretas na parte de validaçao"
4. "O df_resultad_treino ele não era só com o treino interno? aqui como está ficando o ultimo for..."

## 4. Erros da IA identificados e corrigidos

1. **Vazamento de dados (*Data Leakage*) no Merge de Validação Interna:** Inicialmente, a IA sugeriu fazer o merge do treino utilizando o `df_treino` completo logo na etapa de busca de hiperparâmetros. Isso foi corrigido para utilizar estritamente o `df_treino_interno` (dados até `2016-02-28`) na validação e reservar o `df_treino` completo apenas para o treinamento definitivo.
2. **Incompatibilidade de Nomes de Colunas no Scikit-Learn:** O modelo gerava um `ValueError` de *feature names unseen at fit time* devido à presença simultânea de colunas padronizadas de preços de outras séries no `X`. O código foi ajustado para isolar e processar dinamicamente as colunas correspondentes à série em foco em cada iteração.
3. **Escopo de Variáveis no Loop de Validação:** Correção de chamadas a variáveis estáticas e escopos cruzados de hiperparâmetros, passando a armazenar e recuperar os melhores `alpha` e estatísticas de padronização por meio de dicionários indexados pelo nome da série (`melhores_alpha`, `medias`, `desvios`).

## 5. Responsabilidade

O grupo responde por tudo o que está no repositório, inclusive o código e a documentação produzidos com o auxílio de IA. As etapas de validação temporal, isolamento de dados de treino/validação interna e exportação dos modelos foram conferidas de acordo com as diretrizes metodológicas da disciplina e os critérios institucionais da Task 2.

---