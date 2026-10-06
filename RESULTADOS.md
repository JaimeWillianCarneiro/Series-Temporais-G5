# Task 1 — Resultados e discussão

Tudo abaixo vem de `python run.py` (ajuste só no treino, avaliação em 2016-03-28 a 2016-04-24).
Figuras em `figuras/`.

## 1. Diagnóstico (treino)

- **Tendência:** leve alta nas três séries, mais visível em `store_total` e `FOODS` entre 2011 e 2013, depois quase estável (média móvel de 365 dias). ADF no nível não rejeita raiz unitária em `store_total` (p=0,25) e `FOODS` (p=0,14); em `HOBBIES` rejeita a 5% (p=0,035), e o KPSS rejeita estacionariedade nas três (p≈0,01). Conclusão: usar d=1 para `store_total` e `FOODS`; `HOBBIES` é um caso limítrofe.
- **Sazonalidade semanal (m=7):** o ACF das três séries tem picos nos lags 7, 14, 21 e 28 (≈0,8 em `store_total`), e o PACF é significativo nos lags 5–7. Por isso D=1. Depois de diff(1) e diff sazonal(7), o ADF rejeita raiz unitária com folga (p<1e-27) e o ACF cai a ruído a partir do lag ~8.
- **Outliers:** há 5 zeros de calendário em cada série, um por ano no fim de dezembro (loja fechada). São quedas isoladas até 0, visíveis nos gráficos. Não foram removidos nem imputados; entram no treino do SARIMA e na escala do MASE.

## 2. Identificação do SARIMA

Após diff(1) e diff sazonal(7) (`figuras/acf_pacf_diferenciada_*.png`):

- ACF: pico negativo forte no lag 1 (≈ −0,4) e no lag 7 (≈ −0,5), e pequenos picos em 6 e 8. Isso sugere MA(1) não sazonal e MA sazonal Q=1, parte de um "airline model" (0,1,1)(0,1,1)₇.
- PACF: decaimento lento nos lags 1–4 e em múltiplos de 7, o que também indica termos MA e deixa espaço para algum AR.

Isso fixa D=1 e m=7 e restringe a grade a p,q ≤ 2, P,Q ≤ 1 e d ∈ {0,1}. Dentro dela, `run.py` ajusta 72 modelos por série e escolhe o de menor AIC, descartando os que não convergem (`saidas/ordens_sarima.csv`).

| Série | Ordem escolhida | AIC |
|---|---|---|
| `store_total` | SARIMA(2,1,1)(1,1,1)₇ | 28143,9 |
| `FOODS` | SARIMA(2,1,1)(1,1,1)₇ | 27051,5 |
| `HOBBIES` | SARIMA(2,0,1)(0,1,1)₇ | 22117,2 |

As escolhas são coerentes com o diagnóstico: termos MA no lag 1 e no sazonal, e d=0 em `HOBBIES`, onde o ADF já quase rejeitava a raiz unitária. Os melhores modelos ficam a menos de 1 ponto de AIC dos vizinhos (por exemplo (1,1,1)(1,1,1)₇ em `store_total`), então a diferença entre eles não é conclusiva. Comparar AIC entre valores diferentes de d exige cuidado, porque a diferenciação muda a verossimilhança. Por isso o d de `HOBBIES` é uma escolha menos firme que os demais.

## 3. Métricas na validação (28 dias)

| Série | Modelo | MAE | RMSE | MASE |
|---|---|---:|---:|---:|
| store_total | media | 813,9 | 1109,6 | 1,87 |
| store_total | naive | 862,8 | 966,6 | 1,98 |
| store_total | naive_sazonal | 459,2 | 655,5 | 1,06 |
| store_total | drift | 863,3 | 966,8 | 1,98 |
| store_total | **sarima** | **276,9** | **363,8** | **0,64** |
| FOODS | media | 526,9 | 671,8 | 1,54 |
| FOODS | naive | 664,6 | 722,9 | 1,94 |
| FOODS | naive_sazonal | 343,9 | 420,1 | 1,01 |
| FOODS | drift | 665,0 | 723,3 | 1,94 |
| FOODS | **sarima** | **186,1** | **231,5** | **0,54** |
| HOBBIES | media | 111,0 | 141,3 | 1,22 |
| HOBBIES | naive | 169,0 | 204,7 | 1,85 |
| HOBBIES | naive_sazonal | 102,3 | 146,7 | 1,12 |
| HOBBIES | drift | 170,3 | 205,9 | 1,87 |
| HOBBIES | **sarima** | **75,7** | **95,2** | **0,83** |

## 4. Comparação com os baselines

- **O SARIMA vence todos os baselines em MAE, RMSE e MASE nas três séries.** O ganho sobre o naive sazonal em MAE é de 40% em `store_total`, 46% em `FOODS` e 26% em `HOBBIES`.
- **`naive` e `drift` são os piores**: MASE perto de 1,9–2,0 em todas as séries, porque ignoram a sazonalidade semanal. O drift é praticamente igual ao naive (a inclinação do treino é minúscula).
- **A média perde para o naive sazonal em `store_total` e `FOODS`** (MAE 814 vs 459 e 527 vs 344). Em `HOBBIES` o resultado é misto: o naive sazonal tem MAE menor (102,3 vs 111,0), mas a média tem RMSE menor (141,3 vs 146,7). Nessa série os dois têm MASE > 1, ou seja, são piores que o naive sazonal in-sample. `HOBBIES` é mais ruidosa, e copiar a última semana repete o ruído dela.
- **Limitação:** a validação tem só 28 dias (4 ciclos semanais) e um único corte. A diferença entre os modelos é grande, mas a ordem exata entre eles em séries ruidosas como `HOBBIES` não é garantida em outros períodos. Não há intervalos de confiança nem backtesting em múltiplas origens.
