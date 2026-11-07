#!/usr/bin/env python3
import json
from pathlib import Path

RAIZ = Path(__file__).resolve().parent
nb_path = RAIZ / "analise_task2.ipynb"
nb = json.loads(nb_path.read_text(encoding="utf-8"))

cells = []

# Markdown section
cells.append({"cell_type": "markdown", "metadata": {}, "source": [
    "## 5. Modelagem com Árvores (Boosting)\n",
    "\n",
    "Reaproveitando o mesmo pipeline do linear (features de `covariaveis.csv`, lags 1/7/14/28, validação interna com corte em 2016-02-28, previsão recursiva). Treinamos três candidatos por série — RandomForest, GradientBoosting e HistGradientBoosting — e escolhemos o de menor MAE na validação interna. Hiperparâmetros fixados antes de ler o MAE da validação.\n",
]})

# Code: imports
cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": [
    "from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, HistGradientBoostingRegressor\n",
    "from sklearn.metrics import mean_absolute_error\n",
]})

# Code: validacao interna
cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": [
    "candidatos = {\n",
    "    'RandomForest': lambda: RandomForestRegressor(n_estimators=300, max_depth=8, random_state=42, n_jobs=1),\n",
    "    'GradientBoosting': lambda: GradientBoostingRegressor(n_estimators=300, max_depth=3, learning_rate=0.05, random_state=42),\n",
    "    'HistGradientBoosting': lambda: HistGradientBoostingRegressor(max_iter=300, max_depth=6, learning_rate=0.05, random_state=42),\n",
    "}\n",
    "\n",
    "melhor_por_serie = {}\n",
    "for serie in series:\n",
    "    X_tr, y_tr = montar_treino(hist_interno, cov_interno, serie)\n",
    "    y_vl = df_validacao_interno.sort_values('date')[serie].values\n",
    "    erros = {}\n",
    "    for nome, ctor in candidatos.items():\n",
    "        m = ctor().fit(X_tr, y_tr)\n",
    "        yhat = prever_recursivo(m, hist_interno.sort_values('date')[serie].values, df_cov_interno_val)\n",
    "        erros[nome] = mean_absolute_error(y_vl, yhat)\n",
    "    melhor_por_serie[serie] = min(erros, key=erros.get)\n",
    "    print(f\"{serie}: vencedor = {melhor_por_serie[serie]}  | MAE interno: { {k: round(v,1) for k,v in erros.items()} }\")\n",
]})

# Code: treino definitivo + export
cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": [
    "for serie in series:\n",
    "    X_t, y_t = montar_treino(df_treino, df_covariaveis_final, serie)\n",
    "    ctor = candidatos[melhor_por_serie[serie]]\n",
    "    clf = ctor().fit(X_t, y_t)\n",
    "    joblib.dump({'modelo': clf, 'nome': melhor_por_serie[serie], 'lags': LAGS, 'stats_precos': stats_final}, f'modelo_arvore_{serie}.pkl')\n",
    "    print(f\"salvo modelo_arvore_{serie}.pkl ({melhor_por_serie[serie]})\")\n",
]})

# Code: previsoes
cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": [
    "lista_boost = []\n",
    "for serie in series:\n",
    "    pacote = joblib.load(f'modelo_arvore_{serie}.pkl')\n",
    "    yhat = prever_recursivo(pacote['modelo'], treino_hist[serie].values, cov_val)\n",
    "    for d, y in zip(cov_val['date'], yhat):\n",
    "        lista_boost.append({'date': d.strftime('%Y-%m-%d'), 'series': serie, 'modelo': 'boosting', 'yhat': y})\n",
    "df_previsoes_boost = pd.DataFrame(lista_boost)\n",
    "df_previsoes_boost.groupby(['series', 'modelo']).size()\n",
]})

nb["cells"].extend(cells)
nb_path.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
print("células adicionadas")
