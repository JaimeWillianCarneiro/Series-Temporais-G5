#!/usr/bin/env python3
import json
from pathlib import Path

RAIZ = Path(__file__).resolve().parent
nb_path = RAIZ / "analise_task2.ipynb"
nb = json.loads(nb_path.read_text(encoding="utf-8"))

cells = []

cells.append({"cell_type": "markdown", "metadata": {}, "source": [
    "## 6. Visualização — Árvore (Boosting)\n",
]})
cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": [
    "import pandas as pd\n",
    "import matplotlib.pyplot as plt\n",
    "treino = pd.read_csv('dados/treino.csv', parse_dates=['date'])\n",
    "val = pd.read_csv('dados/validacao.csv', parse_dates=['date'])\n",
    "prev = pd.read_csv('saidas/previsoes_validacao.csv', parse_dates=['date'])\n",
    "prev = prev[prev['modelo'] == 'boosting']\n",
    "series = ['store_total', 'FOODS', 'HOBBIES']\n",
    "fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(12, 13))\n",
    "for ax, serie in zip(axes, series):\n",
    "    rec = treino[treino['date'] >= '2016-01-01']\n",
    "    p = prev[prev['series'] == serie].sort_values('date')\n",
    "    ax.plot(rec['date'], rec[serie], label='Treino real', color='tab:blue', alpha=0.6)\n",
    "    ax.plot(val['date'], val[serie], label='Validação real', color='tab:green', alpha=0.8)\n",
    "    ax.plot(p['date'], p['yhat'], label='Boosting', color='tab:orange', linewidth=2, linestyle='--')\n",
    "    ax.axvline(pd.Timestamp('2016-03-28'), color='gray', linestyle=':')\n",
    "    ax.set_title(f'Série: {serie}'); ax.set_ylabel('Vendas'); ax.legend(); ax.grid(True, alpha=0.3)\n",
    "fig.tight_layout(); plt.show()\n",
]})

cells.append({"cell_type": "markdown", "metadata": {}, "source": [
    "## 7. Comparação final — todos os modelos\n",
]})
cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": [
    "import pandas as pd\n",
    "import matplotlib.pyplot as plt\n",
    "treino = pd.read_csv('dados/treino.csv', parse_dates=['date'])\n",
    "val = pd.read_csv('dados/validacao.csv', parse_dates=['date'])\n",
    "prev = pd.read_csv('saidas/previsoes_validacao.csv', parse_dates=['date'])\n",
    "cores = {'linear': 'tab:red', 'boosting': 'tab:orange', 'sarima': 'tab:purple', 'naive_sazonal': 'tab:gray'}\n",
    "series = ['store_total', 'FOODS', 'HOBBIES']\n",
    "fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(12, 14))\n",
    "for ax, serie in zip(axes, series):\n",
    "    rec = treino[treino['date'] >= '2016-01-01']\n",
    "    ax.plot(rec['date'], rec[serie], label='Treino real', color='tab:blue', alpha=0.5)\n",
    "    ax.plot(val['date'], val[serie], label='Validação real', color='tab:green', alpha=0.9, linewidth=2)\n",
    "    for modelo, cor in cores.items():\n",
    "        p = prev[(prev['series'] == serie) & (prev['modelo'] == modelo)].sort_values('date')\n",
    "        if not p.empty:\n",
    "            ax.plot(p['date'], p['yhat'], label=modelo, color=cor, linewidth=1.5, linestyle='--')\n",
    "    ax.axvline(pd.Timestamp('2016-03-28'), color='gray', linestyle=':')\n",
    "    ax.set_title(f'Série: {serie}'); ax.set_ylabel('Vendas'); ax.legend(); ax.grid(True, alpha=0.3)\n",
    "fig.tight_layout(); plt.show()\n",
]})

cells.append({"cell_type": "markdown", "metadata": {}, "source": [
    "## 8. Tabela de métricas — todos os modelos\n",
]})
cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": [
    "import pandas as pd\n",
    "met = pd.read_csv('saidas/metricas.csv')\n",
    "met.sort_values(['series', 'mae']).reset_index(drop=True)\n",
]})

nb["cells"].extend(cells)
nb_path.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
print("ok")
