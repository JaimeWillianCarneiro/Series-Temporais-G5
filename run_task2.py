#!/usr/bin/env python3
"""Task 2 — linear (OLS/Lasso/Ridge), árvore (boosting) e rede (MLP/RNN/LSTM), recursivos, a partir de dados/.

Roda depois de `run.py` (Task 1), que grava as linhas de baselines e SARIMA em
metricas.csv / previsoes_validacao.csv. Este script acrescenta `linear`, `boosting`
e `rede` sem tocar nas linhas da Task 1 e copia as duas tabelas para saidas/.

Mesma lógica de analise_task2.ipynb:
  - features: lags 1, 7, 14, 28 do próprio y; dia da semana; snap_CA; dummies de
    event_name_1/2 e event_type_1/2; os três preços semanais padronizados com
    média/desvio do treino (na validação interna, do treino interno);
  - estratégia recursiva: a previsão de t+1 volta como lag; o y da validação nunca entra;
  - hiperparâmetros: modelo linear (OLS, Lasso ou Ridge, com lags padronizados no treino) e alpha, escolha entre RandomForest / GradientBoosting /
    HistGradientBoosting e grade de redes neurais (MLP / RNN / LSTM) por MAE numa
    validação interna com origem 2016-02-28 (prevê 2016-02-29..2016-03-27, dentro
    do treino). Os 28 dias de validação só publicam.
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import sys
from itertools import product
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.ensemble import GradientBoostingRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from ts_utils import RAIZ, SERIES, calcular_metricas, carregar_dados  # noqa: E402

ORIGEM = pd.Timestamp("2016-03-27")
ORIGEM_INTERNA = pd.Timestamp("2016-02-28")
LAGS = [1, 7, 14, 28]
ALPHAS_LASSO = [0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000]
ALPHAS_RIDGE = [0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000]
COLS_NAO_FEATURE = ["store_total", "FOODS", "HOBBIES", "date", "preco_store_total", "preco_FOODS", "preco_HOBBIES"]
CANDIDATOS_ARVORE = {
    "RandomForest": lambda: RandomForestRegressor(n_estimators=300, max_depth=8, random_state=42, n_jobs=1),
    "GradientBoosting": lambda: GradientBoostingRegressor(
        n_estimators=300, max_depth=3, learning_rate=0.05, random_state=42
    ),
    "HistGradientBoosting": lambda: HistGradientBoostingRegressor(
        max_iter=300, max_depth=6, learning_rate=0.05, random_state=42
    ),
}

GRID_HIDDEN_SIZE = [16, 32, 64]
GRID_NUM_LAYERS = [1, 2]
GRID_LR = [0.001, 0.005, 0.01]
GRID_WEIGHT_DECAY = [0, 1e-4]
DROPOUT_RATE = 0.2


def set_seeds(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dropout_rate=0.2):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc3 = nn.Linear(hidden_size // 2, output_size)

    def forward(self, x):
        out = self.dropout(self.relu(self.fc1(x)))
        out = self.relu(self.fc2(out))
        return self.fc3(out)


class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout_rate=0.2):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = nn.RNN(
            input_size,
            hidden_size,
            num_layers,
            batch_first=True,
            dropout=dropout_rate if num_layers > 1 else 0,
        )
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        self.rnn.flatten_parameters()
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size, device=x.device)
        out, _ = self.rnn(x, h0)
        return self.fc(out[:, -1, :])


class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout_rate=0.2):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers,
            batch_first=True,
            dropout=dropout_rate if num_layers > 1 else 0,
        )
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        self.lstm.flatten_parameters()
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])


def torch_pandas_decorator(funcao):
    def wrapper(*args, **kwargs):
        if (
            funcao.__name__ == "fit"
            and len(args) > 1
            and hasattr(args[0], "feature_names_in_")
            and isinstance(args[1], pd.DataFrame)
        ):
            args[0].feature_names_in_ = args[1].columns.tolist()

        novos_args = tuple(
            torch.tensor(arg.astype("float32").values, dtype=torch.float32)
            if isinstance(arg, (pd.DataFrame, pd.Series))
            else arg
            for arg in args
        )
        novos_kwargs = {
            k: (
                torch.tensor(v.astype("float32").values, dtype=torch.float32)
                if isinstance(v, (pd.DataFrame, pd.Series))
                else v
            )
            for k, v in kwargs.items()
        }
        resultado = funcao(*novos_args, **novos_kwargs)
        if isinstance(resultado, torch.Tensor):
            resultado = resultado.to("cpu").detach().numpy()
            if resultado.ndim == 2 and resultado.shape[1] == 1:
                resultado = resultado.flatten()
        return resultado

    return wrapper


class TorchAdapter:
    def __init__(self, model: nn.Module, criterion: nn.Module, optimizer: torch.optim.Optimizer) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.criterion = criterion
        self.optimizer = optimizer
        self.feature_names_in_: list = None
        self.x_mean = None
        self.x_std = None
        self.y_mean = None
        self.y_std = None

    @torch_pandas_decorator
    def fit(self, X: torch.Tensor, y: torch.Tensor, epochs: int = 150, batch_size: int = 64) -> "TorchAdapter":
        X = X.to(self.device)
        y = y.to(self.device)

        self.x_mean = torch.zeros((1, X.shape[1]), dtype=torch.float32, device=self.device)
        self.x_std = torch.ones((1, X.shape[1]), dtype=torch.float32, device=self.device)
        idx_lags = [i for i, col in enumerate(self.feature_names_in_) if str(col).startswith("lag_")]
        self.x_mean[:, idx_lags] = X[:, idx_lags].mean(dim=0, keepdim=True)
        self.x_std[:, idx_lags] = X[:, idx_lags].std(dim=0, keepdim=True) + 1e-8
        X = (X - self.x_mean) / self.x_std

        if y.dim() == 1 or (y.dim() == 2 and y.shape[1] != 1):
            y = y.view(-1, 1)

        self.y_mean = y.mean(dim=0, keepdim=True)
        self.y_std = y.std(dim=0, keepdim=True)
        self.y_std[self.y_std == 0] = 1.0
        y = (y - self.y_mean) / self.y_std

        is_rnn = hasattr(self.model, "rnn") or hasattr(self.model, "lstm")
        if is_rnn and X.dim() == 2:
            X = X.unsqueeze(1)

        dataset = TensorDataset(X, y)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        self.model.train()
        for _ in range(epochs):
            for bx, by in loader:
                output = self.model(bx)
                loss = self.criterion(output, by)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
        return self

    @torch_pandas_decorator
    def predict(self, X: torch.Tensor) -> torch.Tensor:
        X = X.to(self.device)
        X = (X - self.x_mean) / self.x_std
        is_rnn = hasattr(self.model, "rnn") or hasattr(self.model, "lstm")
        if is_rnn and X.dim() == 2:
            X = X.unsqueeze(1)
        self.model.eval()
        with torch.no_grad():
            previsao_normalizada = self.model(X)
        return (previsao_normalizada * self.y_std) + self.y_mean


def criar_adaptador(input_size, classe_rede, is_recorrente, h, l, taxa_lr, wd, crit):
    set_seeds(42)
    if is_recorrente:
        modelo = classe_rede(
            input_size=input_size, hidden_size=h, num_layers=l, output_size=1, dropout_rate=DROPOUT_RATE
        )
    else:
        modelo = classe_rede(input_size=input_size, hidden_size=h, output_size=1, dropout_rate=DROPOUT_RATE)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    modelo = modelo.to(device)
    otimizador = torch.optim.Adam(modelo.parameters(), lr=taxa_lr, weight_decay=wd)
    return TorchAdapter(modelo, crit, otimizador)


def montar_candidatos_rede(input_size):
    return {
        "RNN_h32_l2_lr0.005_wd0_L1": lambda: criar_adaptador(
            input_size, RNN, True, h=32, l=2, taxa_lr=0.005, wd=0, crit=nn.L1Loss()
        ),
        "LSTM_h16_l2_lr0.01_wd0_MSE": lambda: criar_adaptador(
            input_size, LSTM, True, h=16, l=2, taxa_lr=0.01, wd=0, crit=nn.MSELoss()
        ),
        "RNN_h16_l2_lr0.01_wd0_L1": lambda: criar_adaptador(
            input_size, RNN, True, h=16, l=2, taxa_lr=0.01, wd=0, crit=nn.L1Loss()
        ),
    }


def montar_linear(estimador):
    """Padroniza só os lag_* (média/desvio do treino do próprio ajuste) e aplica o estimador.

    Dummies ficam em 0/1 e os preços já vêm padronizados com o treino; assim a
    penalidade de Lasso/Ridge pesa igual em todas as colunas.
    """
    pre = ColumnTransformer(
        [("lags", StandardScaler(), [f"lag_{l}" for l in LAGS])],
        remainder="passthrough",
        verbose_feature_names_out=False,
    )
    return Pipeline([("pre", pre), ("reg", estimador)])


def carregar_covariaveis():
    cov = pd.read_csv(RAIZ / "dados" / "covariaveis.csv", parse_dates=["date"])
    cov["dia_semana"] = cov["date"].dt.dayofweek
    return pd.get_dummies(cov, columns=["dia_semana", "event_name_1", "event_type_1", "event_name_2", "event_type_2"])


def padronizar_precos(cov, ate):
    """Padroniza preco_<serie> com média/desvio só das datas <= ate."""
    cov = cov.copy()
    stats = {}
    ref = cov["date"] <= ate
    for s in SERIES:
        m, d = cov.loc[ref, f"preco_{s}"].mean(), cov.loc[ref, f"preco_{s}"].std()
        stats[s] = (m, d)
        cov[f"preco_{s}_padronizada"] = (cov[f"preco_{s}"] - m) / d
    return cov, stats


def montar_treino(df_y, cov, serie):
    """Lags reais de y (só no ajuste) + covariáveis do dia."""
    d = df_y[["date", serie]].sort_values("date").copy()
    for lag in LAGS:
        d[f"lag_{lag}"] = d[serie].shift(lag)
    d = d.dropna().merge(cov, on="date", how="inner")
    return d.drop(columns=COLS_NAO_FEATURE, errors="ignore"), d[serie]


def prever_recursivo(modelo, y_hist, cov_fut):
    """A previsão de t+1 entra como lag nas seguintes; só o histórico do treino é real."""
    feats = list(modelo.feature_names_in_)
    base = cov_fut.reset_index(drop=True).reindex(columns=feats, fill_value=0).astype(float)
    ys = list(map(float, y_hist))
    preds = []
    for i in range(len(base)):
        linha = base.iloc[[i]].copy()
        for lag in LAGS:
            linha[f"lag_{lag}"] = ys[-lag]
        p = float(modelo.predict(linha[feats])[0])
        preds.append(p)
        ys.append(p)
    return np.array(preds)

def main():
    treino, val = carregar_dados()
    treino = treino.sort_values("date").reset_index(drop=True)
    assert treino["date"].max() == ORIGEM and len(val) == 28
    cov = carregar_covariaveis()

    # validação interna (dentro do treino)
    cov_int, _ = padronizar_precos(cov, ORIGEM_INTERNA)
    hist_int = treino[treino["date"] <= ORIGEM_INTERNA]
    alvo_int = treino[treino["date"] > ORIGEM_INTERNA]
    cov_int_fut = cov_int[cov_int["date"].isin(alvo_int["date"])].sort_values("date")

    cov_fin, stats = padronizar_precos(cov, ORIGEM)
    cov_val = cov_fin[cov_fin["date"].isin(val["date"])].sort_values("date")
    assert (cov_val["date"].to_numpy() == val["date"].to_numpy()).all()

    X_exemplo, _ = montar_treino(hist_int, cov_int, SERIES[0])
    candidatos_rede = montar_candidatos_rede(X_exemplo.shape[1])

    escolhas, linhas = [], []
    for s in SERIES:
        X_int, y_int = montar_treino(hist_int, cov_int, s)
        y_alvo = alvo_int[s].to_numpy()
        h_int = hist_int[s].to_numpy()

        # --- Validação Interna: OLS, Lasso e Ridge ---
        ols = montar_linear(LinearRegression()).fit(X_int, y_int)
        mae_ols = mean_absolute_error(y_alvo, prever_recursivo(ols, h_int, cov_int_fut))

        erros_lasso = {
            a: mean_absolute_error(
                y_alvo, prever_recursivo(
                    montar_linear(Lasso(alpha=a, max_iter=100000)).fit(X_int, y_int), h_int, cov_int_fut
                )
            ) for a in ALPHAS_LASSO
        }
        melhor_alpha_lasso = min(erros_lasso, key=erros_lasso.get)
        mae_lasso = erros_lasso[melhor_alpha_lasso]

        erros_ridge = {
            a: mean_absolute_error(
                y_alvo, prever_recursivo(montar_linear(Ridge(alpha=a)).fit(X_int, y_int), h_int, cov_int_fut)
            ) for a in ALPHAS_RIDGE
        }
        melhor_alpha_ridge = min(erros_ridge, key=erros_ridge.get)
        mae_ridge = erros_ridge[melhor_alpha_ridge]

        # Escolha do vencedor linear
        candidatos_lineares = {
            'OLS': (mae_ols, None),
            'Lasso': (mae_lasso, melhor_alpha_lasso),
            'Ridge': (mae_ridge, melhor_alpha_ridge)
        }
        tipo_linear = min(candidatos_lineares, key=lambda k: candidatos_lineares[k][0])
        mae_linear, alpha_linear = candidatos_lineares[tipo_linear]

        # --- Validação Interna: Árvore e Rede ---
        mae_arv = {
            n: mean_absolute_error(y_alvo, prever_recursivo(c().fit(X_int, y_int), h_int, cov_int_fut))
            for n, c in CANDIDATOS_ARVORE.items()
        }
        arvore = min(mae_arv, key=mae_arv.get)

        mae_rede = {
            n: mean_absolute_error(y_alvo, prever_recursivo(c().fit(X_int, y_int), h_int, cov_int_fut))
            for n, c in candidatos_rede.items()
        }
        rede = min(mae_rede, key=mae_rede.get)

        escolhas.append({
            "series": s,
            "tipo_linear": tipo_linear,
            "alpha_linear": alpha_linear,
            "mae_int_linear": mae_linear,
            "arvore": arvore,
            "rede": rede,
            "mae_int_rede": mae_rede[rede],
            **{f"mae_int_{n}": v for n, v in mae_arv.items()},
        })
        print(
            f"{s}: linear={tipo_linear} (a={alpha_linear}) | árvore={arvore} | rede={rede}\n"
            f"   MAE int: linear={round(mae_linear,1)} | rede={round(mae_rede[rede], 1)} | "
            f"árvores={ {n: round(v, 1) for n, v in mae_arv.items()} }"
        )

        # --- Ajuste definitivo em todo o treino ---
        X, y = montar_treino(treino, cov_fin, s)
        
        if tipo_linear == 'OLS':
            clf_linear = montar_linear(LinearRegression()).fit(X, y)
        elif tipo_linear == 'Lasso':
            clf_linear = montar_linear(Lasso(alpha=alpha_linear, max_iter=100000)).fit(X, y)
        elif tipo_linear == 'Ridge':
            clf_linear = montar_linear(Ridge(alpha=alpha_linear)).fit(X, y)

        modelos = {
            "linear": clf_linear,
            "boosting": CANDIDATOS_ARVORE[arvore]().fit(X, y),
            "rede": candidatos_rede[rede]().fit(X, y),
        }
        
        joblib.dump(
            {"modelo": modelos["linear"], "tipo_modelo": tipo_linear, "alpha": alpha_linear, "lags": LAGS, "stats_precos": stats},
            RAIZ / f"modelo_{s}.pkl",
        )
        joblib.dump(
            {"modelo": modelos["boosting"], "nome": arvore, "lags": LAGS, "stats_precos": stats},
            RAIZ / f"modelo_arvore_{s}.pkl",
        )
        joblib.dump(
            {"modelo": modelos["rede"], "arquitetura_vencedora": rede, "lags": LAGS, "stats_precos": stats},
            RAIZ / f"modelo_rede_{s}.pkl",
        )
        
        for nome, m in modelos.items():
            yhat = prever_recursivo(m, treino[s].to_numpy(), cov_val)
            linhas += [
                {"date": d.strftime("%Y-%m-%d"), "series": s, "modelo": nome, "yhat": v}
                for d, v in zip(cov_val["date"], yhat)
            ]

    novos = pd.DataFrame(linhas)
    (RAIZ / "saidas").mkdir(exist_ok=True)
    pd.DataFrame(escolhas).to_csv(RAIZ / "saidas" / "escolhas_task2.csv", index=False)

    # acrescenta às tabelas da Task 1 (gravadas por run.py) sem mexer nas linhas dela
    prev = pd.read_csv(RAIZ / "previsoes_validacao.csv")
    prev = prev[~prev["modelo"].isin(novos["modelo"].unique())]
    assert {"naive_sazonal", "sarima"} <= set(prev["modelo"]), "rode python run.py antes"
    prev = pd.concat([prev, novos], ignore_index=True)

    met = pd.read_csv(RAIZ / "metricas.csv")
    met = met[~met["modelo"].isin(novos["modelo"].unique())]
    novas_met = []
    for s in SERIES:
        for nome in novos["modelo"].unique():
            p = novos[(novos.series == s) & (novos.modelo == nome)]
            j = val[["date", s]].assign(date=val["date"].dt.strftime("%Y-%m-%d")).merge(p, on="date")
            assert len(j) == 28
            mae, rmse, mase = calcular_metricas(treino[s].to_numpy(), j[s].to_numpy(), j["yhat"].to_numpy())
            novas_met.append({"series": s, "modelo": nome, "mae": mae, "rmse": rmse, "mase": mase})
    met = pd.concat([met, pd.DataFrame(novas_met)], ignore_index=True)

    for pasta in (RAIZ, RAIZ / "saidas"):
        prev.to_csv(pasta / "previsoes_validacao.csv", index=False)
        met.to_csv(pasta / "metricas.csv", index=False)
    print(met.round(3).to_string(index=False))
    print("\nConcluído: metricas.csv, previsoes_validacao.csv (raiz e saidas/), saidas/escolhas_task2.csv")

if __name__ == "__main__":
    main()