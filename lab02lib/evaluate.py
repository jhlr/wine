"""D1-D3: avaliação final no teste, baseline trivial e curvas ROC e PR."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from sklearn.metrics import (
    accuracy_score,
    auc,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from torch import Tensor, nn
from torch.utils.data import DataLoader

from apoio import obter_escores
from lab02lib.decision import C_FN, C_FP, custo_total, prever

NAO_REPORTADA = "não reportada"


class EscoresUmaVez:
    """Entrega os escores do teste uma única vez e conta as chamadas.

    Uma segunda chamada levanta erro: assim o teste não vira um laço de
    tentativa e erro. ``chamadas`` mostra quantas vezes o teste foi percorrido.
    """

    def __init__(self, modelo: nn.Module, loader: DataLoader[tuple[Tensor, Tensor]], dispositivo: str = "cpu") -> None:
        self.modelo = modelo
        self.loader = loader
        self.dispositivo = dispositivo
        self.chamadas = 0

    def obter(self) -> tuple[np.ndarray, np.ndarray]:
        """Devolve ``(rótulos, escores)`` do teste, só na primeira chamada."""
        if self.chamadas >= 1:
            raise RuntimeError("Os escores do teste já foram obtidos; não é permitido olhar o teste de novo.")
        self.chamadas += 1
        return obter_escores(self.modelo, self.loader, self.dispositivo)


def metricas_regra(rotulos: np.ndarray, previsto: np.ndarray) -> dict[str, float]:
    """Acurácia, precisão, revocação, F1 e custo por observação de uma regra já aplicada.

    Convenção: se nenhuma previsão for positiva, precisão = 0 (``zero_division=0``).
    """
    matriz = confusion_matrix(rotulos, previsto, labels=[0, 1])
    fp, fn = int(matriz[0, 1]), int(matriz[1, 0])
    return {
        "acuracia": float(accuracy_score(rotulos, previsto)),
        "precisao": float(precision_score(rotulos, previsto, zero_division=0)),
        "revocacao": float(recall_score(rotulos, previsto, zero_division=0)),
        "f1": float(f1_score(rotulos, previsto, zero_division=0)),
        "custo_por_obs": custo_total(fp, fn, C_FP, C_FN) / len(rotulos),
        "fp": fp,
        "fn": fn,
    }


def avaliar_teste(
    rotulos: np.ndarray, escores: np.ndarray, rotulos_treino: np.ndarray, limiar_escolhido: float
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float]]:
    """Tabela final (3 linhas), matrizes de confusão (formato longo) e áreas sob as curvas.

    As áreas vêm dos escores, não das classes. AUC-PR é a área trapezoidal
    ``auc(recall, precision)``; AP (``average_precision_score``) é reportada à parte.
    O baseline prevê a classe majoritária do treino para todo o teste.
    """
    precisao, revocacao, _ = precision_recall_curve(rotulos, escores)
    areas = {
        "auc_roc": float(roc_auc_score(rotulos, escores)),
        "auc_pr": float(auc(revocacao, precisao)),
        "ap": float(average_precision_score(rotulos, escores)),
        "prevalencia_teste": float(rotulos.mean()),
    }
    classe_majoritaria = int(rotulos_treino.mean() >= 0.5)
    previstos = {
        "Trivial do treino": np.full(len(rotulos), classe_majoritaria),
        "Rede: padrão": prever(escores, 0.5),
        "Rede: escolhido na validação": prever(escores, limiar_escolhido),
    }
    limiares: dict[str, float | str] = {"Trivial do treino": "não se aplica", "Rede: padrão": 0.5, "Rede: escolhido na validação": limiar_escolhido}
    linhas = []
    matrizes = []
    for regra, previsto in previstos.items():
        m = metricas_regra(rotulos, previsto)
        trivial = regra == "Trivial do treino"
        linhas.append(
            {
                "regra": regra,
                "limiar": limiares[regra],
                "acuracia": m["acuracia"],
                "precisao": m["precisao"],
                "revocacao": m["revocacao"],
                "f1": m["f1"],
                "auc_roc": NAO_REPORTADA if trivial else areas["auc_roc"],
                "auc_pr": NAO_REPORTADA if trivial else areas["auc_pr"],
                "custo_por_obs": m["custo_por_obs"],
                "fp": m["fp"],
                "fn": m["fn"],
            }
        )
        matriz = confusion_matrix(rotulos, previsto, labels=[0, 1])
        for real in (0, 1):
            for pred in (0, 1):
                matrizes.append({"regra": regra, "real": real, "previsto": pred, "contagem": int(matriz[real, pred])})
    return pd.DataFrame(linhas), pd.DataFrame(matrizes), areas


def matriz_rotulada(matrizes: pd.DataFrame, regra: str) -> pd.DataFrame:
    """Matriz de confusão com linhas = real e colunas = previsto, na ordem 0 e 1."""
    sub = matrizes[matrizes["regra"] == regra].pivot(index="real", columns="previsto", values="contagem")
    sub.index = pd.Index(["real 0 (nota < 7)", "real 1 (nota >= 7)"], name="regra: " + regra)
    sub.columns = pd.Index(["previsto 0", "previsto 1"])
    return sub


def grafico_roc_pr(
    rotulos: np.ndarray, escores: np.ndarray, limiar: float, areas: dict[str, float], caminho: Path
) -> Figure:
    """ROC (com a diagonal aleatória) e PR (com a linha da prevalência do teste), lado a lado."""
    fpr, tpr, _ = roc_curve(rotulos, escores)
    precisao, revocacao, _ = precision_recall_curve(rotulos, escores)
    fig = Figure(figsize=(11, 4.8), constrained_layout=True)
    eixo_roc, eixo_pr = fig.subplots(1, 2)
    eixo_roc.plot(fpr, tpr, color="tab:blue", label=f"rede (AUC-ROC = {areas['auc_roc']:.3f})")
    eixo_roc.plot([0, 1], [0, 1], linestyle="--", color="tab:gray", label="aleatório (diagonal, área 0,5)")
    eixo_pr.plot(revocacao, precisao, color="tab:blue", label=f"rede (AUC-PR trapezoidal = {areas['auc_pr']:.3f}; AP = {areas['ap']:.3f})")
    eixo_pr.axhline(areas["prevalencia_teste"], linestyle="--", color="tab:gray", label=f"prevalência do teste ({areas['prevalencia_teste']:.3f})")
    for nome, t, cor, marca in (("limiar 0,5", 0.5, "tab:red", "o"), (f"limiar escolhido ({limiar:.3f})", limiar, "tab:green", "s")):
        previsto = prever(escores, t)
        pos = rotulos == 1
        tpr_t = float(previsto[pos].mean())
        fpr_t = float(previsto[~pos].mean())
        eixo_roc.scatter([fpr_t], [tpr_t], color=cor, marker=marca, zorder=3, label=nome)
        prec_t = float(rotulos[previsto == 1].mean()) if previsto.any() else 0.0
        eixo_pr.scatter([tpr_t], [prec_t], color=cor, marker=marca, zorder=3, label=nome)
    eixo_roc.set_xlabel("taxa de falsos positivos")
    eixo_roc.set_ylabel("taxa de verdadeiros positivos (revocação)")
    eixo_roc.set_title("D3: curva ROC no teste")
    eixo_roc.legend(loc="lower right")
    eixo_pr.set_xlabel("revocação")
    eixo_pr.set_ylabel("precisão")
    eixo_pr.set_title("D3: curva precisão-revocação no teste")
    eixo_pr.set_ylim(0, 1.02)
    eixo_pr.legend(loc="lower left", bbox_to_anchor=(0.0, 0.17))
    fig.savefig(caminho, dpi=130)
    return fig
