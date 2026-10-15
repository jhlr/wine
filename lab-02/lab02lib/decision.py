"""D4-D6: custos, busca de limiar na validação, fórmula, calibração e congelamento da decisão."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss
from torch import nn

from lab02lib.data import MeuDataset

# Custos hipotéticos do exercício, em "pontos de esforço". A base não tem custos reais.
C_FP: int = 1
C_FN: int = 5
REGRA_EMPATE = (
    "entre limiares de custo mínimo igual, o mais próximo de 0,5; "
    "se ainda empatar, o menor. Positivo se escore >= limiar."
)


def prever(escores: np.ndarray, limiar: float) -> np.ndarray:
    """Regra de decisão única do laboratório: positivo se ``escore >= limiar`` (comparação em float64)."""
    return (escores.astype(np.float64) >= float(limiar)).astype(int)


def contar_erros(rotulos: np.ndarray, escores: np.ndarray, limiar: float) -> tuple[int, int]:
    """Falsos positivos e falsos negativos de um limiar."""
    previsto = prever(escores, limiar)
    return int(((previsto == 1) & (rotulos == 0)).sum()), int(((previsto == 0) & (rotulos == 1)).sum())


def custo_total(fp: int, fn: int, c_fp: float = C_FP, c_fn: float = C_FN) -> float:
    """``custo = C_FP × FP + C_FN × FN``; acertos custam zero."""
    return float(c_fp * fp + c_fn * fn)


@dataclass
class BuscaLimiar:
    """Resultado da busca: a tabela de candidatos e o limiar escolhido."""

    tabela: pd.DataFrame
    limiar: float
    custo_total: float
    custo_por_obs: float
    n_empatados: int


def buscar_limiar(rotulos: np.ndarray, escores: np.ndarray, c_fp: float = C_FP, c_fn: float = C_FN) -> BuscaLimiar:
    """Busca exata sobre todos os escores únicos mais o extremo "tudo negativo".

    Os candidatos são valores de escore (a borda superior de cada intervalo de
    limiares equivalentes). O escore mínimo já cobre "tudo positivo"; um número
    logo acima do máximo (``np.nextafter``) cobre "tudo negativo". Custo vetorizado.
    Desempate: ver ``REGRA_EMPATE`` (decidida antes de olhar os dados).
    """
    unicos = np.unique(escores)
    candidatos = np.concatenate([unicos, np.nextafter(unicos[-1:], np.array([np.inf], dtype=unicos.dtype))]).astype(np.float64)
    positivo = escores.astype(np.float64)[None, :] >= candidatos[:, None]
    real_neg = rotulos[None, :] == 0
    fp = (positivo & real_neg).sum(axis=1)
    fn = (~positivo & ~real_neg).sum(axis=1)
    custo = c_fp * fp + c_fn * fn
    empatados = candidatos[custo == custo.min()]
    ordem = np.lexsort((empatados, np.abs(empatados - 0.5)))
    escolhido = float(empatados[ordem[0]])
    tabela = pd.DataFrame(
        {"limiar": candidatos, "fp": fp, "fn": fn, "custo_total": custo, "custo_por_obs": custo / len(rotulos)}
    )
    return BuscaLimiar(
        tabela=tabela,
        limiar=escolhido,
        custo_total=float(custo.min()),
        custo_por_obs=float(custo.min() / len(rotulos)),
        n_empatados=int(len(empatados)),
    )


def comparar_regras(
    rotulos: np.ndarray, escores: np.ndarray, busca: BuscaLimiar, c_fp: float = C_FP, c_fn: float = C_FN
) -> pd.DataFrame:
    """Custos na validação: limiar escolhido, 0,5, fórmula, tudo positivo e tudo negativo."""
    n = len(rotulos)
    formula = c_fp / (c_fp + c_fn)
    linhas = []
    for nome, limiar in (("escolhido na busca", busca.limiar), ("padrão 0,5", 0.5), ("fórmula C_FP/(C_FP+C_FN)", formula)):
        fp, fn = contar_erros(rotulos, escores, limiar)
        total = custo_total(fp, fn, c_fp, c_fn)
        linhas.append({"regra": nome, "limiar": limiar, "fp": fp, "fn": fn, "custo_total": total, "custo_por_obs": total / n})
    n_neg = int((rotulos == 0).sum())
    n_pos = int((rotulos == 1).sum())
    linhas.append({"regra": "tudo positivo", "limiar": float("nan"), "fp": n_neg, "fn": 0, "custo_total": c_fp * n_neg, "custo_por_obs": c_fp * n_neg / n})
    linhas.append({"regra": "tudo negativo", "limiar": float("nan"), "fp": 0, "fn": n_pos, "custo_total": c_fn * n_pos, "custo_por_obs": c_fn * n_pos / n})
    return pd.DataFrame(linhas)


def grafico_custo(busca: BuscaLimiar, formula: float, caminho: Path) -> Figure:
    """Custo total na validação em função do limiar (degraus), com escolhido, 0,5 e fórmula marcados."""
    t = busca.tabela
    fig = Figure(figsize=(8, 4.5), constrained_layout=True)
    eixo = fig.subplots()
    eixo.step(t["limiar"], t["custo_total"], where="post", color="tab:blue")
    eixo.axvline(busca.limiar, color="tab:green", linestyle="--", label=f"escolhido ({busca.limiar:.4f})")
    eixo.axvline(0.5, color="tab:gray", linestyle=":", label="0,5")
    eixo.axvline(formula, color="tab:red", linestyle="-.", label=f"fórmula ({formula:.4f})")
    eixo.set_xlabel("limiar (escore >= limiar prevê positivo)")
    eixo.set_ylabel("custo total na validação (pontos de esforço)")
    eixo.set_title("D5: custo na validação por limiar")
    eixo.legend()
    fig.savefig(caminho, dpi=130)
    return fig


def calibracao(
    rotulos: np.ndarray, escores: np.ndarray, caminho: Path, n_bins: int = 5
) -> tuple[pd.DataFrame, float, Figure]:
    """Tabela do diagrama de calibração (bins por quantis), Brier score e a figura.

    Com poucas dezenas de positivos na validação, poucos bins (5) mantêm alguns
    positivos em cada um; mais bins deixariam proporções instáveis.
    """
    freq_real, media_prevista = calibration_curve(rotulos, escores, n_bins=n_bins, strategy="quantile")
    brier = float(brier_score_loss(rotulos, escores))
    tabela = pd.DataFrame({"bin": np.arange(1, len(freq_real) + 1), "escore_medio": media_prevista, "fracao_positivos": freq_real})
    fig = Figure(figsize=(5.5, 5), constrained_layout=True)
    eixo = fig.subplots()
    eixo.plot([0, 1], [0, 1], linestyle="--", color="tab:gray", label="calibração perfeita")
    eixo.plot(media_prevista, freq_real, marker="o", color="tab:blue", label=f"validação ({len(freq_real)} bins por quantis)")
    eixo.set_xlabel("escore médio previsto no bin")
    eixo.set_ylabel("fração observada de positivos")
    eixo.set_title(f"D6: calibração na validação (Brier {brier:.4f})")
    eixo.legend()
    fig.savefig(caminho, dpi=130)
    return tabela, brier, fig


def hash_pesos(modelo: nn.Module) -> str:
    """SHA-256 dos pesos: nomes e bytes de cada tensor do ``state_dict``, em ordem alfabética."""
    h = hashlib.sha256()
    estado = modelo.state_dict()
    for nome in sorted(estado):
        h.update(nome.encode("utf-8"))
        h.update(estado[nome].detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def congelar(
    caminho: Path,
    modelo: nn.Module,
    limiar: float,
    ds_treino: MeuDataset,
    features: list[str],
    melhor_epoca: int,
    semente: int,
) -> dict[str, object]:
    """Grava a decisão antes de abrir o teste: limiar, regra de empate, custos, entradas, estatísticas, modelo e hash."""
    decisao: dict[str, object] = {
        "limiar": float(limiar),
        "regra_empate": REGRA_EMPATE,
        "C_FP": C_FP,
        "C_FN": C_FN,
        "unidade_custo": "pontos de esforço (hipotéticos)",
        "caracteristicas": list(features),
        "normalizacao_media_treino": [float(v) for v in ds_treino.media],
        "normalizacao_desvio_treino_ddof0": [float(v) for v in ds_treino.desvio],
        "arquitetura": str(modelo),
        "melhor_epoca": int(melhor_epoca),
        "sha256_pesos_restaurados": hash_pesos(modelo),
        "semente": int(semente),
    }
    caminho.write_text(json.dumps(decisao, indent=2, ensure_ascii=False), encoding="utf-8")
    return decisao


def verificar_congelamento(caminho: Path, modelo: nn.Module, limiar: float) -> None:
    """Confere que o modelo e o limiar de agora são os do arquivo congelado (levanta erro se não forem)."""
    decisao = json.loads(caminho.read_text(encoding="utf-8"))
    assert hash_pesos(modelo) == decisao["sha256_pesos_restaurados"], "os pesos mudaram depois do congelamento"
    assert float(limiar) == decisao["limiar"], "o limiar mudou depois do congelamento"
