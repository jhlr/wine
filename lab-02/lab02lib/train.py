"""D7: treino com parada antecipada, guardando e restaurando o melhor estado."""
from __future__ import annotations

import copy
import inspect
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import torch
from matplotlib.figure import Figure
from torch import Tensor, nn
from torch.utils.data import DataLoader

from apoio import avaliar_perda, treinar_epoca


@dataclass
class ResultadoTreino:
    """Tudo o que o laço de treino produziu.

    ``historico`` tem uma linha por época (``epoca``, ``perda_treino``,
    ``perda_validacao``). ``melhor_estado`` é a cópia independente dos pesos da
    melhor época; o modelo recebido já volta com esses pesos restaurados.
    """

    historico: pd.DataFrame
    melhor_epoca: int
    epoca_parada: int
    melhor_perda: float
    motivo: str
    melhor_estado: dict[str, Tensor]


def treinar_com_parada(
    modelo: nn.Module,
    loader_treino: DataLoader[tuple[Tensor, Tensor]],
    loader_validacao: DataLoader[tuple[Tensor, Tensor]],
    max_epocas: int = 100,
    paciencia: int = 10,
    min_delta: float = 0.0001,
    dispositivo: str = "cpu",
    lr: float = 1e-3,
) -> ResultadoTreino:
    """Treina com Adam e ``BCEWithLogitsLoss`` e para pela perda de validação.

    A cada época: treina, avalia a validação sem gradientes e compara. Há
    melhora quando ``perda_val < melhor - min_delta``; então guarda uma cópia
    independente (``copy.deepcopy``) dos pesos, a época e a perda, e zera a
    contagem de paciência. Sem melhora por ``paciencia`` épocas seguidas, para.
    Ao sair, restaura no ``modelo`` os melhores pesos. Esta função não recebe o
    carregador de teste, de propósito: o teste não pode influenciar a parada.
    """
    criterio = nn.BCEWithLogitsLoss()
    otimizador = torch.optim.Adam(modelo.parameters(), lr=lr)
    melhor_perda = float("inf")
    melhor_epoca = 0
    melhor_estado: dict[str, Tensor] = {}
    sem_melhora = 0
    motivo = "max_epocas"
    linhas: list[dict[str, float | int]] = []
    epoca = 0
    for epoca in range(1, max_epocas + 1):
        perda_treino = treinar_epoca(modelo, loader_treino, otimizador, criterio, dispositivo)
        perda_val = avaliar_perda(modelo, loader_validacao, criterio, dispositivo)
        linhas.append({"epoca": epoca, "perda_treino": perda_treino, "perda_validacao": perda_val})
        if perda_val < melhor_perda - min_delta:
            melhor_perda = perda_val
            melhor_epoca = epoca
            melhor_estado = copy.deepcopy(modelo.state_dict())
            sem_melhora = 0
        else:
            sem_melhora += 1
            if sem_melhora >= paciencia:
                motivo = "paciencia"
                break
    modelo.load_state_dict(melhor_estado)
    return ResultadoTreino(
        historico=pd.DataFrame(linhas),
        melhor_epoca=melhor_epoca,
        epoca_parada=epoca,
        melhor_perda=melhor_perda,
        motivo=motivo,
        melhor_estado=melhor_estado,
    )


def assinatura_sem_teste() -> list[str]:
    """Nomes dos parâmetros de ``treinar_com_parada``; garante que nenhum é um carregador de teste."""
    nomes = list(inspect.signature(treinar_com_parada).parameters)
    assert not [n for n in nomes if "teste" in n.lower() or "test" in n.lower()], "a função não pode receber o teste"
    return nomes


def verificar_restauracao(
    modelo: nn.Module,
    resultado: ResultadoTreino,
    loader_validacao: DataLoader[tuple[Tensor, Tensor]],
    dispositivo: str = "cpu",
) -> dict[str, float | bool]:
    """Prova que o modelo contém o melhor estado guardado.

    Cada tensor do ``state_dict`` precisa ser igual (``torch.equal``) ao
    guardado, e a perda de validação recalculada precisa bater com a melhor
    perda guardada (tolerância 1e-6).
    """
    atual = modelo.state_dict()
    iguais = set(atual) == set(resultado.melhor_estado) and all(
        torch.equal(atual[k], resultado.melhor_estado[k]) for k in atual
    )
    assert iguais, "os pesos do modelo não são o melhor estado guardado"
    recalculada = avaliar_perda(modelo, loader_validacao, nn.BCEWithLogitsLoss(), dispositivo)
    diferenca = abs(recalculada - resultado.melhor_perda)
    assert diferenca < 1e-6, f"perda de validação recalculada difere em {diferenca}"
    return {"tensores_iguais": iguais, "perda_recalculada": recalculada, "diferenca_absoluta": diferenca}


def grafico_perdas(resultado: ResultadoTreino, caminho: Path) -> Figure:
    """Curvas de perda de treino e de validação, com a melhor época e a época de parada marcadas."""
    h = resultado.historico
    fig = Figure(figsize=(8, 4.5), constrained_layout=True)
    eixo = fig.subplots()
    eixo.plot(h["epoca"], h["perda_treino"], label="treino", color="tab:blue")
    eixo.plot(h["epoca"], h["perda_validacao"], label="validação", color="tab:orange")
    eixo.axvline(resultado.melhor_epoca, color="tab:green", linestyle="--", label=f"melhor época ({resultado.melhor_epoca})")
    eixo.axvline(resultado.epoca_parada, color="tab:red", linestyle=":", label=f"época de parada ({resultado.epoca_parada})")
    eixo.set_xlabel("época")
    eixo.set_ylabel("perda BCE média")
    eixo.set_title(f"D7: curvas de perda (parada por {resultado.motivo})")
    eixo.legend()
    fig.savefig(caminho, dpi=130)
    return fig
