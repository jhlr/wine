"""A3: tempo de uma época completa para vários tamanhos de lote."""
from __future__ import annotations

import copy
import math
import os
import platform
import subprocess
import time
from collections.abc import Sequence

import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from apoio import criar_modelo, fixar_semente, treinar_epoca
from lab02lib.data import MeuDataset

LOTES_PADRAO: tuple[int, ...] = (1, 8, 64, 512)


def descrever_hardware() -> dict[str, str]:
    """Registra o equipamento e as versões usadas na medição."""
    cpu = subprocess.run(
        ["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True, check=False
    ).stdout.strip()
    return {
        "cpu": cpu or platform.processor() or "desconhecido",
        "nucleos_logicos": str(os.cpu_count()),
        "threads_torch": str(torch.get_num_threads()),
        "torch": torch.__version__,
        "python": platform.python_version(),
        "sistema": f"{platform.system()} {platform.release()}",
        "dispositivo": "cpu",
    }


def _uma_epoca(
    estado_inicial: dict[str, torch.Tensor],
    ds_treino: MeuDataset,
    batch_size: int,
    semente: int,
    dispositivo: str,
) -> tuple[float, int]:
    """Reconstrói o modelo do estado inicial, cria otimizador e loader novos e mede uma época."""
    modelo = criar_modelo(11, batchnorm=False).to(dispositivo)
    modelo.load_state_dict(copy.deepcopy(estado_inicial))
    otimizador = torch.optim.Adam(modelo.parameters(), lr=1e-3)
    gerador = torch.Generator()
    gerador.manual_seed(semente)
    loader = DataLoader(
        ds_treino, batch_size=batch_size, shuffle=True, drop_last=False, num_workers=0, generator=gerador
    )
    inicio = time.perf_counter()
    treinar_epoca(modelo, loader, otimizador, nn.BCEWithLogitsLoss(), dispositivo)
    segundos = time.perf_counter() - inicio
    return segundos, len(loader)


def medir_tempos(
    ds_treino: MeuDataset,
    tamanhos: Sequence[int] = LOTES_PADRAO,
    repeticoes: int = 3,
    semente: int = 42,
    dispositivo: str = "cpu",
) -> list[dict[str, float | int]]:
    """Mede uma época completa (previsão, perda, retropropagação e passo) por tamanho de lote.

    Protocolo: o estado inicial é guardado uma vez (semeado). Cada medição
    reconstrói o modelo a partir dele, com otimizador Adam novo e um
    DataLoader novo (gerador próprio; o ``loader_treino`` do restante do
    notebook não é tocado). Há uma época de aquecimento por tamanho, fora da
    medição. As repetições rodam em rodízio entre os tamanhos (1, 8, 64, 512,
    depois de novo), para que oscilações da máquina não se alinhem a um tamanho.
    Nada é descartado.
    """
    fixar_semente(semente)
    estado_inicial = copy.deepcopy(criar_modelo(11, batchnorm=False).state_dict())
    for b in tamanhos:
        _uma_epoca(estado_inicial, ds_treino, b, semente, dispositivo)
    medidas: list[dict[str, float | int]] = []
    n = len(ds_treino)
    for repeticao in range(1, repeticoes + 1):
        for b in tamanhos:
            segundos, passos = _uma_epoca(estado_inicial, ds_treino, b, semente + repeticao, dispositivo)
            assert passos == math.ceil(n / b)
            medidas.append(
                {
                    "batch_size": b,
                    "repeticao": repeticao,
                    "segundos": segundos,
                    "passos": passos,
                    "exemplos": n,
                    "segundos_por_passo": segundos / passos,
                }
            )
    return medidas


def resumir_tempos(medidas: list[dict[str, float | int]]) -> pd.DataFrame:
    """Mediana de segundos por época e por passo, para cada tamanho de lote."""
    df = pd.DataFrame(medidas)
    resumo = (
        df.groupby("batch_size")
        .agg(
            mediana_segundos=("segundos", "median"),
            minimo_segundos=("segundos", "min"),
            maximo_segundos=("segundos", "max"),
            passos=("passos", "first"),
            exemplos=("exemplos", "first"),
            mediana_segundos_por_passo=("segundos_por_passo", "median"),
        )
        .reset_index()
    )
    return resumo
