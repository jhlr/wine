"""A4: o erro do BatchNorm com último lote de uma observação e a correção."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset, Subset

from apoio import criar_modelo, fixar_semente, treinar_epoca
from lab02lib.data import MeuDataset

MENSAGEM_ESPERADA = "Expected more than 1 value per channel"


class _ComIdLinha(Dataset[tuple[Tensor, Tensor, int]]):
    """Fina camada que devolve também o ``id_linha``. Só serve para mostrar qual linha foi descartada."""

    def __init__(self, base: Subset[tuple[Tensor, Tensor]], ids: np.ndarray) -> None:
        self.base = base
        self.ids = ids

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, indice: int) -> tuple[Tensor, Tensor, int]:
        x, y = self.base[indice]
        return x, y, int(self.ids[indice])


def demonstrar_batchnorm(
    ds_treino: MeuDataset,
    caminho_log: Path,
    semente: int = 42,
    n_subconjunto: int = 65,
    batch_size: int = 64,
    epocas_corrigidas: int = 3,
) -> list[str]:
    """Reproduz o erro de BatchNorm com 65 linhas do treino e lote 64, e mostra a correção.

    Só a linha sobre ``ValueError`` com a mensagem esperada é capturada; qualquer
    outro erro sobe normalmente. O modelo que falhou fica contaminado (já deu um
    passo), então a versão corrigida usa um modelo novo, com a mesma semente.
    Devolve as linhas do log (também salvas em ``caminho_log``).
    """
    log: list[str] = []

    def dizer(texto: str) -> None:
        print(texto)
        log.append(texto)

    resto = n_subconjunto % batch_size
    assert resto == 1, "a demonstração precisa de resto 1"
    dizer(f"Subconjunto de {n_subconjunto} linhas do treino, lote {batch_size}: resto {n_subconjunto} % {batch_size} = {resto}.")
    dizer("OK: 65 % 64 == 1, o último lote terá uma única observação.")

    gerador_sorteio = np.random.default_rng(semente)
    posicoes = np.sort(gerador_sorteio.choice(len(ds_treino), size=n_subconjunto, replace=False))
    subconjunto = Subset(ds_treino, posicoes.tolist())
    ids = ds_treino.id_linha[posicoes]
    criterio = nn.BCEWithLogitsLoss()

    # 1) Reproduz o erro: BatchNorm em train(), drop_last=False.
    fixar_semente(semente)
    modelo_ruim = criar_modelo(11, batchnorm=True)
    tamanhos_vistos: list[int] = []
    modelo_ruim.register_forward_pre_hook(lambda _modulo, entrada: tamanhos_vistos.append(int(entrada[0].shape[0])))
    otimizador = torch.optim.Adam(modelo_ruim.parameters(), lr=1e-3)
    gerador = torch.Generator()
    gerador.manual_seed(semente)
    loader_ruim = DataLoader(subconjunto, batch_size=batch_size, shuffle=True, drop_last=False, num_workers=0, generator=gerador)
    dizer(f"Sem correção (drop_last=False): {len(loader_ruim)} passos por época.")
    try:
        treinar_epoca(modelo_ruim, loader_ruim, otimizador, criterio, "cpu")
    except ValueError as erro:
        if MENSAGEM_ESPERADA not in str(erro):
            raise
        assert tamanhos_vistos[-1] == 1
        dizer(f"Erro capturado (ValueError): {erro}")
        dizer(f"Tamanho do lote que falhou: {tamanhos_vistos[-1]} (lotes vistos antes dele: {tamanhos_vistos[:-1]}).")
    else:
        raise AssertionError("o erro esperado do BatchNorm não aconteceu")

    # 2) Correção: modelo novo, mesma semente, drop_last=True com shuffle=True.
    fixar_semente(semente)
    modelo_bom = criar_modelo(11, batchnorm=True)
    otimizador_bom = torch.optim.Adam(modelo_bom.parameters(), lr=1e-3)
    entradas_treino: list[Tensor] = []
    modelo_bom.register_forward_pre_hook(lambda _modulo, entrada: entradas_treino.append(entrada[0].detach().clone()))
    gerador_treino = torch.Generator()
    gerador_treino.manual_seed(semente)
    gerador_espelho = torch.Generator()
    gerador_espelho.manual_seed(semente)
    loader_bom = DataLoader(subconjunto, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=0, generator=gerador_treino)
    loader_espelho = DataLoader(
        _ComIdLinha(subconjunto, ids), batch_size=batch_size, shuffle=True, drop_last=True, num_workers=0, generator=gerador_espelho
    )
    dizer(f"Com correção (drop_last=True, shuffle=True): {len(loader_bom)} passo por época, {len(loader_bom) * batch_size} de {n_subconjunto} linhas vistas.")
    descartadas: list[int] = []
    for epoca in range(1, epocas_corrigidas + 1):
        entradas_treino.clear()
        perda = treinar_epoca(modelo_bom, loader_bom, otimizador_bom, criterio, "cpu")
        # O carregador espelho (mesma semente) revela os ids das linhas que o treino realmente viu.
        vistas: list[int] = []
        for posicao, (x_b, _y_b, ids_b) in enumerate(loader_espelho):
            assert torch.equal(entradas_treino[posicao], x_b), "o carregador espelho não reproduz o treino"
            vistas.extend(int(i) for i in ids_b)
        faltou = sorted(set(int(i) for i in ids) - set(vistas))
        assert len(faltou) == 1 and len(vistas) == len(loader_bom) * batch_size
        descartadas.append(faltou[0])
        dizer(f"Época {epoca}: perda de treino {perda:.4f}; linhas vistas: {len(vistas)}; linha descartada: id_linha={faltou[0]}.")
    dizer(f"Linhas descartadas nas {epocas_corrigidas} épocas: {descartadas} (mudam entre épocas: {len(set(descartadas)) > 1}).")
    caminho_log.write_text("\n".join(log) + "\n", encoding="utf-8")
    return log

