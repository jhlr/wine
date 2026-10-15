"""A1/A2: Dataset com normalização aprendida no treino e DataLoaders."""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

PROIBIDAS: frozenset[str] = frozenset({"quality", "alvo", "id_linha", "grupo_perfil"})


def calcular_estatisticas(df: pd.DataFrame, features: Sequence[str]) -> tuple[np.ndarray, np.ndarray]:
    """Média e desvio por coluna, em float64, guardados como float32.

    Convenção: desvio populacional (``ddof=0``). Desvio zero vira 1,0 para não
    dividir por zero (a coluna constante fica só centralizada).
    """
    x = df[list(features)].to_numpy(dtype=np.float64)
    media = x.mean(axis=0)
    desvio = x.std(axis=0, ddof=0)
    desvio = np.where(desvio == 0.0, 1.0, desvio)
    return media.astype(np.float32), desvio.astype(np.float32)


class MeuDataset(Dataset[tuple[Tensor, Tensor]]):
    """Dataset de uma única partição, com z-score por coluna.

    Atributos:
        media, desvio: estatísticas (float32, uma por característica).
        X: tensor float32 ``[N, n_features]`` já normalizado.
        y: tensor float32 ``[N]`` com o alvo 0/1.
        id_linha: ``id_linha`` de cada observação (auditoria; não é entrada).
    """

    media: np.ndarray
    desvio: np.ndarray
    X: Tensor
    y: Tensor
    id_linha: np.ndarray

    def __init__(
        self,
        df: pd.DataFrame,
        features: Sequence[str],
        media: np.ndarray | None = None,
        desvio: np.ndarray | None = None,
    ) -> None:
        """Prepara os tensores de ``df``.

        Se ``media`` e ``desvio`` não forem dados (instância de treino), eles
        são aprendidos só com as linhas de ``df``. Validação e teste recebem os
        valores do treino (``ds_treino.media`` e ``ds_treino.desvio``) e apenas
        os aplicam. Passar só um dos dois é erro. O alvo vem de ``df['alvo']``.
        """
        proibidas = PROIBIDAS.intersection(features)
        if proibidas:
            raise ValueError(f"Colunas que não podem ser entrada do modelo: {sorted(proibidas)}")
        if (media is None) != (desvio is None):
            raise ValueError("Passe media e desvio juntos, ou nenhum dos dois.")
        if media is None or desvio is None:
            media, desvio = calcular_estatisticas(df, features)
        self.media = np.asarray(media, dtype=np.float32)
        self.desvio = np.asarray(desvio, dtype=np.float32)
        bruto = df[list(features)].to_numpy(dtype=np.float32)
        self.X = torch.from_numpy((bruto - self.media) / self.desvio)
        self.y = torch.from_numpy(df["alvo"].to_numpy(dtype=np.float32))
        self.id_linha = df["id_linha"].to_numpy()

    def __len__(self) -> int:
        """Número de observações da partição."""
        return int(self.X.shape[0])

    def __getitem__(self, indice: int) -> tuple[Tensor, Tensor]:
        """Devolve ``(X, y)`` da observação ``indice``: ``X`` float32 ``[11]``, ``y`` escalar float32."""
        return self.X[indice], self.y[indice]


def criar_datasets(
    particoes: dict[str, pd.DataFrame], features: Sequence[str]
) -> tuple[MeuDataset, MeuDataset, MeuDataset]:
    """Cria treino (aprende as estatísticas), validação e teste (reutilizam as do treino)."""
    ds_treino = MeuDataset(particoes["treino"], features)
    ds_validacao = MeuDataset(particoes["validacao"], features, ds_treino.media, ds_treino.desvio)
    ds_teste = MeuDataset(particoes["teste"], features, ds_treino.media, ds_treino.desvio)
    return ds_treino, ds_validacao, ds_teste


def criar_loaders(
    ds_treino: Dataset[tuple[Tensor, Tensor]],
    ds_validacao: Dataset[tuple[Tensor, Tensor]],
    ds_teste: Dataset[tuple[Tensor, Tensor]],
    semente: int = 42,
) -> tuple[DataLoader[tuple[Tensor, Tensor]], DataLoader[tuple[Tensor, Tensor]], DataLoader[tuple[Tensor, Tensor]]]:
    """Cria os três carregadores da A2.

    Treino: lote 64, embaralha com gerador próprio semeado, mantém o último
    lote. Validação e teste: lote 512, sem embaralhar, sem descartar nada.
    Todos com ``num_workers=0``.
    """
    gerador = torch.Generator()
    gerador.manual_seed(semente)
    loader_treino = DataLoader(
        ds_treino, batch_size=64, shuffle=True, drop_last=False, num_workers=0, generator=gerador
    )
    loader_validacao = DataLoader(ds_validacao, batch_size=512, shuffle=False, drop_last=False, num_workers=0)
    loader_teste = DataLoader(ds_teste, batch_size=512, shuffle=False, drop_last=False, num_workers=0)
    return loader_treino, loader_validacao, loader_teste


def tabela_parametros(loaders: dict[str, DataLoader[tuple[Tensor, Tensor]]]) -> pd.DataFrame:
    """Lê os quatro parâmetros diretamente dos carregadores criados (não de uma cópia digitada)."""
    linhas = []
    for nome, loader in loaders.items():
        embaralha = type(loader.sampler).__name__ == "RandomSampler"
        linhas.append(
            {
                "conjunto": nome,
                "batch_size": loader.batch_size,
                "shuffle": embaralha,
                "drop_last": loader.drop_last,
                "num_workers": loader.num_workers,
            }
        )
    return pd.DataFrame(linhas)


def verificar_datasets(
    ds_treino: MeuDataset,
    ds_validacao: MeuDataset,
    ds_teste: MeuDataset,
    loaders: dict[str, DataLoader[tuple[Tensor, Tensor]]],
    features: Sequence[str],
) -> pd.DataFrame:
    """Confere formas, tipos, valores finitos e estatísticas; imprime uma linha OK por checagem.

    Devolve a média e o desvio de cada característica nos três conjuntos (já
    normalizados) para mostrar que validação e teste não têm média zero.
    """
    assert PROIBIDAS.isdisjoint(features), "características incluem alvo ou identificadores"
    print("OK: características não incluem quality, alvo, id_linha nem grupo_perfil.")

    x0, y0 = ds_treino[0]
    assert x0.shape == (len(features),) and x0.dtype == torch.float32
    assert y0.shape == () and y0.dtype == torch.float32
    print(f"OK: exemplo X {tuple(x0.shape)} {x0.dtype}, y {tuple(y0.shape)} {y0.dtype}.")

    esperado = {"treino": len(ds_treino), "validacao": len(ds_validacao), "teste": len(ds_teste)}
    for nome, loader in loaders.items():
        total = 0
        for x, y in loader:
            assert x.ndim == 2 and x.shape[1] == len(features) and y.shape == (x.shape[0],)
            assert x.dtype == torch.float32 and y.dtype == torch.float32
            assert torch.isfinite(x).all() and torch.isfinite(y).all()
            total += int(x.shape[0])
        if loader.drop_last:
            continue
        assert total == esperado[nome], f"{nome}: o carregador perdeu observações"
    print("OK: todos os lotes têm forma [B, 11] / [B], float32, sem NaN ou infinito; nenhum carregador perde linhas.")

    for ds in (ds_validacao, ds_teste):
        assert np.array_equal(ds.media, ds_treino.media) and np.array_equal(ds.desvio, ds_treino.desvio)
    print("OK: validação e teste usam exatamente a média e o desvio do treino (ddof=0).")

    linhas = []
    for nome, ds in (("treino", ds_treino), ("validacao", ds_validacao), ("teste", ds_teste)):
        linhas.append(
            {
                "conjunto": nome,
                "max_abs_media": float(ds.X.mean(dim=0).abs().max()),
                "desvio_medio": float(ds.X.std(dim=0, unbiased=False).mean()),
            }
        )
    return pd.DataFrame(linhas)
