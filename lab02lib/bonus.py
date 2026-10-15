"""Bônus: oitava forma de vazamento, sobreamostragem dos positivos antes de separar os conjuntos.

Parecida com a forma 4 (cópias dos dois lados), mas a duplicata **não está nos
dados**: é fabricada pela etapa de pré-processamento. Mesma rede, mesmas cinco
sementes e mesmo protocolo da Parte C, só com treino + validação da B2.
Cada linha positiva é copiada **uma vez** (fator declarado antes, sem ajuste).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from lab02lib.leakage import SEMENTES, registro_execucao, treinar_e_avaliar

FORMA_BONUS = 8


def _copiar_positivos(df: pd.DataFrame) -> pd.DataFrame:
    """Acrescenta uma cópia de cada linha positiva (mesmo ``id_linha``, mesmas medidas)."""
    return pd.concat([df, df[df["alvo"] == 1]], ignore_index=True)


def _mecanismo(c_treino: pd.DataFrame, c_avaliacao: pd.DataFrame, escores: np.ndarray) -> dict[str, float | int]:
    """Positivos da avaliação cujo par exato (mesmo ``id_linha``) está no treino, e a revocação neles e nos demais."""
    positivo = (c_avaliacao["alvo"] == 1).to_numpy()
    ids_treino = set(c_treino.loc[c_treino["alvo"] == 1, "id_linha"])
    com_par = c_avaliacao["id_linha"].isin(ids_treino).to_numpy() & positivo
    previsto = escores >= 0.5
    return {
        "positivos_avaliacao_total": int(positivo.sum()),
        "positivos_com_par_no_treino": int(com_par.sum()),
        "revocacao_com_par": float(previsto[com_par].mean()) if com_par.any() else float("nan"),
        "revocacao_sem_par": float(previsto[positivo & ~com_par].mean()) if (positivo & ~com_par).any() else float("nan"),
    }


def executar_bonus(c_dev: pd.DataFrame, semente: int) -> list[dict[str, float | int | str]]:
    """Incorreto: copia os positivos e só depois separa (por linha). Corrigido: separa e copia só no treino."""
    c_sobre = _copiar_positivos(c_dev)
    c_treino_e, c_avaliacao_e = train_test_split(c_sobre, test_size=0.25, random_state=semente)
    ev_errado = treinar_e_avaliar(c_treino_e, c_avaliacao_e, semente)

    c_treino_0, c_avaliacao_0 = train_test_split(c_dev, test_size=0.25, random_state=semente)
    c_treino_c = _copiar_positivos(c_treino_0)
    assert set(c_treino_c["id_linha"]).isdisjoint(c_avaliacao_0["id_linha"])
    ev_certo = treinar_e_avaliar(c_treino_c, c_avaliacao_0, semente)

    saida = []
    for variante, treino, aval, ev in (
        ("incorreto", c_treino_e, c_avaliacao_e, ev_errado),
        ("corrigido", c_treino_c, c_avaliacao_0, ev_certo),
    ):
        registro = registro_execucao(FORMA_BONUS, variante, semente, treino, aval, ev)
        registro["prevalencia_avaliacao"] = float(aval["alvo"].mean())
        registro.update(_mecanismo(treino, aval, ev.escores))
        saida.append(registro)
    return saida


def executar_bonus_completo(c_dev: pd.DataFrame, sementes: tuple[int, ...] = SEMENTES) -> pd.DataFrame:
    """Todas as sementes, nada descartado."""
    registros: list[dict[str, float | int | str]] = []
    for semente in sementes:
        registros += executar_bonus(c_dev, semente)
    resultado = pd.DataFrame(registros)
    assert len(resultado) == 2 * len(sementes)
    return resultado
