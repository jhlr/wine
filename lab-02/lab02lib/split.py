"""B2: partição treino/validação/teste que mantém perfis idênticos juntos."""
from __future__ import annotations

from itertools import combinations

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

CONJUNTOS: tuple[str, str, str] = ("treino", "validacao", "teste")
PROPORCOES_ALVO: dict[str, float] = {"treino": 0.60, "validacao": 0.20, "teste": 0.20}


def particionar_sem_vazar(df: pd.DataFrame, semente: int = 42) -> dict[str, pd.DataFrame]:
    """Divide ``df`` em treino, validação e teste (cerca de 60/20/20) por ``grupo_perfil``.

    Duas chamadas de ``GroupShuffleSplit``: a primeira reserva 20% dos grupos
    para o teste; a segunda separa 25% dos grupos restantes (0,25 de 0,8 = 0,2
    do total) para a validação. Como o sorteio é feito sobre grupos, todas as
    linhas com as mesmas 11 medidas caem no mesmo conjunto. As proporções
    valem para grupos; a quantidade de linhas pode variar um pouco.

    A semente é declarada de antemão e não é escolhida olhando nenhuma métrica.
    """
    grupos = df["grupo_perfil"]
    primeiro = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=semente)
    pos_dev, pos_teste = next(primeiro.split(df, groups=grupos))
    desenvolvimento = df.iloc[pos_dev]
    teste = df.iloc[pos_teste]

    segundo = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=semente)
    pos_treino, pos_val = next(segundo.split(desenvolvimento, groups=desenvolvimento["grupo_perfil"]))
    treino = desenvolvimento.iloc[pos_treino]
    validacao = desenvolvimento.iloc[pos_val]
    return {
        "treino": treino.reset_index(drop=True),
        "validacao": validacao.reset_index(drop=True),
        "teste": teste.reset_index(drop=True),
    }


def auditar_particoes(
    particoes: dict[str, pd.DataFrame], base: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Audita a partição e levanta ``AssertionError`` se alguma regra falhar.

    Verifica: cobertura de todas as linhas, ``id_linha`` sem sobreposição entre
    os três pares, zero perfis compartilhados por par e presença das duas
    classes em cada conjunto. Devolve duas tabelas: uma por conjunto (linhas,
    proporção, desvio em pontos percentuais, positivos, perfis) e outra por par
    de conjuntos (interseção de linhas e de perfis).
    """
    linhas = []
    for nome in CONJUNTOS:
        parte = particoes[nome]
        n_pos = int(parte["alvo"].sum())
        assert n_pos > 0 and n_pos < len(parte), f"{nome}: falta uma das classes"
        proporcao = len(parte) / len(base)
        linhas.append(
            {
                "conjunto": nome,
                "linhas": len(parte),
                "proporcao_linhas": proporcao,
                "desvio_pp_vs_alvo": 100.0 * (proporcao - PROPORCOES_ALVO[nome]),
                "positivos": n_pos,
                "negativos": len(parte) - n_pos,
                "proporcao_positiva": n_pos / len(parte),
                "perfis": int(parte["grupo_perfil"].nunique()),
            }
        )
    tabela = pd.DataFrame(linhas)

    todos_ids = pd.concat([particoes[n]["id_linha"] for n in CONJUNTOS])
    assert todos_ids.is_unique, "id_linha repetido entre ou dentro dos conjuntos"
    assert set(todos_ids) == set(base["id_linha"]) and len(todos_ids) == len(base), "cobertura incompleta"

    pares = []
    for a, b in combinations(CONJUNTOS, 2):
        ids_comuns = set(particoes[a]["id_linha"]) & set(particoes[b]["id_linha"])
        perfis_comuns = set(particoes[a]["grupo_perfil"]) & set(particoes[b]["grupo_perfil"])
        assert not ids_comuns, f"{a} e {b} compartilham id_linha"
        assert not perfis_comuns, f"{a} e {b} compartilham grupo_perfil"
        pares.append(
            {"par": f"{a}-{b}", "ids_compartilhados": len(ids_comuns), "perfis_compartilhados": len(perfis_comuns)}
        )
    return tabela, pd.DataFrame(pares)
