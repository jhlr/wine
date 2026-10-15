"""C: duas formas de vazamento (1 e 4), cada uma com versão incorreta e corrigida.

Tudo roda só sobre treino + validação da B2. O teste principal não entra aqui.
Rede 11→16→1, 100 épocas, lote 64, Adam 1e-3, sem parada antecipada, e as cinco
sementes declaradas de antemão (0 a 4). Nenhuma semente é descartada.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from torch import nn
from torch.utils.data import DataLoader

from apoio import FEATURES, criar_modelo, fixar_semente, obter_escores, treinar_epoca
from lab02lib.data import MeuDataset, calcular_estatisticas

SEMENTES: tuple[int, ...] = (0, 1, 2, 3, 4)
EPOCAS = 100
LOTE = 64
FORMAS: dict[int, str] = {
    1: "1: pré-processamento aprendido com dados reservados",
    4: "4: duplicatas entre aprendizagem e avaliação",
}


@dataclass
class _Avaliacao:
    """Escores de uma execução e as métricas calculadas na avaliação interna."""

    acuracia: float
    auc_roc: float
    acuracia_majoritaria: float
    rotulos: np.ndarray
    escores: np.ndarray


def _treinar_e_avaliar(
    c_treino: pd.DataFrame,
    c_avaliacao: pd.DataFrame,
    semente: int,
    estatisticas: tuple[np.ndarray, np.ndarray] | None = None,
) -> _Avaliacao:
    """Treina 100 épocas e avalia (acurácia em 0,5 e AUC-ROC) no conjunto de avaliação interna.

    ``estatisticas=None`` aprende média e desvio só com ``c_treino`` (caminho
    correto). Passar estatísticas de fora é o que a forma 1 incorreta faz.
    Inicialização e embaralhamento dependem só da ``semente``.
    """
    ds_t = MeuDataset(c_treino, FEATURES, *(estatisticas or (None, None)))
    ds_a = MeuDataset(c_avaliacao, FEATURES, ds_t.media, ds_t.desvio)
    fixar_semente(semente)
    modelo = criar_modelo(11)
    gerador = torch.Generator()
    gerador.manual_seed(semente)
    loader_t = DataLoader(ds_t, batch_size=LOTE, shuffle=True, drop_last=False, num_workers=0, generator=gerador)
    loader_a = DataLoader(ds_a, batch_size=512, shuffle=False, drop_last=False, num_workers=0)
    otimizador = torch.optim.Adam(modelo.parameters(), lr=1e-3)
    criterio = nn.BCEWithLogitsLoss()
    for _ in range(EPOCAS):
        treinar_epoca(modelo, loader_t, otimizador, criterio, "cpu")
    rotulos, escores = obter_escores(modelo, loader_a, "cpu")
    previsto = (escores >= 0.5).astype(int)
    return _Avaliacao(
        acuracia=float((previsto == rotulos).mean()),
        auc_roc=float(roc_auc_score(rotulos, escores)),
        acuracia_majoritaria=float(max(rotulos.mean(), 1.0 - rotulos.mean())),
        rotulos=rotulos,
        escores=escores,
    )


def _registro(
    forma: int, variante: str, semente: int, c_treino: pd.DataFrame, c_avaliacao: pd.DataFrame, ev: _Avaliacao
) -> dict[str, float | int | str]:
    """Linha do resultado de uma execução, com o mecanismo de perfis repetidos entre treino e avaliação."""
    no_treino = c_avaliacao["grupo_perfil"].isin(set(c_treino["grupo_perfil"])).to_numpy()
    acertou = (ev.escores >= 0.5).astype(int) == ev.rotulos
    return {
        "forma": forma,
        "variante": variante,
        "seed": semente,
        "n_treino": len(c_treino),
        "n_avaliacao": len(c_avaliacao),
        "positivos_avaliacao": int(c_avaliacao["alvo"].sum()),
        "acuracia": ev.acuracia,
        "auc_roc": ev.auc_roc,
        "acuracia_majoritaria": ev.acuracia_majoritaria,
        "aval_perfil_no_treino": int(no_treino.sum()),
        "frac_aval_perfil_no_treino": float(no_treino.mean()),
        "acuracia_perfil_no_treino": float(acertou[no_treino].mean()) if no_treino.any() else float("nan"),
        "acuracia_demais": float(acertou[~no_treino].mean()) if (~no_treino).any() else float("nan"),
    }


def c_desenvolvimento_de(particoes: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Treino + validação da B2, e só eles. Garante que nenhuma linha do teste entrou."""
    c_dev = pd.concat([particoes["treino"], particoes["validacao"]], ignore_index=True)
    assert not set(c_dev["id_linha"]) & set(particoes["teste"]["id_linha"]), "linha do teste no desenvolvimento"
    assert c_dev["id_linha"].is_unique
    return c_dev


def executar_forma1(c_dev: pd.DataFrame, semente: int) -> list[dict[str, float | int | str]]:
    """Forma 1. Estatísticas do z-score aprendidas com todas as linhas (incorreto) ou só com o treino (corrigido).

    Mesmas linhas de treino e de avaliação, mesma inicialização e mesmo
    embaralhamento nas duas variantes; só muda de onde vêm média e desvio.
    """
    divisor = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=semente)
    pos_t, pos_a = next(divisor.split(c_dev, groups=c_dev["grupo_perfil"]))
    c_treino, c_avaliacao = c_dev.iloc[pos_t], c_dev.iloc[pos_a]
    assert set(c_treino["grupo_perfil"]).isdisjoint(c_avaliacao["grupo_perfil"])
    assert not set(c_avaliacao["id_linha"]) & set(c_treino["id_linha"])
    estatisticas_todas = calcular_estatisticas(c_dev, FEATURES)  # inclui as linhas de avaliação: vazamento
    errado = _treinar_e_avaliar(c_treino, c_avaliacao, semente, estatisticas_todas)
    certo = _treinar_e_avaliar(c_treino, c_avaliacao, semente)
    return [
        _registro(1, "incorreto", semente, c_treino, c_avaliacao, errado),
        _registro(1, "corrigido", semente, c_treino, c_avaliacao, certo),
    ]


def executar_forma4(c_dev: pd.DataFrame, semente: int) -> list[dict[str, float | int | str]]:
    """Forma 4. Divisão por linhas (perfis atravessam) contra treino sem os perfis da avaliação.

    O conjunto de avaliação ``E`` é o mesmo nas duas variantes. Na corrigida,
    o treino perde toda linha cujo perfil aparece em ``E``; por isso fica menor.
    Média e desvio vêm sempre do treino de cada variante.
    """
    c_treino_errado, c_avaliacao = train_test_split(c_dev, test_size=0.25, random_state=semente)
    perfis_avaliacao = set(c_avaliacao["grupo_perfil"])
    c_treino_certo = c_treino_errado[~c_treino_errado["grupo_perfil"].isin(perfis_avaliacao)]
    assert set(c_treino_certo["grupo_perfil"]).isdisjoint(perfis_avaliacao)
    assert 0 < c_avaliacao["alvo"].sum() < len(c_avaliacao)
    errado = _treinar_e_avaliar(c_treino_errado, c_avaliacao, semente)
    certo = _treinar_e_avaliar(c_treino_certo, c_avaliacao, semente)
    return [
        _registro(4, "incorreto", semente, c_treino_errado, c_avaliacao, errado),
        _registro(4, "corrigido", semente, c_treino_certo, c_avaliacao, certo),
    ]


def executar_c(c_dev: pd.DataFrame, sementes: tuple[int, ...] = SEMENTES) -> pd.DataFrame:
    """Roda formas 1 e 4 para todas as sementes e devolve todas as execuções (nada é descartado)."""
    registros: list[dict[str, float | int | str]] = []
    for semente in sementes:
        registros += executar_forma1(c_dev, semente)
        registros += executar_forma4(c_dev, semente)
    resultado = pd.DataFrame(registros)
    assert len(resultado) == len(sementes) * 2 * 2, "alguma execução foi perdida"
    return resultado


def resumir_c(resultado: pd.DataFrame) -> pd.DataFrame:
    """Média e desvio entre sementes (``ddof=1``) por forma e métrica, e a diferença pareada por semente.

    Diferença = ``100 * (incorreto - corrigido)`` (pontos percentuais; na AUC-ROC,
    pontos de AUC×100), calculada semente a semente e depois resumida.
    O desvio entre sementes não é erro-padrão: os conjuntos internos se sobrepõem.
    """
    linhas = []
    for forma in sorted(resultado["forma"].unique()):
        sub = resultado[resultado["forma"] == forma]
        for coluna, nome in (("acuracia", "acurácia (limiar 0,5)"), ("auc_roc", "AUC-ROC")):
            inc = sub[sub["variante"] == "incorreto"].sort_values("seed")[coluna].to_numpy()
            cor = sub[sub["variante"] == "corrigido"].sort_values("seed")[coluna].to_numpy()
            dif = 100.0 * (inc - cor)
            linhas.append(
                {
                    "forma": forma,
                    "metrica": nome,
                    "incorreto_media": inc.mean(),
                    "incorreto_desvio": inc.std(ddof=1),
                    "corrigido_media": cor.mean(),
                    "corrigido_desvio": cor.std(ddof=1),
                    "diferenca_pp_media": dif.mean(),
                    "diferenca_pp_desvio": dif.std(ddof=1),
                    "diferenca_pp_min": dif.min(),
                    "diferenca_pp_max": dif.max(),
                    "sementes_com_diferenca_positiva": int((dif > 0).sum()),
                    "n_sementes": len(dif),
                }
            )
    return pd.DataFrame(linhas)


def linhas_para_checagem(resumo: pd.DataFrame) -> list[dict[str, float | str]]:
    """Uma linha por forma e métrica (campos sugeridos pelo template) para a célula de conferência."""
    saida = []
    for _, linha in resumo.iterrows():
        saida.append(
            {
                "forma": FORMAS[int(linha["forma"])],
                "fonte": "Red Wine Quality (real), treino + validação da B2",
                "metrica": linha["metrica"],
                "incorreto": float(linha["incorreto_media"]),
                "corrigido": float(linha["corrigido_media"]),
                "diferenca": float(linha["diferenca_pp_media"]),
                "unidade": "pontos percentuais" if "acurácia" in str(linha["metrica"]) else "pontos de AUC×100",
            }
        )
    return saida


INFORMACAO_INDEVIDA: dict[int, str] = {
    1: "média e desvio calculados também com as linhas de avaliação",
    4: "linhas de avaliação com o mesmo perfil de linhas do treino",
}
DADOS_USADOS: dict[int, str] = {
    1: "treino + validação da B2; partição interna por perfil (75/25), semente da execução",
    4: "treino + validação da B2; partição interna por linha (75/25); corrigido remove do treino os perfis da avaliação",
}


def tabela_enunciado(resumo: pd.DataFrame) -> pd.DataFrame:
    """Tabela no formato do enunciado, com ``média ± desvio`` (entre as 5 sementes, ``ddof=1``) em texto."""
    linhas = []
    for _, r in resumo.iterrows():
        forma = int(r["forma"])
        unidade = "p.p." if "acurácia" in str(r["metrica"]) else "pontos de AUC×100"
        linhas.append(
            {
                "forma": FORMAS[forma],
                "dados usados": DADOS_USADOS[forma],
                "métrica": r["metrica"],
                "incorreto (média ± desvio)": f"{r['incorreto_media']:.4f} ± {r['incorreto_desvio']:.4f}",
                "corrigido (média ± desvio)": f"{r['corrigido_media']:.4f} ± {r['corrigido_desvio']:.4f}",
                "diferença (média ± desvio) / unidade": f"{r['diferenca_pp_media']:.3f} ± {r['diferenca_pp_desvio']:.3f} {unidade}",
                "informação indevida": INFORMACAO_INDEVIDA[forma],
            }
        )
    return pd.DataFrame(linhas)


def mecanismo_forma4(resultado: pd.DataFrame) -> pd.DataFrame:
    """Por semente: tamanhos de treino, perfis da avaliação já vistos no treino e acurácia nessas linhas.

    Na variante corrigida a contagem de perfis compartilhados precisa ser zero
    (afirmado aqui, não só mostrado).
    """
    f4 = resultado[resultado["forma"] == 4]
    inc = f4[f4["variante"] == "incorreto"].set_index("seed")
    cor = f4[f4["variante"] == "corrigido"].set_index("seed")
    assert (cor["aval_perfil_no_treino"] == 0).all(), "a variante corrigida ainda compartilha perfis"
    assert (inc["n_avaliacao"] == cor["n_avaliacao"]).all()
    return pd.DataFrame(
        {
            "n_treino_incorreto": inc["n_treino"],
            "n_treino_corrigido": cor["n_treino"],
            "n_avaliacao": inc["n_avaliacao"],
            "aval_com_perfil_no_treino": inc["aval_perfil_no_treino"],
            "fracao_aval_com_perfil_no_treino": inc["frac_aval_perfil_no_treino"],
            "perfis_compartilhados_corrigido": cor["aval_perfil_no_treino"],
            "acuracia_nas_compartilhadas_incorreto": inc["acuracia_perfil_no_treino"],
            "acuracia_nas_demais_incorreto": inc["acuracia_demais"],
        }
    ).reset_index()
