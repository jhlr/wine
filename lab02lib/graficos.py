"""Todos os gráficos do laboratório (único dono da parte visual).

Cada ``grafico_*`` monta e devolve uma ``Figure`` (sem pyplot, então nada fica
aberto na memória). ``publicar`` renderiza uma vez, grava o PNG em
``resultados/`` e mostra a mesma imagem no notebook.

Regras aplicadas (apostila de visualização): título = conclusão gerada dos
dados; eixos com unidade; barras a partir de zero; sem eixos superior e direito;
rótulos diretos em vez de legenda; nada é codificado só por cor (sempre há
marcador, traço ou rótulo); cinza para o que não é a mensagem; mesma cor para a
mesma entidade em todas as figuras; rodapé com fonte e n.
"""
from __future__ import annotations

import io
import textwrap
from collections.abc import Sequence
from typing import Literal
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from IPython.display import Image, display
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.ticker import FixedLocator, FuncFormatter, MultipleLocator, NullFormatter
from numpy.typing import NDArray
from scipy.stats import ks_2samp
from sklearn.metrics import precision_recall_curve, roc_curve

from lab02lib.batchnorm import ResultadoBatchNorm
from lab02lib.data import MeuDataset
from lab02lib.decision import BuscaLimiar, C_FN, C_FP, prever
from lab02lib.split import CONJUNTOS

DPI = 130
FONTE = "Fonte: Red Wine Quality (Cortez et al., 2009), UCI, CC BY 4.0"

# ---------------------------------------------------------------- paleta
# Cada cor tem um significado fixo em todas as figuras. Os valores passaram pela
# auditoria de ``auditar_paleta`` (contraste >= 3:1 para gráficos e 4,5:1 para texto).
COR: dict[str, str] = {
    "incorreto": "#D55E00",  # Okabe-Ito vermilion (procedimento com vazamento)
    "corrigido": "#0072B2",  # Okabe-Ito azul
    "positivo": "#8E3B8C",  # classe positiva (nota >= 7)
    "escolhido": "#007A5A",  # regra/limiar escolhido na validação
    "formula": "#8F6E00",  # limiar da fórmula C_FP/(C_FP+C_FN)
    "modelo": "#1A1A1A",  # curvas e medidas da rede
    "regra_05": "#404040",  # rede com limiar 0,5
    "trivial": "#8C8C8C",  # classificador trivial
    "referencia": "#767676",  # linhas de referência, classe negativa, treino
}
COR_TEXTO = "#333333"
COR_TEXTO_SEC = "#595959"
COR_TEXTO_INCORRETO = "#A84300"  # variante do vermelho com 4,5:1 para texto
COR_GRADE = "#E6E6E6"  # decorativa (não carrega informação)

AUDITORIA_CORES: dict[str, tuple[str, str]] = {
    "incorreto": (COR["incorreto"], "gráfico"),
    "corrigido": (COR["corrigido"], "gráfico"),
    "positivo": (COR["positivo"], "gráfico"),
    "escolhido": (COR["escolhido"], "gráfico"),
    "formula": (COR["formula"], "gráfico"),
    "modelo": (COR["modelo"], "gráfico"),
    "regra_05": (COR["regra_05"], "gráfico"),
    "trivial": (COR["trivial"], "gráfico"),
    "referencia": (COR["referencia"], "gráfico"),
    "texto": (COR_TEXTO, "texto"),
    "texto_secundario": (COR_TEXTO_SEC, "texto"),
    "texto_incorreto": (COR_TEXTO_INCORRETO, "texto"),
    "texto_positivo": (COR["positivo"], "texto"),
    "texto_corrigido": (COR["corrigido"], "texto"),
    "texto_escolhido": (COR["escolhido"], "texto"),
    "texto_formula": (COR["formula"], "texto"),
}
# Pares que precisam ser distinguidos na mesma figura (além de marcador/rótulo).
AUDITORIA_PARES: tuple[tuple[str, str, str, str], ...] = (
    ("incorreto", "corrigido", "diferenças pareadas, mecanismo, bônus", "preenchimento cheio × hachurado e rótulo direto"),
    ("positivo", "referencia", "ECDF por classe; taxa de positivos", "traço contínuo × tracejado e rótulo direto"),
    ("trivial", "regra_05", "regras: revocação e custo", "hachura no trivial e rótulo no eixo"),
    ("regra_05", "escolhido", "regras, matrizes, ROC/PR, custo", "quadrado × círculo, traço e rótulo direto"),
    ("trivial", "escolhido", "regras: revocação e custo", "hachura no trivial e rótulo no eixo"),
    ("escolhido", "formula", "curva de custo (D5)", "traço-ponto × tracejado, marcador e rótulo direto"),
    ("regra_05", "formula", "curva de custo (D5)", "pontilhado × tracejado, marcador e rótulo direto"),
    ("modelo", "referencia", "perdas treino/validação, ROC/PR, calibração", "contínuo × tracejado e rótulo direto"),
    ("modelo", "escolhido", "ROC/PR: curva e regra escolhida", "curva × ponto, rótulo direto"),
)
LIMITE_DELTA_E = 20.0  # abaixo disso (deuteranopia) ou ΔL < 10 em cinza, a cor sozinha é fraca
LIMITE_DELTA_L = 10.0

# ---------------------------------------------------------------- formatação pt-BR


def fmt(valor: float, casas: int = 2, sinal: bool = False) -> str:
    """Número com vírgula decimal (``0,25``); ``sinal`` força ``+`` e usa o menos tipográfico."""
    texto = f"{valor:+.{casas}f}" if sinal else f"{valor:.{casas}f}"
    return texto.replace(".", ",").replace("-", "−")


def pct(fracao: float, casas: int = 1) -> str:
    """Fração (0 a 1) como porcentagem pt-BR, ex.: ``14,5%``."""
    return fmt(100.0 * fracao, casas) + "%"


def fmt_e(valor: float, casas: int = 1) -> str:
    """Notação científica com vírgula decimal, ex.: ``8,2e-07``."""
    return f"{valor:.{casas}e}".replace(".", ",")


def _fmt_g(valor: float) -> str:
    return f"{valor:g}".replace(".", ",").replace("-", "−")


def _formatador(casas: int) -> FuncFormatter:
    return FuncFormatter(lambda v, _pos: fmt(v, casas))


def _formatador_g() -> FuncFormatter:
    return FuncFormatter(lambda v, _pos: _fmt_g(v))


# ---------------------------------------------------------------- blocos comuns


def _figura(largura: float, altura: float, colunas: int = 1, linhas: int = 1, sharex: bool = False, sharey: bool = False,
            width_ratios: Sequence[float] | None = None) -> tuple[Figure, list[Axes]]:
    fig = Figure(figsize=(largura, altura), layout="constrained")
    eixos = fig.subplots(linhas, colunas, squeeze=False, sharex=sharex, sharey=sharey, width_ratios=width_ratios)
    return fig, [eixo for eixo in eixos.flat]


def _estilo(eixo: Axes, grade: Literal["x", "y", "both", ""] = "y") -> None:
    """Sem bordas superior e direita, grade só no eixo que se lê."""
    for lado in ("top", "right"):
        eixo.spines[lado].set_visible(False)
    for lado in ("left", "bottom"):
        eixo.spines[lado].set_color(COR["referencia"])
    eixo.tick_params(colors=COR_TEXTO_SEC, labelsize=9)
    eixo.xaxis.label.set_color(COR_TEXTO)
    eixo.yaxis.label.set_color(COR_TEXTO)
    eixo.set_axisbelow(True)
    if grade:
        eixo.grid(axis=grade, color=COR_GRADE, linewidth=0.7)


def _titulo(fig: Figure, texto: str) -> None:
    largura = fig.get_figwidth()
    fig.suptitle(textwrap.fill(texto, int(largura * 9.2)), x=0.005, ha="left", fontsize=12.5, fontweight="bold", color=COR_TEXTO)


def _rodape(fig: Figure, texto: str) -> None:
    largura = fig.get_figwidth()
    fig.supxlabel(textwrap.fill(texto, int(largura * 14.5)), x=0.005, ha="left", fontsize=8, color=COR_TEXTO_SEC)


def _luminancia(hex_cor: str) -> float:
    rgb = np.array([int(hex_cor[i : i + 2], 16) / 255 for i in (1, 3, 5)])
    lin = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
    return float(lin @ np.array([0.2126, 0.7152, 0.0722]))


def _contraste(a: str, b: str) -> float:
    """Razão de contraste WCAG entre duas cores ``#RRGGBB``."""
    la, lb = sorted((_luminancia(a), _luminancia(b)), reverse=True)
    return (la + 0.05) / (lb + 0.05)


def _cor_texto_sobre(rgba: tuple[float, float, float, float]) -> str:
    """Branco ou quase preto, o que tiver mais contraste com o fundo da célula."""
    fundo = "#" + "".join(f"{int(round(c * 255)):02X}" for c in rgba[:3])
    return "#FFFFFF" if _contraste("#FFFFFF", fundo) >= _contraste(COR["modelo"], fundo) else COR["modelo"]


def _num(valor: object) -> float:
    """Converte um escalar de tabela (pandas/numpy) em ``float``."""
    assert isinstance(valor, (int, float, np.number)), f"valor não numérico: {valor!r}"
    return float(valor)


def _rotulo_conjunto(nome: str) -> str:
    return "validação" if nome == "validacao" else nome


def publicar(fig: Figure, caminho: Path) -> None:
    """Renderiza uma única vez, grava o PNG e mostra a mesma imagem no notebook."""
    memoria = io.BytesIO()
    fig.savefig(memoria, format="png", dpi=DPI)
    dados = memoria.getvalue()
    caminho.write_bytes(dados)
    display(Image(data=dados, format="png"))


# ---------------------------------------------------------------- B2


def grafico_b2_taxa_positiva(tabela: pd.DataFrame) -> Figure:
    """Taxa de positivos por partição, com a prevalência geral como referência."""
    t = tabela.set_index("conjunto").loc[list(CONJUNTOS)]
    geral = float(t["positivos"].sum() / t["linhas"].sum())
    fig, (eixo,) = _figura(8.5, 4.6)
    _estilo(eixo)
    nomes = [_rotulo_conjunto(c) for c in t.index]
    taxas = 100.0 * t["proporcao_positiva"].to_numpy()
    eixo.bar(nomes, taxas, color=COR["positivo"], width=0.55)
    for i, (nome, taxa) in enumerate(zip(t.index, taxas)):
        eixo.text(i, taxa + 0.4, f"{fmt(taxa)}%\nn = {int(t.loc[nome, 'linhas'])} ({int(t.loc[nome, 'positivos'])} positivos)",
                  ha="center", va="bottom", fontsize=9, color=COR_TEXTO)
    eixo.axhline(100 * geral, color=COR["referencia"], linestyle="--", linewidth=1.3)
    eixo.text(2.42, 100 * geral, f"base inteira\n{pct(geral)}", ha="left", va="center", fontsize=9, color=COR_TEXTO_SEC)
    eixo.set_xlim(-0.5, 3.05)
    eixo.set_ylim(0, taxas.max() * 1.35)
    eixo.yaxis.set_major_locator(MultipleLocator(5))
    eixo.set_ylabel("positivos (nota ≥ 7), % das linhas")
    eixo.yaxis.set_major_formatter(_formatador(0))
    taxa = {str(k): float(v) for k, v in t["proporcao_positiva"].items()}
    menor, maior = min(taxa, key=lambda k: taxa[k]), max(taxa, key=lambda k: taxa[k])
    _titulo(fig, f"A taxa de positivos vai de {pct(taxa[menor])} ({_rotulo_conjunto(menor)}) "
                 f"a {pct(taxa[maior])} ({_rotulo_conjunto(maior)}); a base inteira tem {pct(geral)}")
    _rodape(fig, f"{FONTE}. Partições por perfil (GroupShuffleSplit, semente 42), n = {int(t['linhas'].sum())} linhas. "
                 "A variação vem do sorteio por grupos; a semente não foi ajustada.")
    return fig


def grafico_b2_particoes(tabela: pd.DataFrame, pares: pd.DataFrame, alvo: dict[str, float]) -> Figure:
    """Linhas por partição contra o alvo 60/20/20 e matriz de perfis compartilhados por par."""
    t = tabela.set_index("conjunto").loc[list(CONJUNTOS)]
    fig, (ax_a, ax_b) = _figura(11, 4.8, colunas=2, width_ratios=[1.35, 1])
    _estilo(ax_a)
    pos = np.arange(len(CONJUNTOS))
    obtido = 100 * t["proporcao_linhas"].to_numpy()
    previsto = np.array([100 * alvo[c] for c in CONJUNTOS])
    ax_a.bar(pos - 0.2, previsto, 0.38, facecolor="white", edgecolor=COR["referencia"], hatch="//", label="alvo")
    ax_a.bar(pos + 0.2, obtido, 0.38, color=COR["regra_05"])
    for i in range(len(pos)):
        ax_a.text(pos[i] - 0.2, previsto[i] + 1, f"alvo\n{fmt(previsto[i], 0)}%", ha="center", fontsize=8.5, color=COR_TEXTO_SEC)
        ax_a.text(pos[i] + 0.2, obtido[i] + 1, f"obtido\n{fmt(obtido[i])}%\n(n = {int(t['linhas'].iloc[i])})", ha="center", fontsize=8.5, color=COR_TEXTO)
    ax_a.set_xticks(pos, [_rotulo_conjunto(c) for c in CONJUNTOS])
    ax_a.set_ylim(0, max(obtido.max(), previsto.max()) * 1.38)
    ax_a.set_ylabel("linhas da partição, % do total")
    ax_a.yaxis.set_major_formatter(_formatador(0))
    # matriz de perfis compartilhados
    n = len(CONJUNTOS)
    matriz = np.full((n, n), np.nan)
    for _, linha in pares.iterrows():
        a, b = str(linha["par"]).split("-")
        i, j = CONJUNTOS.index(a), CONJUNTOS.index(b)
        matriz[i, j] = matriz[j, i] = float(linha["perfis_compartilhados"])
    mapa = matplotlib.colormaps["Greys"].copy()
    mapa.set_bad("#EDEDED")
    topo = max(1.0, float(np.nanmax(matriz)))
    ax_b.imshow(matriz, cmap=mapa, vmin=0, vmax=topo)
    for i in range(n):
        for j in range(n):
            if i == j:
                ax_b.text(j, i, f"{int(t['perfis'].iloc[i])}\nperfis", ha="center", va="center", fontsize=9, color=COR_TEXTO_SEC)
            else:
                ax_b.text(j, i, f"{int(matriz[i, j])}", ha="center", va="center", fontsize=14, fontweight="bold",
                          color=_cor_texto_sobre(mapa(matriz[i, j] / topo)))
    nomes = [_rotulo_conjunto(c) for c in CONJUNTOS]
    ax_b.set_xticks(range(n), nomes)
    ax_b.set_yticks(range(n), nomes)
    ax_b.tick_params(length=0, labelsize=9)
    for lado in ax_b.spines.values():
        lado.set_visible(False)
    ax_b.set_title("perfis compartilhados por par", fontsize=10, color=COR_TEXTO)
    desvio = float(t["desvio_pp_vs_alvo"].abs().max())
    total = int(np.nansum(matriz) / 2)
    frase = "nenhum perfil cruza a fronteira" if total == 0 else f"{total} perfis cruzam a fronteira"
    _titulo(fig, f"As partições ficam a até {fmt(desvio)} p.p. do alvo 60/20/20 e {frase}")
    _rodape(fig, f"{FONTE}. n = {int(t['linhas'].sum())} linhas. Diagonal: perfis distintos da partição (não é contagem de compartilhados). "
                 "Perfil = as 11 medidas idênticas; igualdade de perfil não prova o mesmo vinho.")
    return fig


# ---------------------------------------------------------------- EDA

UNIDADES: dict[str, str] = {
    "fixed acidity": "g/dm³", "volatile acidity": "g/dm³", "citric acid": "g/dm³",
    "residual sugar": "g/dm³", "chlorides": "g/dm³", "free sulfur dioxide": "mg/dm³",
    "total sulfur dioxide": "mg/dm³", "density": "g/cm³", "pH": "escala de pH",
    "sulphates": "g/dm³", "alcohol": "% vol.",
}


def separacao_ks(treino: pd.DataFrame, features: Sequence[str]) -> pd.DataFrame:
    """Estatística de Kolmogorov-Smirnov (positivo × negativo) de cada característica, em ordem decrescente."""
    pos = treino[treino["alvo"] == 1]
    neg = treino[treino["alvo"] == 0]
    linhas = [{"caracteristica": f, "ks": float(ks_2samp(pos[f], neg[f]).statistic)} for f in features]
    return pd.DataFrame(linhas).sort_values("ks", ascending=False, ignore_index=True)


def _ecdf(valores: NDArray[np.float64]) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    x = np.sort(valores)
    return x, np.arange(1, len(x) + 1) / len(x)


def grafico_eda(treino: pd.DataFrame, features: Sequence[str]) -> tuple[Figure, pd.DataFrame]:
    """ECDF por classe de cada característica (só treino), em ordem de separação; devolve também a tabela de KS."""
    ks = separacao_ks(treino, features)
    pos = treino[treino["alvo"] == 1]
    neg = treino[treino["alvo"] == 0]
    fig, eixos = _figura(13, 9.2, colunas=4, linhas=3)
    for k, (nome, valor_ks) in enumerate(zip(ks["caracteristica"], ks["ks"])):
        eixo = eixos[k]
        _estilo(eixo, grade="")
        xp, yp = _ecdf(pos[nome].to_numpy(dtype=float))
        xn, yn = _ecdf(neg[nome].to_numpy(dtype=float))
        eixo.step(xn, yn, where="post", color=COR["referencia"], linestyle="--", linewidth=1.6)
        eixo.step(xp, yp, where="post", color=COR["positivo"], linestyle="-", linewidth=1.8)
        eixo.set_title(nome, fontsize=10, fontweight="bold" if k == 0 else "normal", color=COR_TEXTO, loc="left")
        eixo.text(0.97, 0.05, f"KS = {fmt(float(valor_ks))}", transform=eixo.transAxes, ha="right", fontsize=9, color=COR_TEXTO)
        eixo.set_xlabel(UNIDADES[nome], fontsize=8.5)
        eixo.xaxis.set_major_formatter(_formatador_g())
        eixo.set_yticks([0, 0.5, 1])
        eixo.yaxis.set_major_formatter(_formatador(1))
        if k % 4 == 0:
            eixo.set_ylabel("fração acumulada", fontsize=8.5)
        if k == 0:
            mp, mn = float(np.median(pos[nome])), float(np.median(neg[nome]))
            direita = mp > mn
            eixo.text(mp, 0.5, " positivo " if direita else " positivo ", ha="left" if direita else "right",
                      va="top", fontsize=9, color=COR["positivo"], fontweight="bold")
            eixo.text(mn, 0.5, " negativo ", ha="right" if direita else "left", va="bottom", fontsize=9, color=COR_TEXTO_SEC, fontweight="bold")
    chave = eixos[len(ks)]
    chave.axis("off")
    chave.plot([0.05, 0.3], [0.7, 0.7], color=COR["positivo"], linewidth=1.8, transform=chave.transAxes)
    chave.text(0.35, 0.7, f"positivo (nota ≥ 7), n = {len(pos)}", va="center", fontsize=9, color=COR["positivo"], transform=chave.transAxes)
    chave.plot([0.05, 0.3], [0.45, 0.45], color=COR["referencia"], linestyle="--", linewidth=1.6, transform=chave.transAxes)
    chave.text(0.35, 0.45, f"negativo, n = {len(neg)}", va="center", fontsize=9, color=COR_TEXTO_SEC, transform=chave.transAxes)
    melhor, pior = ks.iloc[0], ks.iloc[-1]
    _titulo(fig, f"{melhor['caracteristica']} é a característica que mais separa as classes no treino (KS = {fmt(float(melhor['ks']))}); "
                 f"{pior['caracteristica']}, a que menos ({fmt(float(pior['ks']))})")
    _rodape(fig, f"{FONTE}. Conjunto de treino, n = {len(treino)} ({len(pos)} positivos, {len(neg)} negativos). "
                 "Escala do eixo x livre em cada painel porque as unidades diferem (g/dm³, mg/dm³, g/cm³, % vol., pH; unidades segundo Cortez et al., 2009). "
                 "KS = maior distância vertical entre as duas curvas (0 = iguais, 1 = separadas).")
    return fig, ks


# ---------------------------------------------------------------- A1


def estatisticas_por_conjunto(
    datasets: dict[str, MeuDataset],
) -> dict[str, tuple[NDArray[np.float64], NDArray[np.float64], int]]:
    """Média e desvio (``ddof=0``) de cada característica já normalizada, por conjunto, e o n do conjunto."""
    return {
        nome: (ds.X.double().mean(dim=0).numpy(), ds.X.double().std(dim=0, unbiased=False).numpy(), len(ds))
        for nome, ds in datasets.items()
    }


def grafico_normalizacao(estatisticas: dict[str, tuple[NDArray[np.float64], NDArray[np.float64], int]], features: Sequence[str]) -> Figure:
    """Média e desvio de cada característica nos três conjuntos, depois do z-score com as estatísticas do treino."""
    marcador = {"treino": "o", "validacao": "s", "teste": "^"}
    tom = {"treino": COR["modelo"], "validacao": COR["regra_05"], "teste": COR["trivial"]}
    fig, (ax_m, ax_d) = _figura(11.5, 5.8, colunas=2, sharey=True)
    y = np.arange(len(features))[::-1]
    for eixo, indice, rotulo, ref in ((ax_m, 0, "média (em desvios do treino)", 0.0), (ax_d, 1, "desvio padrão (ddof = 0)", 1.0)):
        _estilo(eixo, grade="y")
        eixo.axvline(ref, color=COR["referencia"], linewidth=1.0, linestyle=":")
        for nome in CONJUNTOS:
            eixo.scatter(estatisticas[nome][indice], y, marker=marcador[nome], s=44, color=tom[nome],
                         edgecolors="white", linewidths=0.6, zorder=3)
        eixo.set_xlabel(rotulo)
        eixo.xaxis.set_major_formatter(_formatador(2))
    ax_m.set_yticks(y, list(features))
    # rótulos diretos na linha com maior espalhamento da média
    espalhamento = np.ptp(np.vstack([estatisticas[c][0] for c in CONJUNTOS]), axis=0)
    linha = int(np.argmax(espalhamento))
    deslocamentos = {"treino": (-34, 14), "validacao": (6, 16), "teste": (30, -18)}
    for nome in CONJUNTOS:
        ax_m.annotate(_rotulo_conjunto(nome), (estatisticas[nome][0][linha], y[linha]), xytext=deslocamentos[nome],
                      textcoords="offset points", fontsize=9, color=COR_TEXTO, ha="center",
                      arrowprops={"arrowstyle": "-", "color": COR["referencia"], "linewidth": 0.8})
    maior_media = max(float(np.abs(estatisticas[c][0]).max()) for c in ("validacao", "teste"))
    desvios = np.concatenate([estatisticas[c][1] for c in ("validacao", "teste")])
    dev_treino = float(np.abs(estatisticas["treino"][1] - 1).max())
    _titulo(fig, f"Com a escala do treino, o treino fica em média 0 e desvio 1; validação e teste chegam a {fmt(maior_media)} desvio de média "
                 f"e têm desvios entre {fmt(float(desvios.min()))} e {fmt(float(desvios.max()))}")
    _rodape(fig, f"{FONTE}. n = treino/validação/teste = " + "/".join(str(estatisticas[c][2]) for c in CONJUNTOS) + ". "
                 f"Média e desvio do treino aplicados às três partições (desvio do treino difere de 1 em até {fmt_e(dev_treino)}). "
                 "Marcadores: círculo = treino, quadrado = validação, triângulo = teste.")
    return fig


# ---------------------------------------------------------------- A3


def grafico_tempos(medidas: list[dict[str, float | int]], hardware: dict[str, str]) -> Figure:
    """Tempo por época e custo por passo contra o tamanho do lote (log-log), com rótulos diretos."""
    df = pd.DataFrame(medidas)
    resumo = df.groupby("batch_size").agg(mediana=("segundos", "median"), passos=("passos", "first"),
                                          por_passo=("segundos_por_passo", "median"), exemplos=("exemplos", "first")).reset_index()
    lotes = resumo["batch_size"].to_numpy()
    fig, (ax_a, ax_b) = _figura(11.5, 5.0, colunas=2)
    rotulos_x = [f"{int(b)}\n({int(p)} passos)" for b, p in zip(lotes, resumo["passos"])]
    for eixo in (ax_a, ax_b):
        _estilo(eixo, grade="y")
        eixo.set_xscale("log", base=2)
        eixo.set_yscale("log")
        eixo.xaxis.set_major_locator(FixedLocator([float(b) for b in lotes]))
        eixo.set_xticklabels(rotulos_x)
        eixo.xaxis.set_minor_formatter(NullFormatter())
        eixo.yaxis.set_minor_formatter(NullFormatter())
        eixo.set_xlabel("tamanho do lote (observações por passo)")
        eixo.margins(x=0.12)
    ax_a.scatter(df["batch_size"], df["segundos"], s=22, color=COR["referencia"], alpha=0.7, zorder=2)
    ax_a.plot(lotes, resumo["mediana"], marker="o", color=COR["modelo"], linewidth=1.8, zorder=3)
    medianas = resumo["mediana"].to_numpy()
    for b, m in zip(lotes, medianas):
        ax_a.annotate(f"{fmt(float(m), 4)} s", (b, m), xytext=(8, 9), textcoords="offset points", fontsize=9, color=COR_TEXTO)
    razoes = medianas[:-1] / medianas[1:]
    for i, r in enumerate(razoes):
        ax_a.text(np.sqrt(lotes[i] * lotes[i + 1]) * 1.1, np.sqrt(medianas[i] * medianas[i + 1]) * 1.1, f"÷ {fmt(float(r), 1)}",
                  fontsize=9, color=COR_TEXTO_SEC, ha="left")
    ax_a.set_ylabel("tempo de uma época (s, escala log)")
    ax_a.text(0.97, 0.95, "pontos cinza: 3 repetições\nlinha: mediana", transform=ax_a.transAxes, ha="right", va="top", fontsize=8.5, color=COR_TEXTO_SEC)
    por_passo = 1000 * resumo["por_passo"].to_numpy()
    ax_b.plot(lotes, por_passo, marker="s", color=COR["modelo"], linewidth=1.8, linestyle="--")
    for b, v in zip(lotes, por_passo):
        ax_b.annotate(f"{fmt(float(v), 3)} ms", (b, v), xytext=(8, -14), textcoords="offset points", fontsize=9, color=COR_TEXTO)
    ax_b.set_ylabel("custo de um passo (ms, mediana, escala log)")
    fator_linhas = lotes[-1] / lotes[0]
    fator_passo = por_passo[-1] / por_passo[0]
    ax_b.text(0.03, 0.95, f"{int(fator_linhas)}× mais linhas por passo,\n{fmt(float(fator_passo), 1)}× mais tempo por passo",
              transform=ax_b.transAxes, ha="left", va="top", fontsize=9, color=COR_TEXTO)
    for eixo, valores in ((ax_a, df["segundos"].to_numpy()), (ax_b, por_passo)):
        baixo, alto = float(valores.min()), float(valores.max())
        eixo.set_ylim(baixo / 1.8, alto * 2.2)
        marcas = [c * 10.0**k for k in range(-5, 3) for c in (1, 3) if baixo / 1.8 <= c * 10.0**k <= alto * 2.2]
        eixo.yaxis.set_major_locator(FixedLocator(marcas))
        eixo.yaxis.set_major_formatter(_formatador_g())
    ganho = medianas[0] / medianas[-1]
    _titulo(fig, f"Com o lote {int(lotes[-1])}, uma época leva {fmt(float(medianas[-1]), 4)} s contra {fmt(float(medianas[0]), 3)} s com o lote {int(lotes[0])} "
                 f"({fmt(float(ganho), 0)}× mais rápido), mas cada passo custa {fmt(float(fator_passo), 1)}× mais")
    _rodape(fig, f"Treino, n = {int(resumo['exemplos'].iloc[0])} linhas; época completa (previsão, perda, retropropagação, Adam). Equipamento: {hardware['cpu']}, "
                 f"CPU, {hardware['threads_torch']} threads, torch {hardware['torch']}, Python {hardware['python']}. "
                 "Menos passos por época significa menos atualizações de pesos: tempo menor não é o mesmo aprendizado.")
    return fig


# ---------------------------------------------------------------- A4


def grafico_batchnorm(r: ResultadoBatchNorm) -> Figure:
    """Lotes de uma época sem e com ``drop_last`` e a linha descartada a cada época."""
    fig, (ax_a, ax_b) = _figura(11.5, 4.8, colunas=2, width_ratios=[1, 1.15])
    _estilo(ax_a, grade="x")
    cumulativo = 0
    for k, tamanho in enumerate(r.tamanhos_sem_correcao):
        ax_a.barh(1, tamanho, left=cumulativo, color=COR["regra_05"] if tamanho > 1 else COR["incorreto"], edgecolor="white", height=0.5)
        cumulativo += tamanho
    ax_a.text(r.batch_size / 2, 1, f"lote 1: {r.batch_size} linhas", ha="center", va="center", color="white", fontsize=9)
    ax_a.annotate(f"lote 2: {r.tamanhos_sem_correcao[-1]} linha\n→ ValueError", (r.n_linhas - 0.5, 1.25), xytext=(-5, 28),
                  textcoords="offset points", ha="right", fontsize=9, color=COR_TEXTO_INCORRETO,
                  arrowprops={"arrowstyle": "-", "color": COR["incorreto"]})
    ax_a.text(0, 1.42, "sem correção (drop_last=False)", fontsize=9, color=COR_TEXTO, fontweight="bold")
    ax_a.barh(0, r.batch_size, color=COR["corrigido"], height=0.5)
    ax_a.barh(0, r.n_linhas - r.batch_size, left=r.batch_size, facecolor="white", edgecolor=COR["referencia"], hatch="//", height=0.5)
    ax_a.text(r.batch_size / 2, 0, f"lote 1: {r.batch_size} linhas", ha="center", va="center", color="white", fontsize=9)
    ax_a.annotate("1 linha descartada", (r.n_linhas - 0.5, 0.25), xytext=(-5, 22), textcoords="offset points", ha="right",
                  fontsize=9, color=COR["referencia"], arrowprops={"arrowstyle": "-", "color": COR["referencia"]})
    ax_a.text(0, -0.58, "com correção (drop_last=True, shuffle=True)", fontsize=9, color=COR_TEXTO, fontweight="bold")
    ax_a.set_yticks([])
    ax_a.set_ylim(-0.75, 1.9)
    ax_a.set_xlim(0, r.n_linhas)
    ax_a.set_xlabel(f"linhas do subconjunto em uma época (n = {r.n_linhas})")
    ax_a.spines["left"].set_visible(False)
    # linha descartada por época
    _estilo(ax_b, grade="")
    ordem = {i: p for p, i in enumerate(sorted(r.ids))}
    epocas = np.arange(1, len(r.descartadas) + 1)
    for e in epocas:
        ax_b.scatter(np.full(len(r.ids), e), [ordem[i] for i in r.ids], s=14, color=COR["referencia"], alpha=0.55, zorder=2)
    for e, i in zip(epocas, r.descartadas):
        ax_b.scatter([e], [ordem[i]], s=90, marker="X", color=COR["incorreto"], edgecolors="white", zorder=3)
        ax_b.annotate(f"id_linha {i}", (e, ordem[i]), xytext=(9, 0), textcoords="offset points", va="center", fontsize=9, color=COR_TEXTO_INCORRETO)
    ax_b.set_xticks(epocas)
    ax_b.set_xlim(0.5, len(epocas) + 0.9)
    ax_b.set_xlabel("época")
    ax_b.set_ylabel("posição da linha (ordenada por id_linha)")
    iguais = len(set(r.descartadas)) == len(r.descartadas)
    complemento = "e a linha descartada muda a cada época" if iguais else "e a mesma linha foi descartada em mais de uma época"
    _titulo(fig, f"Com {r.n_linhas} linhas e lote {r.batch_size}, o lote final tem {r.tamanhos_sem_correcao[-1]} linha e o BatchNorm falha; "
                 f"com drop_last=True o treino vê {r.n_linhas - 1} de {r.n_linhas} linhas {complemento}")
    _rodape(fig, f"Subconjunto de {r.n_linhas} linhas do treino (semente 42), rede com BatchNorm1d em train(). "
                 "X = linha descartada; pontos cinza = linhas vistas. Só vale para o treino: validação e teste usam drop_last=False.")
    return fig


# ---------------------------------------------------------------- C e bônus


def _barras_pareadas(eixo: Axes, sementes: NDArray[np.int_], inc: NDArray[np.float64], cor: NDArray[np.float64], formato: str,
                     rotulo_y: str, teto: float | None = None) -> None:
    """Barras agrupadas por semente (incorreto × corrigido), a partir de zero, com o valor sobre cada barra."""
    pos = np.arange(len(sementes))
    _estilo(eixo)
    eixo.bar(pos - 0.2, inc, 0.38, color=COR["incorreto"])
    eixo.bar(pos + 0.2, cor, 0.38, facecolor="white", edgecolor=COR["corrigido"], hatch="//", linewidth=1.4)
    topo = max(float(inc.max()), float(cor.max()), 1e-9)
    for p, a, b in zip(pos, inc, cor):
        eixo.text(p - 0.2, a + topo * 0.02, formato.format(a).replace(".", ","), ha="center", va="bottom", fontsize=8, color=COR_TEXTO)
        eixo.text(p + 0.2, b + topo * 0.02, formato.format(b).replace(".", ","), ha="center", va="bottom", fontsize=8, color=COR_TEXTO)
    eixo.set_xticks(pos, [str(int(s)) for s in sementes])
    eixo.set_xlabel("semente")
    eixo.set_ylabel(rotulo_y)
    eixo.set_ylim(0, topo * 1.3 if teto is None else min(topo * 1.3, teto))
    eixo.text(0.02, 0.97, "■ incorreto", transform=eixo.transAxes, ha="left", va="top", fontsize=9.5, fontweight="bold", color=COR_TEXTO_INCORRETO)
    eixo.text(0.40, 0.97, "▨ corrigido", transform=eixo.transAxes, ha="left", va="top", fontsize=9.5, fontweight="bold", color=COR["corrigido"])


def _pontos_pareados(eixo: Axes, sementes: NDArray[np.int_], a: NDArray[np.float64], b: NDArray[np.float64], nome_a: str, nome_b: str,
                     rotulo_y: str) -> None:
    """Dois pontos por semente (círculo e losango) ligados por um traço; eixo não parte de zero."""
    pos = np.arange(len(sementes))
    _estilo(eixo)
    for p, va, vb in zip(pos, a, b):
        eixo.plot([p, p], [va, vb], color=COR["referencia"], linewidth=1.0, zorder=1)
    eixo.scatter(pos, a, marker="o", s=55, color=COR["incorreto"], edgecolors="white", zorder=3)
    eixo.scatter(pos, b, marker="D", s=45, color=COR["referencia"], edgecolors="white", zorder=3)
    eixo.set_xticks(pos, [str(int(s)) for s in sementes])
    eixo.set_xlabel("semente")
    eixo.set_ylabel(rotulo_y)
    eixo.yaxis.set_major_formatter(_formatador(2))
    baixo, alto = float(min(a.min(), b.min())), float(max(a.max(), b.max()))
    folga = (alto - baixo) * 0.12 + 0.005
    eixo.set_ylim(baixo - folga, alto + 3 * folga)
    eixo.text(0.02, 0.97, f"● {nome_a}", transform=eixo.transAxes, ha="left", va="top", fontsize=9.5, fontweight="bold", color=COR_TEXTO_INCORRETO)
    eixo.text(0.02, 0.90, f"◆ {nome_b}", transform=eixo.transAxes, ha="left", va="top", fontsize=9.5, fontweight="bold", color=COR_TEXTO_SEC)


def grafico_mecanismo_forma4(resultado: pd.DataFrame) -> Figure:
    """Forma 4: perfis da avaliação já vistos no treino, tamanho do treino e acurácia nessas linhas."""
    f4 = resultado[resultado["forma"] == 4]
    inc = f4[f4["variante"] == "incorreto"].sort_values("seed")
    cor = f4[f4["variante"] == "corrigido"].sort_values("seed")
    sementes = inc["seed"].to_numpy()
    fig, (ax_a, ax_b, ax_c) = _figura(13.5, 5.0, colunas=3)
    _barras_pareadas(ax_a, sementes, 100 * inc["frac_aval_perfil_no_treino"].to_numpy(), 100 * cor["frac_aval_perfil_no_treino"].to_numpy(),
                     "{:.1f}%", "linhas de avaliação com perfil no treino (%)", teto=100)
    _barras_pareadas(ax_b, sementes, inc["n_treino"].to_numpy(dtype=float), cor["n_treino"].to_numpy(dtype=float), "{:.0f}",
                     "linhas de treino")
    comp, resto = inc["acuracia_perfil_no_treino"].to_numpy(), inc["acuracia_demais"].to_numpy()
    _pontos_pareados(ax_c, sementes, comp, resto, "perfil também no treino", "demais linhas", "acurácia do incorreto (limiar 0,5; eixo não parte de zero)")
    media_frac = 100 * float(inc["frac_aval_perfil_no_treino"].mean())
    zero_corrigido = float(cor["frac_aval_perfil_no_treino"].max()) == 0.0
    encolhe = 100 * (1 - float(cor["n_treino"].mean() / inc["n_treino"].mean()))
    f1 = resultado[resultado["forma"] == 1]["frac_aval_perfil_no_treino"].max()
    maior = int((comp > resto).sum())
    _titulo(fig, f"Na forma 4, em média {fmt(media_frac)}% das linhas de avaliação têm perfil no treino do incorreto "
                 f"({'0%' if zero_corrigido else 'não zero'} no corrigido), que também tem {fmt(encolhe)}% menos linhas de treino")
    _rodape(fig, f"{FONTE}. Treino + validação da B2, partição interna 75/25 por linha, n avaliação = {int(inc['n_avaliacao'].iloc[0])}, 5 sementes. "
                 f"Forma 1 (divisão por perfil): no máximo {pct(float(f1))} das linhas com perfil no treino nas duas variantes. "
                 f"Acurácia maior nas linhas com perfil repetido em {maior} de {len(sementes)} sementes: contraste ruidoso, não é prova de memorização.")
    return fig


def grafico_mecanismo_bonus(bonus: pd.DataFrame) -> Figure:
    """Bônus: positivos da avaliação com par exato no treino, revocação nesses e prevalência da avaliação."""
    inc = bonus[bonus["variante"] == "incorreto"].sort_values("seed")
    cor = bonus[bonus["variante"] == "corrigido"].sort_values("seed")
    sementes = inc["seed"].to_numpy()
    fig, (ax_a, ax_b, ax_c) = _figura(13.5, 5.0, colunas=3)
    taxa = lambda d: 100 * d["positivos_com_par_no_treino"].to_numpy() / d["positivos_avaliacao_total"].to_numpy()  # noqa: E731
    _barras_pareadas(ax_a, sementes, taxa(inc), taxa(cor), "{:.1f}%", "positivos da avaliação com par exato no treino (%)", teto=100)
    _barras_pareadas(ax_b, sementes, 100 * inc["prevalencia_avaliacao"].to_numpy(), 100 * cor["prevalencia_avaliacao"].to_numpy(), "{:.1f}%",
                     "positivos na avaliação (%)", teto=100)
    com, sem = inc["revocacao_com_par"].to_numpy(), inc["revocacao_sem_par"].to_numpy()
    _pontos_pareados(ax_c, sementes, com, sem, "positivos com par", "positivos sem par", "revocação do incorreto (limiar 0,5; eixo não parte de zero)")
    media = float(taxa(inc).mean())
    maior = int((com > sem).sum())
    leitura = "maior" if com.mean() > sem.mean() + 0.05 else "parecida ou menor"
    _titulo(fig, f"No bônus, em média {fmt(media)}% dos positivos da avaliação do incorreto têm o par exato no treino (corrigido: {fmt(float(taxa(cor).mean()), 0)}%); "
                 f"a revocação nesses é {leitura} que nos demais")
    _rodape(fig, f"{FONTE}. Treino + validação da B2, 5 sementes, divisão por linha 75/25. Revocação maior nos com par em {maior} de {len(sementes)} sementes. "
                 "As prevalências diferem entre as variantes, então a acurácia não é comparável; a AUC-ROC é a métrica de leitura.")
    return fig


# coluna de ``resultado`` -> (nome da métrica, rótulo do eixo com a unidade)
METRICAS_C: dict[str, tuple[str, str]] = {
    "acuracia": ("acurácia (limiar 0,5)", "incorreto − corrigido, em p.p. de acurácia"),
    "auc_roc": ("AUC-ROC", "incorreto − corrigido, em pontos de AUC-ROC × 100"),
}
# no bônus as prevalências diferem entre as variantes: só a AUC-ROC é comparável
METRICAS_BONUS: dict[str, tuple[str, str]] = {"auc_roc": METRICAS_C["auc_roc"]}


def grafico_diferencas(resultado: pd.DataFrame, formas: dict[int, str], metricas: dict[str, tuple[str, str]]) -> Figure:
    """Diferença pareada por semente (``100 × (incorreto − corrigido)``), uma linha por forma × métrica.

    ``metricas`` mapeia coluna → (nome, rótulo do eixo com a unidade). A última métrica entra no título.
    """
    fig, eixos = _figura(max(6.2 * len(metricas) + 1.0, 9.5), 1.55 * len(formas) + 3.0, colunas=len(metricas), sharex=True, sharey=True)
    ordem_formas = list(formas)
    resumo: dict[str, dict[int, tuple[float, int, int]]] = {}
    for eixo, (coluna, (nome, rotulo)) in zip(eixos, metricas.items()):
        _estilo(eixo, grade="x")
        eixo.axvline(0, color=COR_TEXTO, linewidth=1.2)
        resumo[coluna] = {}
        for linha, forma in enumerate(ordem_formas):
            sub = resultado[resultado["forma"] == forma]
            inc = sub[sub["variante"] == "incorreto"].sort_values("seed")
            cor = sub[sub["variante"] == "corrigido"].sort_values("seed")
            dif = 100.0 * (inc[coluna].to_numpy() - cor[coluna].to_numpy())
            y = len(ordem_formas) - 1 - linha
            deslocamento = (np.arange(len(dif)) - (len(dif) - 1) / 2) * 0.11
            eixo.scatter(dif, y + deslocamento, s=34, color=COR["modelo"], alpha=0.85, edgecolors="white", linewidths=0.5, zorder=3)
            media = float(dif.mean())
            eixo.scatter([media], [y], marker="D", s=120, color="white", edgecolors=COR["modelo"], linewidths=2.0, zorder=4)
            eixo.annotate(f"média {fmt(media, 2, sinal=True)}", (media, y), xytext=(0, 20), textcoords="offset points", ha="center",
                          fontsize=9, color=COR_TEXTO, fontweight="bold")
            resumo[coluna][forma] = (media, int((dif > 0).sum()), len(dif))
        eixo.set_xlabel(rotulo)
        eixo.set_title(nome, fontsize=10, loc="left", color=COR_TEXTO)
        eixo.xaxis.set_major_formatter(_formatador(1))
        eixo.set_ylim(-0.6, len(ordem_formas) - 0.4)
        eixo.text(-0.04, 0.02, "← corrigido melhor", transform=eixo.get_xaxis_transform(), ha="right", fontsize=8.5, color=COR["corrigido"], fontweight="bold")
        eixo.text(0.04, 0.02, "incorreto melhor →", transform=eixo.get_xaxis_transform(), ha="left", fontsize=8.5, color=COR_TEXTO_INCORRETO, fontweight="bold")
    eixos[0].set_yticks(range(len(ordem_formas)), [textwrap.fill(formas[f], 24) for f in reversed(ordem_formas)])
    ultima = list(metricas)[-1]
    nome_ultima = metricas[ultima][0]
    partes = [f"forma {f}: {fmt(resumo[ultima][f][0], 2, sinal=True)} (positiva em {resumo[ultima][f][1]} de {resumo[ultima][f][2]} sementes)" for f in ordem_formas]
    unidade = ", em pontos de AUC×100" if "AUC" in nome_ultima else ", em pontos percentuais"
    _titulo(fig, f"{nome_ultima}, incorreto − corrigido{unidade}: " + "; ".join(partes))
    n_sem = len(resultado["seed"].unique())
    _rodape(fig, f"{FONTE}. Treino + validação da B2; {n_sem} sementes (0 a {n_sem - 1}); na mesma semente os dois procedimentos compartilham inicialização e embaralhamento. "
                 "Cada círculo é uma semente (deslocamento vertical = ordem da semente, para todas ficarem visíveis); losango = média. "
                 "Diferença = 100 × (incorreto − corrigido). O espalhamento entre sementes não é erro-padrão.")
    return fig


# ---------------------------------------------------------------- D7


def grafico_perdas(historico: pd.DataFrame, melhor_epoca: int, epoca_parada: int, motivo: str, melhor_perda: float,
                   prova: dict[str, float | bool], n_treino: int, n_validacao: int) -> Figure:
    """Curvas de perda com a melhor época, a época de parada e a janela de paciência."""
    fig, (eixo,) = _figura(9.5, 5.2)
    _estilo(eixo)
    ep = historico["epoca"].to_numpy()
    treino = historico["perda_treino"].to_numpy()
    val = historico["perda_validacao"].to_numpy()
    eixo.axvspan(melhor_epoca, epoca_parada, color="#F0F0F0", zorder=0)
    eixo.plot(ep, treino, color=COR["referencia"], linewidth=1.8, linestyle="--")
    eixo.plot(ep, val, color=COR["modelo"], linewidth=2.0)
    eixo.axvline(melhor_epoca, color=COR["escolhido"], linewidth=1.4, linestyle="-.")
    eixo.axvline(epoca_parada, color=COR["incorreto"], linewidth=1.4, linestyle=":")
    eixo.scatter([melhor_epoca], [melhor_perda], s=90, facecolor="white", edgecolor=COR["escolhido"], linewidths=2.2, zorder=4)
    eixo.text(ep[-1] + 0.8, treino[-1] - 0.011, "treino", color=COR_TEXTO_SEC, fontsize=10, fontweight="bold", va="center")
    eixo.text(ep[-1] + 0.8, val[-1] + 0.009, "validação", color=COR["modelo"], fontsize=10, fontweight="bold", va="center")
    topo = float(max(treino.max(), val.max()))
    eixo.text(melhor_epoca - 1, topo * 0.78, f"melhor época: {melhor_epoca}\nperda de validação {fmt(melhor_perda, 4)}", ha="right", fontsize=9.5,
              color=COR["escolhido"], fontweight="bold")
    razao = "paciência" if motivo == "paciencia" else "limite de épocas"
    eixo.text(epoca_parada + 1, topo * 0.78, f"parada: época {epoca_parada}\n({razao})", ha="left", fontsize=9.5, color=COR_TEXTO_INCORRETO, fontweight="bold")
    eixo.set_xlim(0, epoca_parada + 11)
    eixo.set_xlabel("época")
    eixo.set_ylabel("perda BCE média")
    eixo.yaxis.set_major_formatter(_formatador(2))
    treino_caiu = bool(treino[-1] < treino[melhor_epoca - 1])
    val_subiu = bool(val[-1] > melhor_perda + 1e-4)
    if motivo == "paciencia" and treino_caiu and val_subiu:
        conclusao = f"A validação parou de melhorar na época {melhor_epoca}, enquanto o treino seguiu caindo até a {epoca_parada}"
    elif motivo == "paciencia":
        conclusao = f"O treino parou na época {epoca_parada}, {epoca_parada - melhor_epoca} épocas depois da melhor ({melhor_epoca})"
    else:
        conclusao = f"A paciência não disparou: o treino foi até o limite de {epoca_parada} épocas (melhor época: {melhor_epoca})"
    _titulo(fig, conclusao)
    _rodape(fig, f"{FONTE}. Treino n = {n_treino}, validação n = {n_validacao}. Área cinza: épocas sem melhora maior que 0,0001 até a parada. "
                 f"Restauração: pesos da época {melhor_epoca} devolvidos ao modelo, todos os tensores iguais ao estado guardado (torch.equal); "
                 f"perda de validação recalculada {fmt(float(prova['perda_recalculada']), 8)}, diferença {fmt_e(float(prova['diferenca_absoluta']))} (< 1e-6).")
    return fig


# ---------------------------------------------------------------- D4, D5, D6


def grafico_custos(c_fp: float, c_fn: float) -> Figure:
    """Matriz de custos (D4) e por que o limiar da fórmula é ``C_FP/(C_FP+C_FN)``."""
    fig, (ax_a, ax_b) = _figura(11.5, 4.8, colunas=2)
    matriz = np.array([[0.0, c_fp], [c_fn, 0.0]])
    mapa = matplotlib.colormaps["Greys"]
    ax_a.imshow(matriz, cmap=mapa, vmin=0, vmax=max(c_fp, c_fn) * 1.4)
    nomes = [["acerto", "falso positivo (FP)"], ["falso negativo (FN)", "acerto"]]
    for i in range(2):
        for j in range(2):
            ax_a.text(j, i, f"{nomes[i][j]}\ncusto {fmt(float(matriz[i, j]), 0)}", ha="center", va="center", fontsize=11,
                      color=_cor_texto_sobre(mapa(matriz[i, j] / (max(c_fp, c_fn) * 1.4))))
    ax_a.set_xticks([0, 1], ["previsto 0\n(não seleciona)", "previsto 1\n(seleciona)"])
    ax_a.set_yticks([0, 1], ["real 0\n(nota < 7)", "real 1\n(nota ≥ 7)"])
    ax_a.tick_params(length=0, labelsize=9)
    for lado in ax_a.spines.values():
        lado.set_visible(False)
    p = np.linspace(0, 1, 201)
    formula = c_fp / (c_fp + c_fn)
    _estilo(ax_b, grade="y")
    ax_b.plot(p, c_fp * (1 - p), color=COR["referencia"], linestyle="--", linewidth=2.0)
    ax_b.plot(p, c_fn * p, color=COR["modelo"], linewidth=2.0)
    ax_b.axvline(formula, color=COR["formula"], linestyle="-.", linewidth=1.5)
    ax_b.scatter([formula], [c_fn * formula], s=70, color=COR["formula"], zorder=4)
    ax_b.text(0.62, c_fn * 0.62 + 0.15, "não selecionar: C_FN × p", ha="right", fontsize=9.5, color=COR["modelo"], fontweight="bold", rotation=0)
    ax_b.text(0.5, c_fp * 0.5 + 0.35, "selecionar: C_FP × (1 − p)", ha="left", va="bottom", fontsize=9.5, color=COR_TEXTO_SEC, fontweight="bold")
    ax_b.text(formula + 0.02, c_fn * 0.9, f"empate em p = {fmt(formula, 4)}\n(limiar da fórmula)", ha="left", va="top", fontsize=9.5, color=COR["formula"], fontweight="bold")
    ax_b.set_xlabel("probabilidade p de a amostra ser de nota alta")
    ax_b.set_ylabel("custo esperado (pontos de esforço)")
    ax_b.set_ylim(0, c_fn * 1.05)
    ax_b.xaxis.set_major_formatter(_formatador(1))
    ax_b.yaxis.set_major_formatter(_formatador(0))
    _titulo(fig, f"Deixar passar uma amostra boa (FN) custa {fmt(c_fn / c_fp, 0)}× selecionar uma ruim (FP); "
                 f"com escores calibrados, selecionar compensa a partir de p = {fmt(formula, 4)}")
    _rodape(fig, f"Custos são hipóteses do exercício (C_FP = {fmt(c_fp, 0)}, C_FN = {fmt(c_fn, 0)}, pontos de esforço); a base não traz custo real. "
                 "Acertos custam zero. A fórmula só vale se o escore for probabilidade, o que a D6 examina.")
    return fig


def grafico_custo_limiar(busca: BuscaLimiar, comparacao: pd.DataFrame, n_val: int, n_pos_val: int) -> Figure:
    """Custo total na validação por limiar, com o escolhido, 0,5, a fórmula e os extremos marcados."""
    c = comparacao.set_index("regra")
    t = busca.tabela
    fig, (eixo,) = _figura(10.5, 5.4)
    _estilo(eixo)
    eixo.step(t["limiar"], t["custo_total"], where="post", color=COR["modelo"], linewidth=1.6)
    marcas = (("escolhido na busca", COR["escolhido"], "-.", "o", "escolhido"), ("padrão 0,5", COR["regra_05"], ":", "s", "0,5"),
              ("fórmula C_FP/(C_FP+C_FN)", COR["formula"], "--", "^", "fórmula"))
    deslocamento = {"escolhido": (-8, 28), "0,5": (10, 30), "fórmula": (-20, 100)}
    nomes_curtos = {"escolhido": "escolhido", "0,5": "limiar 0,5", "fórmula": "fórmula"}
    for regra, cor, estilo, marcador, curto in marcas:
        lim, custo = _num(c.loc[regra, "limiar"]), _num(c.loc[regra, "custo_total"])
        eixo.axvline(lim, color=cor, linestyle=estilo, linewidth=1.5)
        eixo.scatter([lim], [custo], marker=marcador, s=75, color=cor, edgecolors="white", zorder=4)
        eixo.annotate(f"{'limiar 0,5' if curto == '0,5' else curto + ' ' + fmt(lim, 4)}\ncusto {fmt(custo, 0)}", (lim, custo), xytext=deslocamento[curto], textcoords="offset points",
                      ha="left" if curto != "escolhido" else "right", fontsize=9.5, color=cor, fontweight="bold",
                      arrowprops={"arrowstyle": "-", "color": cor, "linewidth": 0.8})
    for regra, texto in (("tudo positivo", "tudo positivo"), ("tudo negativo", "tudo negativo")):
        custo = _num(c.loc[regra, "custo_total"])
        eixo.axhline(custo, color=COR["referencia"], linestyle="--", linewidth=1.0)
        eixo.text(1.0 if regra == "tudo negativo" else 0.995, custo + 4, f"{texto}: custo {fmt(custo, 0)}", ha="right", fontsize=9, color=COR_TEXTO_SEC)
    eixo.set_ylim(0, float(t["custo_total"].max()) * 1.08)
    eixo.set_xlim(0, 1.0)
    eixo.set_xlabel("limiar (escore ≥ limiar prevê positivo)")
    eixo.set_ylabel("custo total na validação (pontos de esforço)")
    eixo.xaxis.set_major_formatter(_formatador(1))
    eixo.yaxis.set_major_formatter(_formatador(0))
    esc, p05, frm = (_num(c.loc[r, "custo_total"]) for r in ("escolhido na busca", "padrão 0,5", "fórmula C_FP/(C_FP+C_FN)"))
    if esc < p05:
        comp = "custa menos que"
    elif esc == p05:
        comp = "custa o mesmo que"
    else:
        comp = "custa mais que"
    _titulo(fig, f"O limiar {fmt(busca.limiar, 4)} {comp} 0,5 na validação ({fmt(esc, 0)} contra {fmt(p05, 0)} pontos) e a fórmula custa {fmt(frm, 0)}")
    _rodape(fig, f"{FONTE}. Validação, n = {n_val} ({n_pos_val} positivos); {len(t)} candidatos, {busca.n_empatados} com o custo mínimo. "
                 f"Custos hipotéticos (C_FP = {C_FP}, C_FN = {C_FN}). Otimista: o limiar foi ajustado nesta mesma validação, que também decidiu a parada.")
    return fig


def grafico_calibracao(tabela: pd.DataFrame, n_por_bin: NDArray[np.int_], brier: float, brier_ref: float, n_val: int) -> Figure:
    """Diagrama de calibração na validação (bins por quantis), com n por bin e a diagonal de referência."""
    fig, (eixo,) = _figura(6.8, 6.4)
    _estilo(eixo, grade="both")
    x, y = tabela["escore_medio"].to_numpy(), tabela["fracao_positivos"].to_numpy()
    teto = float(max(x.max(), y.max())) * 1.12
    eixo.plot([0, teto], [0, teto], color=COR["referencia"], linestyle="--", linewidth=1.4)
    eixo.text(teto * 0.55, teto * 0.55 + teto * 0.045, "calibração perfeita", rotation=45, rotation_mode="anchor", fontsize=9, color=COR_TEXTO_SEC, ha="left")
    eixo.plot(x, y, color=COR["modelo"], linewidth=1.6, marker="o", markersize=8, markeredgecolor="white")
    acima = 0
    deslocamentos = [(7, -14), (-13, 5), (-14, 6), (9, 6), (9, -14)]
    for k, (xi, yi) in enumerate(zip(x, y)):
        acima += int(xi > yi)
        eixo.annotate(str(k + 1), (xi, yi), xytext=deslocamentos[k % len(deslocamentos)], textcoords="offset points", fontsize=10,
                      fontweight="bold", color=COR_TEXTO)
    eixo.text(0.03, 0.97, "bin: n\n" + "\n".join(f"{k + 1}: {int(n)}" for k, n in enumerate(n_por_bin)), transform=eixo.transAxes,
              ha="left", va="top", fontsize=9.5, color=COR_TEXTO)
    eixo.set_xlim(0, teto)
    eixo.set_ylim(0, teto)
    eixo.set_aspect("equal")
    eixo.set_xlabel("escore médio previsto no bin")
    eixo.set_ylabel("fração observada de positivos no bin")
    eixo.xaxis.set_major_formatter(_formatador(1))
    eixo.yaxis.set_major_formatter(_formatador(1))
    distancia = np.abs(x - y)
    pior = int(np.argmax(distancia))
    _titulo(fig, f"Em {acima} de {len(x)} bins o escore médio supera a fração observada; a maior distância está no bin {pior + 1} "
                 f"({fmt(float(x[pior]), 3)} contra {fmt(float(y[pior]), 3)})")
    _rodape(fig, f"{FONTE}. Validação, n = {n_val}, {len(x)} bins por quantis. Brier = {fmt(brier, 4)} (escore constante igual à prevalência: {fmt(brier_ref, 4)}). "
                 "Poucos positivos por bin: apoia a discussão, não prova calibração.")
    return fig


# ---------------------------------------------------------------- D1-D3


def _matriz(matrizes: pd.DataFrame, regra: str) -> NDArray[np.int_]:
    sub = matrizes[matrizes["regra"] == regra].pivot(index="real", columns="previsto", values="contagem")
    return sub.loc[[0, 1], [0, 1]].to_numpy(dtype=int)


def grafico_matrizes(matrizes: pd.DataFrame, regras: dict[str, str]) -> Figure:
    """Duas matrizes de confusão (linhas = real, colunas = previsto, ordem [0, 1]) com contagem e % da linha."""
    fig, eixos = _figura(5.4 * len(regras), 5.0, colunas=len(regras))
    mapa = matplotlib.colormaps["Greys"]
    tp: dict[str, int] = {}
    fp: dict[str, int] = {}
    for eixo, (regra, rotulo) in zip(eixos, regras.items()):
        m = _matriz(matrizes, regra)
        linha = m / m.sum(axis=1, keepdims=True)
        eixo.imshow(linha, cmap=mapa, vmin=0, vmax=1)
        for i in range(2):
            for j in range(2):
                eixo.text(j, i, f"{m[i, j]}\n{pct(float(linha[i, j]))}", ha="center", va="center", fontsize=13, fontweight="bold",
                          color=_cor_texto_sobre(mapa(float(linha[i, j]))))
        eixo.set_xticks([0, 1], ["previsto 0", "previsto 1"])
        eixo.set_yticks([0, 1], ["real 0 (nota < 7)", "real 1 (nota ≥ 7)"])
        eixo.tick_params(length=0, labelsize=9)
        for lado in eixo.spines.values():
            lado.set_visible(False)
        eixo.set_title(rotulo, fontsize=10.5, color=COR_TEXTO)
        tp[regra], fp[regra] = int(m[1, 1]), int(m[0, 1])
    (r1, _), (r2, _) = list(regras.items())[0], list(regras.items())[1]
    m1 = _matriz(matrizes, r1)
    total_pos = int(m1[1].sum())
    _titulo(fig, f"Com o limiar escolhido a rede acha {tp[r2]} dos {total_pos} positivos do teste, contra {tp[r1]} com 0,5, "
                 f"ao custo de {abs(fp[r2] - fp[r1])} falsos positivos a {'mais' if fp[r2] >= fp[r1] else 'menos'}")
    _rodape(fig, f"{FONTE}. Teste, n = {int(m1.sum())} ({total_pos} positivos). Cada célula: contagem e % da linha (a cor mostra esse %). "
                 "Mesma escala de cor nos dois painéis.")
    return fig


def grafico_regras(tabela: pd.DataFrame, n_teste: int, n_pos_teste: int) -> Figure:
    """Revocação e custo por observação das três regras, na mesma ordem e com as mesmas cores."""
    t = tabela.set_index("regra")
    ordem = ["Trivial do treino", "Rede: padrão", "Rede: escolhido na validação"]
    cores = [COR["trivial"], COR["regra_05"], COR["escolhido"]]
    hachuras = ["//", "", ""]
    rotulos = ["trivial\n(sempre 0)", "rede\nlimiar 0,5", f"rede, limiar\nescolhido ({fmt(_num(t.loc[ordem[2], 'limiar']), 4)})"]
    fig, (ax_a, ax_b) = _figura(11, 5.0, colunas=2)
    for eixo, coluna, rotulo_y, casas in ((ax_a, "revocacao", "revocação (fração dos positivos encontrada)", 3),
                                          (ax_b, "custo_por_obs", "custo por observação (pontos de esforço)", 4)):
        _estilo(eixo)
        valores = [_num(t.loc[r, coluna]) for r in ordem]
        for i, v in enumerate(valores):
            eixo.bar(i, v, color=cores[i], hatch=hachuras[i], edgecolor="white" if not hachuras[i] else COR_TEXTO_SEC, width=0.62)
            extra = f"\n{int(_num(t.loc[ordem[i], 'fp']))} FP, {int(_num(t.loc[ordem[i], 'fn']))} FN" if coluna == "custo_por_obs" else ""
            eixo.text(i, v + max(valores) * 0.02, f"{fmt(v, casas)}{extra}", ha="center", va="bottom", fontsize=9.5, color=COR_TEXTO)
        eixo.set_xticks(range(3), rotulos, fontsize=9)
        eixo.set_ylim(0, max(max(valores) * 1.25, 0.1))
        eixo.set_ylabel(rotulo_y)
        eixo.yaxis.set_major_formatter(_formatador(2))
    acuracias = ", ".join(f"{n} {pct(_num(t.loc[r, 'acuracia']))}" for n, r in zip(("trivial", "rede 0,5", "rede escolhido"), ordem))
    _titulo(fig, f"O trivial acerta {pct(_num(t.loc[ordem[0], 'acuracia']))} mas encontra {'nenhum' if _num(t.loc[ordem[0], 'revocacao']) == 0 else fmt(_num(t.loc[ordem[0], 'revocacao']), 2)} dos positivos; "
                 f"o limiar escolhido encontra {fmt(_num(t.loc[ordem[2], 'revocacao']), 2)} e custa {fmt(_num(t.loc[ordem[2], 'custo_por_obs']), 4)} por observação "
                 f"contra {fmt(_num(t.loc[ordem[0], 'custo_por_obs']), 4)}")
    _rodape(fig, f"{FONTE}. Teste, n = {n_teste} ({n_pos_teste} positivos); um único conjunto, diferenças pequenas ficam dentro do ruído. "
                 f"Acurácia: {acuracias}. Custo = (C_FP×FP + C_FN×FN)/n com C_FP = {C_FP}, C_FN = {C_FN}, hipotéticos.")
    return fig


def grafico_roc_pr(rotulos: NDArray[np.int_], escores: NDArray[np.float64], limiar: float, areas: dict[str, float]) -> Figure:
    """ROC (com a diagonal aleatória) e PR (com a linha da prevalência do teste), lado a lado, com os dois pontos de operação."""
    fpr, tpr, _ = roc_curve(rotulos, escores)
    precisao, revocacao, _ = precision_recall_curve(rotulos, escores)
    fig, (ax_r, ax_p) = _figura(11.5, 5.8, colunas=2)
    for eixo in (ax_r, ax_p):
        _estilo(eixo, grade="both")
        eixo.set_xlim(0, 1)
        eixo.set_ylim(0, 1.02)
        eixo.set_aspect("equal")
        eixo.xaxis.set_major_formatter(_formatador(1))
        eixo.yaxis.set_major_formatter(_formatador(1))
    ax_r.plot(fpr, tpr, color=COR["modelo"], linewidth=2.0)
    ax_r.plot([0, 1], [0, 1], color=COR["referencia"], linestyle="--", linewidth=1.4)
    ax_r.text(0.55, 0.47, "aleatório (diagonal, área 0,5)", rotation=45, rotation_mode="anchor", fontsize=9, color=COR_TEXTO_SEC, ha="left", va="top")
    ax_r.text(0.42, 0.12, f"rede: AUC-ROC = {fmt(areas['auc_roc'], 3)}", fontsize=10, color=COR["modelo"], fontweight="bold", ha="left")
    prev = areas["prevalencia_teste"]
    ax_p.plot(revocacao, precisao, color=COR["modelo"], linewidth=2.0)
    ax_p.axhline(prev, color=COR["referencia"], linestyle="--", linewidth=1.4)
    ax_p.text(0.99, prev + 0.015, f"aleatório: prevalência do teste = {fmt(prev, 3)}", ha="right", fontsize=9, color=COR_TEXTO_SEC)
    ax_p.text(0.55, 0.90, f"rede: AUC-PR = {fmt(areas['auc_pr'], 3)} (trapezoidal)\nAP = {fmt(areas['ap'], 3)}", fontsize=10,
              color=COR["modelo"], fontweight="bold", ha="left", va="top")
    regras = (("limiar 0,5", 0.5, COR["regra_05"], "s", (12, -22)), (f"limiar escolhido ({fmt(limiar, 4)})", limiar, COR["escolhido"], "o", (14, -26)))
    for nome, t, cor, marcador, desloc in regras:
        previsto = prever(escores, t)
        pos = rotulos == 1
        tpr_t, fpr_t = float(previsto[pos].mean()), float(previsto[~pos].mean())
        prec_t = float(rotulos[previsto == 1].mean()) if previsto.any() else 0.0
        ax_r.scatter([fpr_t], [tpr_t], color=cor, marker=marcador, s=90, edgecolors="white", zorder=4)
        ax_r.annotate(f"{nome}\nrevoc. {fmt(tpr_t, 2)}, FPR {fmt(fpr_t, 2)}", (fpr_t, tpr_t), xytext=desloc, textcoords="offset points", fontsize=9,
                      color=cor, fontweight="bold", arrowprops={"arrowstyle": "-", "color": cor, "linewidth": 0.8})
        ax_p.scatter([tpr_t], [prec_t], color=cor, marker=marcador, s=90, edgecolors="white", zorder=4)
        ax_p.annotate(f"{nome}\nprec. {fmt(prec_t, 2)}, revoc. {fmt(tpr_t, 2)}", (tpr_t, prec_t), xytext=(14, 24) if marcador == "s" else (16, 34),
                      textcoords="offset points", fontsize=9, color=cor, fontweight="bold", arrowprops={"arrowstyle": "-", "color": cor, "linewidth": 0.8})
    ax_r.set_xlabel("taxa de falsos positivos (FPR)")
    ax_r.set_ylabel("taxa de verdadeiros positivos (revocação)")
    ax_p.set_xlabel("revocação")
    ax_p.set_ylabel("precisão")
    ax_r.set_title("ROC", fontsize=10.5, loc="left", color=COR_TEXTO)
    ax_p.set_title("precisão × revocação", fontsize=10.5, loc="left", color=COR_TEXTO)
    acima = "acima" if areas["auc_pr"] > prev else "abaixo"
    _titulo(fig, f"No teste a rede tem AUC-ROC {fmt(areas['auc_roc'], 3)} e a AUC-PR ({fmt(areas['auc_pr'], 3)}) fica {acima} da prevalência ({fmt(prev, 3)})")
    _rodape(fig, f"{FONTE}. Teste, n = {len(rotulos)} ({int(rotulos.sum())} positivos). As áreas valem para as duas regras: o limiar muda as classes, não o ranking dos escores. "
                 "Quadrado = limiar 0,5; círculo = limiar escolhido na validação.")
    return fig


# ---------------------------------------------------------------- auditoria da paleta


def _linear(hex_cor: str) -> NDArray[np.float64]:
    rgb = np.array([int(hex_cor[i : i + 2], 16) / 255 for i in (1, 3, 5)])
    return np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)


def _lab(rgb_linear: NDArray[np.float64]) -> NDArray[np.float64]:
    xyz = np.array([[0.4124, 0.3576, 0.1805], [0.2126, 0.7152, 0.0722], [0.0193, 0.1192, 0.9505]]) @ rgb_linear
    xyz = xyz / np.array([0.95047, 1.0, 1.08883])
    f = np.where(xyz > 216 / 24389, np.cbrt(xyz), (24389 / 27 * xyz + 16) / 116)
    return np.array([116 * f[1] - 16, 500 * (f[0] - f[1]), 200 * (f[1] - f[2])])


_DEUTERANOPIA = np.array([[0.367322, 0.860646, -0.227968], [0.280085, 0.672501, 0.047413], [-0.011820, 0.042940, 0.968881]])


def _lab_deuteranopia(hex_cor: str) -> NDArray[np.float64]:
    """Lab da cor como a vê uma pessoa com deuteranopia (Machado et al., 2009, severidade 1,0, em RGB linear)."""
    return _lab(np.clip(_DEUTERANOPIA @ _linear(hex_cor), 0.0, 1.0))


def auditar_paleta() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Auditoria calculada da paleta.

    Tabela 1, por cor: contraste WCAG contra branco (gráfico >= 3:1, texto >= 4,5:1),
    cor simulada em deuteranopia e luminosidade L* (para a leitura em tons de cinza).
    Tabela 2, por par que precisa ser distinguido: ΔE (CIE 1976) em deuteranopia e
    diferença de L*. Nenhum par depende só da cor: todos também têm marcador, traço ou rótulo.
    """
    linhas = []
    for nome, (hex_cor, uso) in AUDITORIA_CORES.items():
        razao = _contraste("#FFFFFF", hex_cor)
        minimo = 4.5 if uso == "texto" else 3.0
        sim = np.clip(_DEUTERANOPIA @ _linear(hex_cor), 0, 1)
        sim_hex = "#" + "".join(f"{int(round(float(255 * (12.92 * c if c <= 0.0031308 else 1.055 * c ** (1 / 2.4) - 0.055)))):02X}" for c in sim)
        linhas.append({"cor": nome, "hex": hex_cor, "uso": uso, "contraste_vs_branco": round(razao, 2), "minimo_exigido": minimo,
                       "passa": bool(razao >= minimo), "hex_deuteranopia": sim_hex, "L_estrela": round(float(_lab(_linear(hex_cor))[0]), 1)})
    pares = []
    for a, b, onde, reforco in AUDITORIA_PARES:
        ha, hb = AUDITORIA_CORES[a][0], AUDITORIA_CORES[b][0]
        de = float(np.linalg.norm(_lab_deuteranopia(ha) - _lab_deuteranopia(hb)))
        dl = abs(float(_lab(_linear(ha))[0] - _lab(_linear(hb))[0]))
        pares.append({"cor_a": a, "cor_b": b, "onde": onde, "delta_E_deuteranopia": round(de, 1), "delta_L_cinza": round(dl, 1),
                      "so_cor_basta": bool(de >= LIMITE_DELTA_E and dl >= LIMITE_DELTA_L), "reforco_nao_cromatico": reforco})
    return pd.DataFrame(linhas), pd.DataFrame(pares)
