"""Registro opcional dos experimentos no MLflow (um único ``mlflow.db`` em SQLite).

O registro nunca pode quebrar o laboratório: falhas do MLflow ou do SQLite viram
um aviso e o notebook segue. ``LAB02_TRACKING=0`` desliga tudo (as funções
viram no-ops e o MLflow nem é importado). Pesos de modelo não são registrados
e não se usa autolog.
"""
from __future__ import annotations

import logging
import os
import sqlite3
import warnings
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path

import pandas as pd

BANCO = Path(__file__).resolve().parent.parent / "mlflow.db"
TRACKING_URI = "sqlite:///" + str(BANCO)
TABELA_BLOBS = "lab02_artifact_blobs"

# O MLflow imprime uma dica de "skills" ao ser importado; esta variável de ambiente é a configuração oficial para desligá-la.
os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

Valor = str | int | float | bool | None


def _silenciar_info() -> None:
    """Configuração de logging (não é filtro de avisos): linhas INFO de migração do banco poluiriam o notebook.

    Precisa rodar depois de ``import mlflow``, que reconfigura o logger ``mlflow`` na importação.
    """
    logging.getLogger("alembic").setLevel(logging.WARNING)
    logging.getLogger("mlflow").setLevel(logging.WARNING)


def habilitado() -> bool:
    """O registro fica ligado a menos que ``LAB02_TRACKING=0``."""
    return os.environ.get("LAB02_TRACKING", "1") != "0"


def _avisar(acao: str, erro: Exception) -> None:
    warnings.warn(
        f"MLflow: falha ao {acao} ({type(erro).__name__}: {erro}). O laboratório continua sem registrar. "
        "Para desligar o registro, defina LAB02_TRACKING=0.",
        stacklevel=3,
    )


def setup() -> None:
    """Aponta o MLflow para o ``mlflow.db`` ao lado de ``lab02lib/`` (nunca para ``mlruns/``)."""
    import mlflow

    _silenciar_info()
    mlflow.set_tracking_uri(TRACKING_URI)


@contextmanager
def run(
    experimento: str,
    nome: str,
    params: Mapping[str, Valor] | None = None,
    tags: Mapping[str, Valor] | None = None,
) -> Iterator[str | None]:
    """Abre uma execução do MLflow e entrega o ``run_id`` (ou ``None`` se o registro estiver desligado ou falhar).

    Valores ``None`` em ``params`` e ``tags`` são descartados. Se já houver uma
    execução ativa, a nova fica aninhada. Só ``MlflowException`` e
    ``sqlite3.OperationalError`` são tratados; erros do código do chamador sobem.
    """
    if not habilitado():
        yield None
        return
    import mlflow
    from mlflow.exceptions import MlflowException

    _silenciar_info()
    run_id: str | None = None
    try:
        setup()
        mlflow.set_experiment(experimento)
        ativo = mlflow.start_run(run_name=nome, nested=mlflow.active_run() is not None)
        run_id = ativo.info.run_id
        mlflow.log_params({k: v for k, v in (params or {}).items() if v is not None})
        mlflow.set_tags({k: str(v) for k, v in (tags or {}).items() if v is not None})
    except (MlflowException, sqlite3.OperationalError) as erro:
        _avisar("abrir a execução", erro)
    try:
        yield run_id
    finally:
        if run_id is not None:
            try:
                mlflow.end_run()
            except (MlflowException, sqlite3.OperationalError) as erro:
                _avisar("fechar a execução", erro)


def log_metrics(metricas: Mapping[str, float], step: int | None = None) -> None:
    """Registra métricas na execução ativa; sem execução ativa ou com falha, não faz nada (avisa)."""
    if not habilitado():
        return
    import mlflow
    from mlflow.exceptions import MlflowException

    _silenciar_info()
    if mlflow.active_run() is None:
        return
    try:
        mlflow.log_metrics({k: float(v) for k, v in metricas.items()}, step=step)
    except (MlflowException, sqlite3.OperationalError) as erro:
        _avisar("registrar métricas", erro)


def save_blob(run_id: str | None, name: str, content_type: str, data: bytes) -> None:
    """Guarda um arquivo pequeno (CSV/PNG) como BLOB na tabela ``lab02_artifact_blobs`` do mesmo ``mlflow.db``.

    Deve ser chamado depois que a execução foi fechada. ``INSERT OR REPLACE`` por
    ``(run_id, name)``.
    """
    if run_id is None or not habilitado():
        return
    conexao: sqlite3.Connection | None = None
    try:
        conexao = sqlite3.connect(BANCO, timeout=30)
        conexao.execute(
            f"CREATE TABLE IF NOT EXISTS {TABELA_BLOBS} "
            "(run_id TEXT, name TEXT, content_type TEXT, data BLOB, PRIMARY KEY (run_id, name))"
        )
        conexao.execute(f"INSERT OR REPLACE INTO {TABELA_BLOBS} VALUES (?, ?, ?, ?)", (run_id, name, content_type, data))
        conexao.commit()
    except sqlite3.OperationalError as erro:
        _avisar("guardar o artefato", erro)
    finally:
        if conexao is not None:
            conexao.close()


def list_names(run_id: str) -> list[str]:
    """Nomes dos artefatos guardados para uma execução."""
    conexao = sqlite3.connect(BANCO, timeout=30)
    try:
        linhas = conexao.execute(f"SELECT name FROM {TABELA_BLOBS} WHERE run_id = ? ORDER BY name", (run_id,)).fetchall()
    finally:
        conexao.close()
    return [str(linha[0]) for linha in linhas]


def _guardar_arquivos(run_id: str | None, arquivos: list[Path]) -> None:
    """Guarda cada arquivo existente como BLOB da execução (CSV como text/csv, PNG como image/png)."""
    tipos = {".csv": "text/csv", ".png": "image/png", ".json": "application/json", ".txt": "text/plain"}
    for caminho in arquivos:
        save_blob(run_id, caminho.name, tipos.get(caminho.suffix, "application/octet-stream"), caminho.read_bytes())


def registrar_tempos(tempos: list[dict[str, float | int]], hardware: Mapping[str, str], pasta: Path) -> None:
    """A3: uma execução por tamanho de lote, repetições como ``step``. Chamar só depois de terminar as medições."""
    for lote in sorted({int(t["batch_size"]) for t in tempos}):
        medidas = sorted((t for t in tempos if int(t["batch_size"]) == lote), key=lambda t: int(t["repeticao"]))
        params: dict[str, Valor] = {"batch_size": lote, **hardware}
        with run("lab02-a-timing", f"lote-{lote}", params, {"parte": "A3"}) as run_id:
            for t in medidas:
                log_metrics({"segundos": float(t["segundos"]), "segundos_por_passo": float(t["segundos_por_passo"])}, step=int(t["repeticao"]))
        _guardar_arquivos(run_id, [pasta / "a3_tempos.csv", pasta / "a3_resumo.csv", pasta / "a3_tempo_x_lote.png"])


def registrar_vazamento(c_resultado: pd.DataFrame, pasta: Path) -> None:
    """C: uma execução por forma × variante × semente."""
    for r in c_resultado.to_dict("records"):
        nome = f"forma{int(r['forma'])}-{r['variante']}-seed{int(r['seed'])}"
        tags: dict[str, Valor] = {"forma": int(r["forma"]), "variante": str(r["variante"]), "seed": int(r["seed"])}
        with run("lab02-c-leakage", nome, {"epocas": 100, "batch_size": 64}, tags) as run_id:
            log_metrics({k: float(r[k]) for k in ("acuracia", "auc_roc", "acuracia_majoritaria", "n_treino", "n_avaliacao")})
        _guardar_arquivos(run_id, [pasta / "c_resultados.csv"])


def registrar_treino(historico: pd.DataFrame, melhor_epoca: int, epoca_parada: int, motivo: str, pasta: Path) -> None:
    """D7: execução ``train-es`` com perdas por época (``step=epoca``)."""
    params: dict[str, Valor] = {"max_epocas": 100, "paciencia": 10, "min_delta": 1e-4, "lr": 1e-3, "semente": 42, "motivo_parada": motivo}
    with run("lab02-d-main", "train-es", params, {"parte": "D7"}) as run_id:
        for r in historico.to_dict("records"):
            log_metrics({"train_loss": float(r["perda_treino"]), "val_loss": float(r["perda_validacao"])}, step=int(r["epoca"]))
        log_metrics({"melhor_epoca": melhor_epoca, "epoca_parada": epoca_parada})
    _guardar_arquivos(run_id, [pasta / "d7_historico.csv", pasta / "d7_curvas_perda.png"])


def registrar_avaliacao(tabela: pd.DataFrame, areas: Mapping[str, float], limiar: float, pasta: Path) -> None:
    """D1-D3: execução ``eval-test``; ``eval/*`` é a regra congelada (limiar escolhido), ``eval_0p5/*`` o limiar 0,5."""
    params: dict[str, Valor] = {"limiar": limiar, "C_FP": 1, "C_FN": 5}
    with run("lab02-d-main", "eval-test", params, {"parte": "D1-D3"}) as run_id:
        for prefixo, regra in (("eval", "Rede: escolhido na validação"), ("eval_0p5", "Rede: padrão")):
            r = tabela[tabela["regra"] == regra].to_dict("records")[0]
            log_metrics(
                {
                    f"{prefixo}/accuracy": float(r["acuracia"]),
                    f"{prefixo}/precision": float(r["precisao"]),
                    f"{prefixo}/recall": float(r["revocacao"]),
                    f"{prefixo}/f1": float(r["f1"]),
                    f"{prefixo}/auc_roc": areas["auc_roc"],
                    f"{prefixo}/auc_pr": areas["auc_pr"],
                    f"{prefixo}/cost_per_obs": float(r["custo_por_obs"]),
                }
            )
    _guardar_arquivos(run_id, [pasta / "d_tabela_final.csv", pasta / "d_matrizes.csv", pasta / "d_curvas_roc_pr.png"])
