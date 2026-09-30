"""Carga, limpeza e divisão dos dados de interação usuário-item."""
from dataclasses import dataclass
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy import sparse

ROOT = Path(__file__).resolve().parents[2]


def load_config(path: str | Path = ROOT / "config.yaml") -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _resolve(p: str | None) -> Path | None:
    return None if p is None else (ROOT / p)


def load_interactions(cfg: dict) -> pd.DataFrame:
    """Lê as interações e padroniza as colunas para user, item, rating, timestamp."""
    ds = cfg["dataset"]
    cols = ds["columns"]
    # ids de item lidos como texto: ISBNs têm zeros à esquerda e letras (ex.: 034545104X)
    raw = pd.read_csv(_resolve(ds["interactions_path"]), sep=ds.get("sep", ","),
                      dtype={cols["item"]: str})

    df = pd.DataFrame({"user": raw[cols["user"]], "item": raw[cols["item"]]})
    # Sem coluna de nota -> feedback implícito (toda interação vale 1.0)
    df["rating"] = raw[cols["rating"]] if cols.get("rating") else 1.0
    df["timestamp"] = raw[cols["timestamp"]] if cols.get("timestamp") else np.arange(len(raw))

    return df


def load_new_ratings(cfg: dict) -> pd.DataFrame:
    """Avaliações gravadas pela interface (vazio se ainda não houver)."""
    path = _resolve(cfg.get("storage", {}).get("new_ratings_path"))
    if path is None or not path.exists():
        return pd.DataFrame(columns=["user", "item", "rating", "timestamp"])
    return pd.read_csv(path, dtype={"item": str})


def append_rating(cfg: dict, user, item, rating: float) -> None:
    """Grava uma avaliação feita na interface (a mais recente prevalece na leitura)."""
    path = _resolve(cfg["storage"]["new_ratings_path"])
    path.parent.mkdir(parents=True, exist_ok=True)
    row = pd.DataFrame([{"user": user, "item": item, "rating": rating,
                         "timestamp": int(time.time())}])
    row.to_csv(path, mode="a", header=not path.exists(), index=False)


def next_user_id(ds: "Dataset"):
    """Id para um novo usuário, no mesmo formato dos ids da base (numérico ou texto)."""
    ids = ds.user_ids
    if np.issubdtype(ids.dtype, np.integer):
        return int(ids.max()) + 1
    return f"novo_{len(ids) + 1}"


def load_items(cfg: dict) -> pd.DataFrame:
    """Catálogo de itens (título e campo extra opcional). Indexado pelo id do item."""
    ds = cfg["dataset"]
    path = _resolve(ds.get("items_path"))
    if path is None or not path.exists():
        return pd.DataFrame(columns=["title", "extra"])
    ic = ds["items_columns"]
    raw = pd.read_csv(path, sep=ds.get("sep", ","), dtype={ic["item"]: str})
    def col(key):
        return raw[ic[key]].fillna("").astype(str) if ic.get(key) else ""

    items = pd.DataFrame({
        "item": raw[ic["item"]],
        "title": raw[ic["title"]].astype(str),
        "extra": col("extra"),
        "author": col("author"),
        "year": col("year"),
    })
    return items.drop_duplicates("item").set_index("item")


def clean(df: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, dict]:
    """Tratamento: nulos, tipos, notas fora da escala, duplicatas e filtro de esparsidade.

    Retorna o dataframe limpo e um relatório com o que foi removido em cada etapa.
    """
    lo, hi = cfg["dataset"]["rating_scale"]
    pre = cfg["preprocessing"]
    report = {"linhas_originais": len(df)}

    df = df.dropna(subset=["user", "item", "rating"])
    report["apos_remover_nulos"] = len(df)

    df = df.assign(rating=pd.to_numeric(df["rating"], errors="coerce")).dropna(subset=["rating"])
    df = df[(df["rating"] >= lo) & (df["rating"] <= hi)]
    report["apos_filtrar_escala"] = len(df)

    # Mesma dupla usuário-item: mantém a avaliação mais recente
    df = df.sort_values("timestamp").drop_duplicates(["user", "item"], keep="last")
    report["apos_remover_duplicatas"] = len(df)

    # Filtro iterativo: remover itens/usuários pouco ativos pode derrubar outros
    while True:
        n = len(df)
        df = df[df.groupby("item")["user"].transform("size") >= pre["min_item_interactions"]]
        df = df[df.groupby("user")["item"].transform("size") >= pre["min_user_interactions"]]
        if len(df) == n:
            break
    report["apos_filtro_minimo_interacoes"] = len(df)
    return df.reset_index(drop=True), report


@dataclass
class Dataset:
    """Interações limpas + mapeamentos id <-> índice + matriz esparsa usuário x item."""

    df: pd.DataFrame
    items: pd.DataFrame
    user_ids: np.ndarray
    item_ids: np.ndarray
    matrix: sparse.csr_matrix  # notas (0 = sem avaliação)

    @property
    def n_users(self) -> int:
        return len(self.user_ids)

    @property
    def n_items(self) -> int:
        return len(self.item_ids)

    def user_index(self, user_id) -> int | None:
        return self._uidx.get(user_id)

    def item_index(self, item_id) -> int | None:
        return self._iidx.get(item_id)

    def title(self, item_id) -> str:
        if item_id in self.items.index:
            return self.items.at[item_id, "title"]
        return str(item_id)

    def extra(self, item_id) -> str:
        if item_id in self.items.index:
            return self.items.at[item_id, "extra"]
        return ""

    def author(self, item_id) -> str:
        if item_id in self.items.index:
            return self.items.at[item_id, "author"]
        return ""

    def year(self, item_id) -> str:
        if item_id in self.items.index:
            return self.items.at[item_id, "year"]
        return ""

    def __post_init__(self):
        self._uidx = {u: i for i, u in enumerate(self.user_ids)}
        self._iidx = {v: i for i, v in enumerate(self.item_ids)}


def build_dataset(df: pd.DataFrame, items: pd.DataFrame,
                  user_ids=None, item_ids=None) -> Dataset:
    """Monta a matriz esparsa. Passar user_ids/item_ids mantém o mesmo espaço de índices
    (usado para que treino e teste sejam comparáveis)."""
    user_ids = np.sort(df["user"].unique()) if user_ids is None else np.asarray(user_ids)
    item_ids = np.sort(df["item"].unique()) if item_ids is None else np.asarray(item_ids)
    uidx = pd.Series(np.arange(len(user_ids)), index=user_ids)
    iidx = pd.Series(np.arange(len(item_ids)), index=item_ids)
    rows = uidx.reindex(df["user"]).to_numpy()
    cols = iidx.reindex(df["item"]).to_numpy()
    ok = ~(np.isnan(rows) | np.isnan(cols))
    matrix = sparse.csr_matrix(
        (df["rating"].to_numpy(dtype=float)[ok], (rows[ok].astype(int), cols[ok].astype(int))),
        shape=(len(user_ids), len(item_ids)),
    )
    return Dataset(df=df.reset_index(drop=True), items=items,
                   user_ids=user_ids, item_ids=item_ids, matrix=matrix)


def load_dataset(cfg: dict, include_new: bool = True) -> tuple[Dataset, dict]:
    """Carrega + trata a base. As avaliações da interface entram depois da limpeza, para que
    um usuário novo (com poucas avaliações) não seja removido pelo filtro de mínimo."""
    df, report = clean(load_interactions(cfg), cfg)
    if include_new:
        new = load_new_ratings(cfg)
        if not new.empty:
            new = new.astype({"user": df["user"].dtype, "item": df["item"].dtype})
            new = new[new["item"].isin(df["item"].unique())]
            df = (pd.concat([df, new[df.columns]], ignore_index=True)
                  .sort_values("timestamp").drop_duplicates(["user", "item"], keep="last"))
            report["avaliacoes_da_interface"] = len(new)
    return build_dataset(df, load_items(cfg)), report


def train_test_split(ds: Dataset, cfg: dict) -> tuple[Dataset, pd.DataFrame]:
    """Hold-out por usuário: uma fração das interações de cada usuário com histórico
    suficiente vai para teste (as mais recentes se split="temporal"; sorteadas, com semente
    fixa, se split="random" — para bases sem data). Os demais usuários ficam só no treino."""
    ev = cfg["evaluation"]
    if ev.get("split", "temporal") == "random":
        key = np.random.default_rng(ev.get("seed", 42)).random(len(ds.df))
        df = ds.df.assign(_key=key).sort_values(["user", "_key"]).drop(columns="_key")
    else:
        df = ds.df.sort_values(["user", "timestamp", "item"])
    rank = df.groupby("user").cumcount()
    size = df.groupby("user")["item"].transform("size")
    n_test = np.floor(size * ev["test_fraction"]).astype(int)
    eligible = size >= ev["min_user_interactions_for_test"]
    is_test = eligible & (rank >= size - n_test)
    train = build_dataset(df[~is_test], ds.items, ds.user_ids, ds.item_ids)
    return train, df[is_test].reset_index(drop=True)


def dataset_summary(ds: Dataset) -> dict:
    n = ds.matrix.nnz
    return {
        "usuarios": ds.n_users,
        "itens": ds.n_items,
        "interacoes": int(n),
        "esparsidade_%": round(100 * (1 - n / (ds.n_users * ds.n_items)), 2),
        "nota_media": round(float(ds.df["rating"].mean()), 3),
    }
