"""Recomendadores: popularidade (baseline / cold-start), aleatório e filtragem colaborativa."""
import numpy as np
from scipy import sparse

from .data import Dataset


class BaseRecommender:
    """Interface comum. Subclasses implementam `_fit` e `scores`."""

    name = "base"

    def fit(self, ds: Dataset):
        self.ds = ds
        self._fit(ds)
        return self

    def _fit(self, ds: Dataset):
        pass

    def scores(self, u_idx: int) -> np.ndarray:
        """Pontuação de todos os itens para o usuário (maior = melhor)."""
        raise NotImplementedError

    def has_history(self, user_id) -> bool:
        u = self.ds.user_index(user_id)
        return u is not None and self.ds.matrix.indptr[u + 1] > self.ds.matrix.indptr[u]

    def recommend(self, user_id, n: int = 10, exclude_seen: bool = True) -> list[tuple]:
        """Top-N (item_id, score). Itens já conhecidos pelo usuário são excluídos.

        Usuário sem histórico (cold-start) recebe os itens mais populares.
        """
        if not self.has_history(user_id):
            return self._cold_start(n)
        u = self.ds.user_index(user_id)
        s = self.scores(u).astype(float).copy()
        if exclude_seen:
            seen = self.ds.matrix[u].indices
            s[seen] = -np.inf
        return self._top(s, n)

    def _cold_start(self, n: int) -> list[tuple]:
        pop = self if isinstance(self, PopularityRecommender) else PopularityRecommender().fit(self.ds)
        return pop._top(pop.item_scores.copy(), n)

    def _top(self, s: np.ndarray, n: int) -> list[tuple]:
        n = min(n, int(np.isfinite(s).sum()))
        idx = np.argpartition(-s, n - 1)[:n] if n > 0 else np.array([], dtype=int)
        idx = idx[np.argsort(-s[idx], kind="stable")]
        return [(self.ds.item_ids[i], float(s[i])) for i in idx]


class PopularityRecommender(BaseRecommender):
    """Média bayesiana das notas: (v*R + m*C) / (v + m).

    v = nº de avaliações do item, R = nota média do item, C = média global,
    m = mínimo de votos. Evita que itens com poucos votos dominem o ranking.
    """

    name = "Popularidade"

    def __init__(self, min_votes: int = 20):
        self.min_votes = min_votes

    def _fit(self, ds: Dataset):
        m = ds.matrix
        votes = np.diff(m.tocsc().indptr).astype(float)
        sums = np.asarray(m.sum(axis=0), dtype=float).ravel()
        mean_item = np.divide(sums, votes, out=np.zeros_like(sums), where=votes > 0)
        c = m.data.mean()
        self.item_scores = (votes * mean_item + self.min_votes * c) / (votes + self.min_votes)

    def scores(self, u_idx: int) -> np.ndarray:
        return self.item_scores

    def predict_ratings(self, u_idx: int) -> np.ndarray:
        return self.item_scores


class RandomRecommender(BaseRecommender):
    """Baseline mínimo de comparação (sorteia itens)."""

    name = "Aleatório"

    def __init__(self, seed: int = 42):
        self.seed = seed

    def scores(self, u_idx: int) -> np.ndarray:
        return np.random.default_rng(self.seed + u_idx).random(self.ds.n_items)


def _center_by_user(m: sparse.csr_matrix) -> tuple[sparse.csr_matrix, np.ndarray]:
    """Subtrai de cada nota a média do usuário (só nas posições avaliadas).

    Em feedback implícito (todas as notas iguais) não há o que centrar: centrar zeraria
    a matriz inteira, então os valores originais são mantidos.
    """
    counts = np.diff(m.indptr)
    means = np.divide(np.asarray(m.sum(axis=1)).ravel(), counts,
                      out=np.zeros(m.shape[0]), where=counts > 0)
    if np.ptp(m.data) == 0:
        means = np.zeros(m.shape[0])
    c = m.copy().astype(np.float64)
    c.data = c.data - np.repeat(means, counts)
    return c, means


def _cosine_topk(x: sparse.csr_matrix, b: sparse.csr_matrix,
                 k: int, shrinkage: float) -> sparse.csr_matrix:
    """Similaridade cosseno entre as LINHAS de x, com shrinkage e só os k mais similares.

    b é a matriz binária (avaliado / não avaliado); n_ij = co-avaliações.
    Shrinkage: sim * n_ij / (n_ij + shrinkage) reduz o peso de pares com poucas evidências.
    """
    norms = np.sqrt(np.asarray(x.multiply(x).sum(axis=1)).ravel())
    norms[norms == 0] = 1.0
    dot = (x @ x.T).toarray().astype(np.float32)
    co = (b @ b.T).toarray().astype(np.float32)
    sim = dot / norms[:, None] / norms[None, :]
    sim *= co / (co + shrinkage)
    np.fill_diagonal(sim, 0.0)
    k = min(k, sim.shape[0] - 1)
    if k < sim.shape[0] - 1:
        drop = np.argpartition(-np.abs(sim), k, axis=1)[:, k:]
        np.put_along_axis(sim, drop, 0.0, axis=1)
    return sparse.csr_matrix(sim)


class ItemKNN(BaseRecommender):
    """Filtragem colaborativa item-based.

    Itens são parecidos se os mesmos usuários os avaliam de forma parecida (cosseno sobre
    notas centradas na média do usuário). O item i é pontuado pelas notas que o usuário
    deu aos itens mais similares a i.
    """

    name = "Item-based CF"

    def __init__(self, neighbors: int = 30, shrinkage: float = 10):
        self.neighbors, self.shrinkage = neighbors, shrinkage

    def _fit(self, ds: Dataset):
        centered, self.user_means = _center_by_user(ds.matrix)
        self.centered = centered.tocsr()
        self.binary = (ds.matrix > 0).astype(np.float32).tocsr()
        self.sim = _cosine_topk(centered.T.tocsr(), self.binary.T.tocsr(),
                                self.neighbors, self.shrinkage)
        self.abs_sim = abs(self.sim)

    def _num_den(self, u_idx: int):
        rc = self.centered[u_idx].T          # itens x 1
        bu = self.binary[u_idx].T
        num = np.asarray((self.sim @ rc).todense()).ravel()
        den = np.asarray((self.abs_sim @ bu).todense()).ravel()
        return num, den

    def scores(self, u_idx: int) -> np.ndarray:
        """Score de ranking: soma ponderada dos desvios de nota nos vizinhos.

        Diferente da predição de nota, não divide por sum|sim|: itens com mais
        evidência (mais vizinhos avaliados) sobem no ranking.
        """
        num, _ = self._num_den(u_idx)
        return num

    def predict_ratings(self, u_idx: int) -> np.ndarray:
        """Predição de nota: média do usuário + média ponderada dos desvios nos vizinhos."""
        num, den = self._num_den(u_idx)
        pred = np.full(self.ds.n_items, np.nan)
        ok = den > 0
        pred[ok] = self.user_means[u_idx] + num[ok] / den[ok]
        lo, hi = self.ds.matrix.data.min(), self.ds.matrix.data.max()
        return np.clip(pred, lo, hi)

    def similar_items(self, item_id, n: int = 5) -> list[tuple]:
        i = self.ds.item_index(item_id)
        row = self.sim[i].toarray().ravel()
        idx = np.argsort(-row)[:n]
        return [(self.ds.item_ids[j], float(row[j])) for j in idx if row[j] > 0]

    def explain(self, user_id, item_id, n: int = 3) -> list[tuple]:
        """Itens do histórico do usuário que mais contribuíram para recomendar item_id."""
        u, i = self.ds.user_index(user_id), self.ds.item_index(item_id)
        rated = self.ds.matrix[u].indices
        sims = self.sim[i, rated].toarray().ravel()
        order = np.argsort(-sims)[:n]
        return [(self.ds.item_ids[rated[j]], float(sims[j])) for j in order if sims[j] > 0]


class UserKNN(BaseRecommender):
    """Filtragem colaborativa user-based.

    Encontra os usuários com gosto mais parecido e pontua cada item pela média ponderada
    (pela similaridade) dos desvios de nota que esses vizinhos deram a ele.
    """

    name = "User-based CF"

    def __init__(self, neighbors: int = 30, shrinkage: float = 10):
        self.neighbors, self.shrinkage = neighbors, shrinkage

    def _fit(self, ds: Dataset):
        centered, self.user_means = _center_by_user(ds.matrix)
        self.centered = centered.tocsr()
        self.binary = (ds.matrix > 0).astype(np.float32).tocsr()
        self.sim = _cosine_topk(self.centered, self.binary, self.neighbors, self.shrinkage)
        self.abs_sim = abs(self.sim)

    def _num_den(self, u_idx: int):
        num = np.asarray((self.sim[u_idx] @ self.centered).todense()).ravel()
        den = np.asarray((self.abs_sim[u_idx] @ self.binary).todense()).ravel()
        return num, den

    def scores(self, u_idx: int) -> np.ndarray:
        num, den = self._num_den(u_idx)
        # Suavização (+ constante no denominador) evita que um único vizinho decida o ranking
        return num / (den + 1.0)

    def predict_ratings(self, u_idx: int) -> np.ndarray:
        num, den = self._num_den(u_idx)
        pred = np.full(self.ds.n_items, np.nan)
        ok = den > 0
        pred[ok] = self.user_means[u_idx] + num[ok] / den[ok]
        lo, hi = self.ds.matrix.data.min(), self.ds.matrix.data.max()
        return np.clip(pred, lo, hi)
