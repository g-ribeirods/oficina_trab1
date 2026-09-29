"""Métricas de avaliação. `recommended` é uma lista ordenada de ids; `relevant` um conjunto."""
import numpy as np
from sklearn.preprocessing import normalize


def precision_at_k(recommended, relevant, k: int) -> float:
    """Fração dos K primeiros recomendados que são relevantes."""
    return sum(i in relevant for i in recommended[:k]) / k


def recall_at_k(recommended, relevant, k: int) -> float:
    """Fração dos itens relevantes que aparecem nos K primeiros."""
    if not relevant:
        return 0.0
    return sum(i in relevant for i in recommended[:k]) / len(relevant)


def average_precision_at_k(recommended, relevant, k: int) -> float:
    """AP@K: média da precisão em cada posição em que há um acerto."""
    if not relevant:
        return 0.0
    hits, total = 0, 0.0
    for pos, i in enumerate(recommended[:k], start=1):
        if i in relevant:
            hits += 1
            total += hits / pos
    return total / min(len(relevant), k)


def ndcg_at_k(recommended, relevant, k: int) -> float:
    """NDCG@K com relevância binária: acertos no topo valem mais (desconto log2)."""
    if not relevant:
        return 0.0
    dcg = sum(1 / np.log2(pos + 1) for pos, i in enumerate(recommended[:k], start=1) if i in relevant)
    ideal = sum(1 / np.log2(pos + 1) for pos in range(1, min(len(relevant), k) + 1))
    return dcg / ideal


def rmse(y_true, y_pred) -> float:
    d = np.asarray(y_true, dtype=float) - np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean(d ** 2)))


def mae(y_true, y_pred) -> float:
    d = np.asarray(y_true, dtype=float) - np.asarray(y_pred, dtype=float)
    return float(np.mean(np.abs(d)))


def catalog_coverage(all_recommendations, n_items: int) -> float:
    """Fração do catálogo que aparece em pelo menos uma lista de recomendação."""
    return len({i for recs in all_recommendations for i in recs}) / n_items


def intra_list_diversity(item_indices, item_vectors) -> float:
    """1 - similaridade cosseno média entre os itens da lista (maior = mais diversa).

    item_vectors: matriz esparsa/densa (itens x usuários) com o perfil de cada item.
    """
    if len(item_indices) < 2:
        return 0.0
    v = normalize(item_vectors[list(item_indices)])
    sim = np.asarray((v @ v.T).todense() if hasattr(v, "todense") else v @ v.T)
    n = len(item_indices)
    return float(1 - (sim.sum() - np.trace(sim)) / (n * (n - 1)))
