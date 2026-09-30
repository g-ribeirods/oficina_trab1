import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from recsys import metrics as m  # noqa: E402
from recsys.data import Dataset, build_dataset  # noqa: E402
from recsys.models import ItemKNN, PopularityRecommender, UserKNN  # noqa: E402
import pandas as pd  # noqa: E402


def test_precision_recall():
    rec, rel = [1, 2, 3, 4], {2, 4, 9}
    assert m.precision_at_k(rec, rel, 4) == 0.5
    assert m.recall_at_k(rec, rel, 4) == pytest.approx(2 / 3)


def test_ndcg_perfect_and_zero():
    assert m.ndcg_at_k([1, 2], {1, 2}, 2) == pytest.approx(1.0)
    assert m.ndcg_at_k([3, 4], {1, 2}, 2) == 0.0
    assert m.ndcg_at_k([2, 1], {1}, 2) < m.ndcg_at_k([1, 2], {1}, 2)


def test_average_precision():
    assert m.average_precision_at_k([1, 9, 2], {1, 2}, 3) == pytest.approx((1 + 2 / 3) / 2)


def test_rmse_mae():
    assert m.rmse([1, 3], [2, 5]) == pytest.approx(np.sqrt(2.5))
    assert m.mae([1, 3], [2, 5]) == pytest.approx(1.5)


def test_coverage():
    assert m.catalog_coverage([[1, 2], [2, 3]], 10) == 0.3


@pytest.fixture
def tiny() -> Dataset:
    rows = [(u, i, r, 0) for u, i, r in [
        (1, "a", 5), (1, "b", 4), (1, "c", 1),
        (2, "a", 5), (2, "b", 5), (2, "d", 4),
        (3, "a", 1), (3, "c", 5), (3, "e", 4),
        (4, "c", 5), (4, "e", 5), (4, "d", 2),
    ]]
    df = pd.DataFrame(rows, columns=["user", "item", "rating", "timestamp"])
    return build_dataset(df, pd.DataFrame(columns=["title", "extra"]))


@pytest.mark.parametrize("model", [PopularityRecommender(1), ItemKNN(5, 0), UserKNN(5, 0)])
def test_recommendations_exclude_seen_items(tiny, model):
    model.fit(tiny)
    seen = set(tiny.df[tiny.df.user == 1].item)
    recs = [i for i, _ in model.recommend(1, 10)]
    assert recs and not seen & set(recs)


def test_cold_start_returns_popular_items(tiny):
    model = ItemKNN(5, 0).fit(tiny)
    assert len(model.recommend("usuario_novo", 3)) == 3


def test_implicit_feedback_is_not_degenerate():
    """Sem notas (tudo 1.0) o CF deve continuar diferenciando itens, não virar popularidade."""
    inter = [(u, i) for u, items in {1: "abc", 2: "abc", 3: "abd", 4: "de", 5: "de", 6: "ae"}.items()
             for i in items]
    df = pd.DataFrame(inter, columns=["user", "item"]).assign(rating=1.0, timestamp=0)
    ds = build_dataset(df, pd.DataFrame(columns=["title", "extra"]))
    model = ItemKNN(5, 0).fit(ds)
    assert len(set(model.scores(ds.user_index(1)).round(6))) > 1
