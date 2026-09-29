"""Avaliação offline: hold-out temporal por usuário, métricas de ranking e de predição."""
import numpy as np
import pandas as pd

from . import metrics as M
from .data import Dataset, build_dataset


def evaluate_models(models: list, train: Dataset, test: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Treina cada modelo no treino e mede o desempenho nas interações de teste."""
    k = cfg["evaluation"]["k"]
    thr = cfg["dataset"]["relevance_threshold"]
    item_vectors = train.matrix.T.tocsr()
    relevant = (test[test["rating"] >= thr].groupby("user")["item"].apply(set)).to_dict()
    users = list(relevant)  # só usuários com ao menos um item relevante no teste
    test_by_user = {u: g for u, g in test.groupby("user")}

    rows = []
    for model in models:
        model.fit(train)
        per_user = {"P": [], "R": [], "AP": [], "NDCG": [], "DIV": []}
        all_recs = []
        for u in users:
            recs = [i for i, _ in model.recommend(u, k)]
            all_recs.append(recs)
            rel = relevant[u]
            per_user["P"].append(M.precision_at_k(recs, rel, k))
            per_user["R"].append(M.recall_at_k(recs, rel, k))
            per_user["AP"].append(M.average_precision_at_k(recs, rel, k))
            per_user["NDCG"].append(M.ndcg_at_k(recs, rel, k))
            per_user["DIV"].append(M.intra_list_diversity(
                [train.item_index(i) for i in recs], item_vectors))
        row = {
            "Modelo": model.name,
            f"Precision@{k}": np.mean(per_user["P"]),
            f"Recall@{k}": np.mean(per_user["R"]),
            f"MAP@{k}": np.mean(per_user["AP"]),
            f"NDCG@{k}": np.mean(per_user["NDCG"]),
            "Cobertura": M.catalog_coverage(all_recs, train.n_items),
            "Diversidade": np.mean(per_user["DIV"]),
        }
        row["RMSE"], row["MAE"] = _rating_errors(model, train, test_by_user)
        rows.append(row)
    return pd.DataFrame(rows).set_index("Modelo")


def _rating_errors(model, train: Dataset, test_by_user: dict) -> tuple[float, float]:
    """RMSE/MAE da predição de nota nos pares de teste (NaN se o modelo não prediz nota).
    Onde não há vizinhos suficientes, usa-se a média do usuário."""
    if not hasattr(model, "predict_ratings"):
        return np.nan, np.nan
    counts = np.diff(train.matrix.indptr)
    means = np.divide(np.asarray(train.matrix.sum(axis=1)).ravel(), counts,
                      out=np.full(train.n_users, train.matrix.data.mean()), where=counts > 0)
    y_true, y_pred = [], []
    for u, g in test_by_user.items():
        ui = train.user_index(u)
        pred = model.predict_ratings(ui)
        ii = np.array([train.item_index(i) for i in g["item"]])
        p = pred[ii]
        y_true.extend(g["rating"].tolist())
        y_pred.extend(np.where(np.isnan(p), means[ui], p).tolist())
    return M.rmse(y_true, y_pred), M.mae(y_true, y_pred)


def users_examples(model, train: Dataset, test: pd.DataFrame, cfg: dict,
                   n_users: int = 5, seed: int = 7) -> str:
    """Relatório em Markdown com histórico, recomendações e acertos para n_users usuários
    de teste, mais o caso de usuário sem histórico."""
    k, thr = cfg["evaluation"]["k"], cfg["dataset"]["relevance_threshold"]
    rng = np.random.default_rng(seed)
    candidates = sorted(test[test["rating"] >= thr]["user"].unique())
    chosen = rng.choice(candidates, size=n_users, replace=False)
    liked_in_test = test[test["rating"] >= thr].groupby("user")["item"].apply(set)

    out = [f"# Exemplos de recomendação ({model.name}, top-{k})\n"]
    for u in chosen:
        hist = train.df[train.df["user"] == u].sort_values("rating", ascending=False)
        recs = model.recommend(u, k)
        liked = liked_in_test.get(u, set())
        out.append(f"## Usuário {u}\n")
        out.append(f"Histórico de treino: {len(hist)} itens. Mais bem avaliados:\n")
        for _, r in hist.head(5).iterrows():
            out.append(f"- {train.title(r['item'])} — nota {r['rating']:g}")
        out.append("\nRecomendações (nenhuma delas está no histórico):\n")
        for pos, (i, score) in enumerate(recs, 1):
            why = ""
            if hasattr(model, "explain"):
                ex = model.explain(u, i, 2)
                if ex:
                    why = " — por semelhança com: " + "; ".join(train.title(j) for j, _ in ex)
            hit = " ✅ **acerto no teste**" if i in liked else ""
            out.append(f"{pos}. {train.title(i)} (score {score:.2f}){why}{hit}")
        n_hit = sum(i in liked for i, _ in recs)
        out.append(f"\nAcertos: {n_hit} de {len(liked)} itens relevantes do teste.\n")

    out.append("## Usuário sem histórico (cold-start)\n")
    out.append("Sem interações, não há como calcular similaridade; o sistema recorre à popularidade "
               "(média bayesiana):\n")
    for pos, (i, score) in enumerate(model.recommend("__novo_usuario__", k), 1):
        out.append(f"{pos}. {train.title(i)} (score {score:.2f})")
    return "\n".join(out) + "\n"
