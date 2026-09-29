"""Avalia os modelos e gera results/metrics.csv e results/exemplos_5_usuarios.md."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from recsys.data import load_config, load_dataset, train_test_split  # noqa: E402
from recsys.evaluation import evaluate_models, users_examples  # noqa: E402
from recsys.models import ItemKNN, PopularityRecommender, RandomRecommender, UserKNN  # noqa: E402


def main():
    cfg = load_config()
    ds, report = load_dataset(cfg, include_new=False)
    train, test = train_test_split(ds, cfg)
    print(f"Treino: {train.matrix.nnz} interações | Teste: {len(test)} interações")

    m = cfg["model"]
    models = [
        RandomRecommender(),
        PopularityRecommender(m["popularity_min_votes"]),
        UserKNN(m["neighbors"], m["shrinkage"]),
        ItemKNN(m["neighbors"], m["shrinkage"]),
    ]
    results = evaluate_models(models, train, test, cfg)
    out = ROOT / "results"
    out.mkdir(exist_ok=True)
    results.round(4).to_csv(out / "metrics.csv")
    print(results.round(4).to_string())

    item_model = models[-1]  # já treinado no treino
    (out / "exemplos_5_usuarios.md").write_text(
        users_examples(item_model, train, test, cfg), encoding="utf-8")
    print(f"Arquivos gerados em {out}")


if __name__ == "__main__":
    main()
