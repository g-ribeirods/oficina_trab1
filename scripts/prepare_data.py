"""Baixa o Book-Crossing (Kaggle: ruchi798/bookcrossing-dataset) e gera dois CSVs enxutos
em data/bookcrossing/: ratings.csv (interações) e books.csv (catálogo com autor, ano e
categoria). O download é anônimo, sem conta nem token do Kaggle."""
import html
import io
import ssl
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd

URL = "https://www.kaggle.com/api/v1/datasets/download/ruchi798/bookcrossing-dataset"
DEST = Path(__file__).resolve().parents[1] / "data" / "bookcrossing"
RATINGS = "Book reviews/Book reviews/BX-Book-Ratings.csv"
BOOKS = "Book reviews/Book reviews/BX_Books.csv"
CATEGORIES = "Books Data with Category Language and Summary/Preprocessed_data.csv"


def download() -> zipfile.ZipFile:
    print(f"Baixando {URL} (~80 MB) ...")
    try:
        import certifi
        ctx = ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        ctx = ssl.create_default_context()
    with urllib.request.urlopen(URL, timeout=300, context=ctx) as resp:
        return zipfile.ZipFile(io.BytesIO(resp.read()))


def clean_category(raw: str) -> str:
    """"['Fiction']" -> "Fiction"; "9" (desconhecida) -> ""."""
    c = str(raw).strip("[]'\" ")
    return "" if c in ("", "9", "nan") else c.title() if c.isupper() else c


def main():
    if (DEST / "ratings.csv").exists() and (DEST / "books.csv").exists():
        print(f"Dados já preparados em {DEST}")
        return
    DEST.mkdir(parents=True, exist_ok=True)
    zf = download()

    with zf.open(RATINGS) as f:
        ratings = pd.read_csv(f, sep=";", encoding="latin-1", dtype=str)
    ratings.columns = ["user", "isbn", "rating"]

    with zf.open(BOOKS) as f:
        books = pd.read_csv(f, sep=";", encoding="latin-1", dtype=str, on_bad_lines="skip")
    books = books.rename(columns={"ISBN": "isbn", "Book-Title": "title", "Book-Author": "author",
                                  "Year-Of-Publication": "year", "Publisher": "publisher"})
    for col in ("title", "author", "publisher"):
        books[col] = books[col].fillna("").map(html.unescape)
    books = books.drop_duplicates("isbn")[["isbn", "title", "author", "year", "publisher"]]

    # Categoria vem de outro arquivo do dataset (grande: lido em blocos, só 2 colunas)
    cats = {}
    with zf.open(CATEGORIES) as f:
        for chunk in pd.read_csv(f, usecols=["isbn", "Category"], dtype=str, chunksize=200_000):
            for isbn, c in zip(chunk["isbn"], chunk["Category"]):
                cats.setdefault(isbn, c)
    books["category"] = books["isbn"].map(cats).map(clean_category)

    # Só interações de livros que existem no catálogo
    ratings = ratings[ratings["isbn"].isin(books["isbn"])]
    ratings.to_csv(DEST / "ratings.csv", index=False)
    books.to_csv(DEST / "books.csv", index=False)
    print(f"Pronto: {len(ratings):,} avaliações e {len(books):,} livros em {DEST}")


if __name__ == "__main__":
    main()
