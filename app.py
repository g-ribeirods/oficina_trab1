"""Interface Streamlit do sistema de recomendação de livros. Execute: streamlit run app.py"""
import hashlib
import sys
from html import escape
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from recsys.data import (append_rating, dataset_summary, load_config,  # noqa: E402
                         load_dataset, next_user_id)
from recsys.models import ItemKNN, PopularityRecommender, UserKNN  # noqa: E402

st.set_page_config(page_title="Recomendador de Livros", page_icon="📚", layout="wide")

NEW_USER = "Novo usuário (sem histórico)"
ACCENT = "#F2A93B"
MODELS = ["Item-based CF", "User-based CF", "Popularidade"]

st.markdown("""
<style>
.block-container {padding-top: 1.6rem; max-width: 1180px;}
h1, h2, h3 {letter-spacing: -.01em;}

.hero {background: linear-gradient(120deg, #2A2116 0%, #1B1E27 70%); border: 1px solid #3A3021;
       border-radius: 16px; padding: 22px 28px; margin-bottom: 14px;}
.hero h1 {margin: 0; font-size: 1.9rem; padding: 0;}
.hero p {margin: 6px 0 0; color: #B9B3A5; font-size: .98rem;}
.pill {display:inline-block; background:#2C2618; color:#F2A93B; border:1px solid #4A3B1E;
       border-radius:999px; padding:2px 12px; font-size:.75rem; margin:10px 8px 0 0;}

.kpi {background:#1B1E27; border:1px solid #2A2E3B; border-radius:14px; padding:14px 18px;}
.kpi .v {font-size:1.7rem; font-weight:700; color:#F2A93B; line-height:1.15;}
.kpi .l {color:#9AA0AE; font-size:.8rem; margin-top:2px;}

.step {background:#1B1E27; border:1px solid #2A2E3B; border-radius:14px; padding:16px 18px; height:100%;}
.step .n {display:inline-block; width:26px; height:26px; line-height:26px; text-align:center;
          border-radius:50%; background:#F2A93B; color:#12141A; font-weight:700; margin-bottom:8px;}
.step b {display:block; margin-bottom:4px;}
.step span {color:#9AA0AE; font-size:.86rem;}

.book {display:flex; gap:14px; background:#1B1E27; border:1px solid #2A2E3B; border-radius:14px;
       padding:12px 16px; margin-bottom:10px; align-items:flex-start;}
.book.compact {padding:8px 12px; gap:10px;}
.cover {position:relative; flex:0 0 58px; height:84px; border-radius:6px; overflow:hidden;
        display:flex; align-items:center; justify-content:center; font-size:1.6rem; font-weight:700;
        color:rgba(255,255,255,.85); box-shadow:0 2px 8px rgba(0,0,0,.4);}
.compact .cover {flex-basis:40px; height:58px; font-size:1.1rem;}
.cover img {position:absolute; inset:0; width:100%; height:100%; object-fit:cover;}
.info {flex:1; min-width:0;}
.info .title {font-size:1.02rem; font-weight:600; line-height:1.3;}
.info .rank {color:#F2A93B; font-weight:700; margin-right:8px;}
.info .author {color:#B9B3A5; font-size:.85rem; margin-top:1px;}
.info .why {color:#9AA0AE; font-size:.82rem; margin-top:6px; border-left:2px solid #F2A93B;
            padding-left:8px;}
.chip {display:inline-block; background:#262A36; color:#C9CEDA; border-radius:999px;
       padding:1px 10px; font-size:.72rem; margin:6px 6px 0 0;}
.side {flex:0 0 96px; text-align:right; color:#9AA0AE; font-size:.8rem;}
.side .big {color:#ECEAE4; font-size:1.05rem; font-weight:700;}
.bar {height:6px; background:#2A2E3B; border-radius:3px; margin-top:6px; overflow:hidden;}
.bar > div {height:100%; background:linear-gradient(90deg,#F2A93B,#F7C873);}

.banner {border-radius:12px; padding:12px 16px; margin-bottom:14px; font-size:.92rem;}
.banner.cold {background:#2A2116; border:1px solid #5A4520; color:#F5D9A6;}
.banner.ok {background:#152A22; border:1px solid #24503F; color:#A8E0C6;}
.note {color:#9AA0AE; font-size:.85rem;}
</style>
""", unsafe_allow_html=True)


# ---------- Dados e modelos ----------
@st.cache_data(show_spinner="Carregando e tratando os dados...")
def get_data(version: float):
    cfg = load_config()
    ds, report = load_dataset(cfg)
    return cfg, ds, report


@st.cache_resource(show_spinner="Treinando o modelo...")
def get_model(kind: str, version: float):
    cfg, ds, _ = get_data(version)
    m = cfg["model"]
    model = {
        "Item-based CF": lambda: ItemKNN(m["neighbors"], m["shrinkage"]),
        "User-based CF": lambda: UserKNN(m["neighbors"], m["shrinkage"]),
        "Popularidade": lambda: PopularityRecommender(m["popularity_min_votes"]),
    }[kind]()
    return model.fit(ds)


def data_version() -> float:
    """Muda quando novas avaliações são gravadas -> invalida os caches."""
    cfg = load_config()
    p = ROOT / cfg["storage"]["new_ratings_path"]
    return p.stat().st_mtime if p.exists() else 0.0


# ---------- Componentes visuais ----------
def kpi(value: str, label: str) -> str:
    return f'<div class="kpi"><div class="v">{escape(str(value))}</div><div class="l">{escape(label)}</div></div>'


def cover(item_id, title: str) -> str:
    """Capa via Open Library (o ISBN é o id do item). Sem capa, sobra o gradiente com a inicial."""
    h = int(hashlib.md5(str(item_id).encode()).hexdigest()[:6], 16)
    hue = h % 360
    initial = escape((title.strip() or "?")[0].upper())
    url = f"https://covers.openlibrary.org/b/isbn/{escape(str(item_id))}-M.jpg?default=false"
    return (f'<div class="cover" style="background:linear-gradient(145deg,hsl({hue},45%,38%),'
            f'hsl({(hue + 40) % 360},50%,24%))">{initial}'
            f'<img src="{url}" alt="" loading="lazy" referrerpolicy="no-referrer"></div>')


def book_card(ds, item, rank: int | None = None, side: str = "", bar: float | None = None,
              why: str = "", compact: bool = False) -> str:
    title, author, cat, year = ds.title(item), ds.author(item), ds.extra(item), ds.year(item)
    r = f'<span class="rank">{rank}</span>' if rank else ""
    meta = " · ".join(x for x in (author, year if year not in ("", "0", "nan") else "") if x)
    chip = f'<span class="chip">{escape(cat)}</span>' if cat and cat != "nan" else ""
    w = f'<div class="why">{why}</div>' if why and not compact else ""
    b = f'<div class="bar"><div style="width:{max(0, min(bar, 1)) * 100:.0f}%"></div></div>' if bar is not None else ""
    return (f'<div class="book{" compact" if compact else ""}">{cover(item, title)}'
            f'<div class="info"><div class="title">{r}{escape(title)}</div>'
            f'<div class="author">{escape(meta)}</div>{"" if compact else chip}{w}</div>'
            f'<div class="side">{side}{b}</div></div>')


def show_recs(model, user_id, n, user_hist, compact=False):
    """Renderiza o top-N de um modelo e devolve os ids recomendados."""
    recs = model.recommend(user_id, n)
    top = max((s for _, s in recs), default=1.0) or 1.0
    cold = user_hist.empty
    for pos, (item, score) in enumerate(recs, 1):
        why = ""
        if isinstance(model, ItemKNN) and not cold:
            ex = model.explain(user_id, item, 2)
            if ex:
                why = "Porque você leu: " + "; ".join(f"<b>{escape(ds.title(j))}</b>" for j, _ in ex)
        if isinstance(model, PopularityRecommender) or cold:
            side, bar = f'<span class="big">★ {score:.2f}</span><br>média ajustada', None
        else:
            side, bar = f'<span class="big">{100 * score / top:.0f}%</span><br>afinidade', score / top
        st.markdown(book_card(ds, item, pos, side, bar, why, compact), unsafe_allow_html=True)
    return [i for i, _ in recs]


# ---------- Estado ----------
version = data_version()
cfg, ds, report = get_data(version)
relevant = cfg["dataset"]["relevance_threshold"]
counts = pd.Series(np.diff(ds.matrix.indptr), index=ds.user_ids)

# ---------- Barra lateral ----------
st.sidebar.markdown("## 📚 Recomendador")
st.sidebar.caption("Filtragem colaborativa · Book-Crossing")
st.session_state.setdefault("user_select", counts.sort_values().index[len(counts) // 2])
if st.session_state["user_select"] not in [NEW_USER, *ds.user_ids.tolist()]:
    st.session_state["user_select"] = NEW_USER


def pick_random_user():
    st.session_state["user_select"] = int(np.random.default_rng().choice(ds.user_ids))


user_choice = st.sidebar.selectbox(
    "Usuário", [NEW_USER, *ds.user_ids.tolist()], key="user_select",
    format_func=lambda u: u if u == NEW_USER else f"Usuário {u} · {counts.get(u, 0)} livros")
st.sidebar.button("🎲 Sortear usuário", on_click=pick_random_user, width="stretch")
model_kind = st.sidebar.selectbox("Modelo", MODELS)
n_recs = st.sidebar.slider("Quantidade de recomendações", 5, 20, 10)
st.sidebar.caption("**Item-based CF** é o modelo principal. **Popularidade** é o baseline e o "
                   "que atende usuários sem histórico.")

model = get_model(model_kind, version)
user_id = None if user_choice == NEW_USER else user_choice
user_hist = (ds.df[ds.df["user"] == user_id].sort_values("rating", ascending=False, kind="stable")
             if user_id is not None else ds.df.iloc[0:0])

# ---------- Cabeçalho ----------
who = "Novo usuário" if user_id is None else f"Usuário {user_id}"
st.markdown(f"""
<div class="hero">
  <h1>📚 Sistema de Recomendação de Livros</h1>
  <p>Descobre o que cada leitor deve ler a seguir a partir das notas de milhares de outros leitores,
     sem nunca repetir um livro que ele já conhece.</p>
  <span class="pill">Filtragem colaborativa</span>
  <span class="pill">Book-Crossing · Kaggle</span>
  <span class="pill">Visualizando: {escape(who)} · {escape(model_kind)}</span>
</div>""", unsafe_allow_html=True)

tab_over, tab_hist, tab_recs, tab_rate, tab_res = st.tabs(
    ["🏠 Visão geral", "📖 Histórico", "✨ Recomendações", "⭐ Avaliar livros", "📊 Resultados"])

# ---------- Visão geral ----------
with tab_over:
    s = dataset_summary(ds)
    cols = st.columns(5)
    for c, (v, l) in zip(cols, [(f"{s['usuarios']:,}", "leitores"), (f"{s['itens']:,}", "livros"),
                                (f"{s['interacoes']:,}", "avaliações"),
                                (f"{s['esparsidade_%']:.2f}%", "esparsidade da matriz"),
                                (f"{s['nota_media']:.2f}", "nota média (1 a 10)")]):
        c.markdown(kpi(v, l), unsafe_allow_html=True)

    st.markdown("### Como o sistema funciona")
    steps = [("Dados", "Book-Crossing: avaliações de 1 a 10 que leitores deram a livros."),
             ("Tratamento", "Remove notas 0 (sem avaliação real), duplicatas e leitores/livros com poucas notas."),
             ("Modelo", "Filtragem colaborativa: livros parecidos são os que os mesmos leitores avaliam de forma parecida."),
             ("Recomendação", "Pontua todos os livros, tira os já lidos e entrega o top-N com a justificativa.")]
    for c, (i, (t, d)) in zip(st.columns(4), enumerate(steps, 1)):
        c.markdown(f'<div class="step"><div class="n">{i}</div><b>{t}</b><span>{d}</span></div>',
                   unsafe_allow_html=True)

    st.markdown("### O que sobra em cada etapa do tratamento")
    labels = {"linhas_originais": "Avaliações originais",
              "apos_remover_nulos": "Sem valores nulos",
              "apos_filtrar_escala": "Só notas explícitas (1 a 10)",
              "apos_remover_duplicatas": "Sem duplicatas",
              "apos_filtro_minimo_interacoes": "Leitores ≥ %d e livros ≥ %d avaliações" % (
                  cfg["preprocessing"]["min_user_interactions"],
                  cfg["preprocessing"]["min_item_interactions"])}
    funnel = pd.Series({labels[k]: v for k, v in report.items() if k in labels})
    left, right = st.columns([3, 2])
    with left:
        st.bar_chart(funnel, horizontal=True, color=ACCENT)
    with right:
        st.markdown(
            f"- **{funnel.iloc[0]:,}** avaliações no arquivo original.\n"
            f"- **{int(funnel.iloc[0] - funnel.iloc[2]):,}** eram nota 0, que no Book-Crossing "
            "significa *leu, mas não avaliou*. Não é uma nota baixa, então foram descartadas.\n"
            f"- Restam **{int(funnel.iloc[-1]):,}** avaliações, com ao menos "
            f"{cfg['preprocessing']['min_user_interactions']} por leitor para que exista um perfil de gosto.")

    st.markdown("### Por que o cold-start é um caso à parte")
    a, b = st.columns(2)
    a.markdown('<div class="banner ok"><b>Com histórico:</b> o modelo compara os livros que o '
               'leitor avaliou com todos os demais e recomenda os mais parecidos, com nota alta '
               'entre leitores afins.</div>', unsafe_allow_html=True)
    b.markdown('<div class="banner cold"><b>Sem histórico:</b> não há gosto para comparar. O '
               'sistema recomenda os livros mais bem avaliados no geral (média ajustada pelo nº de '
               'votos) até a primeira avaliação.</div>', unsafe_allow_html=True)

# ---------- Histórico ----------
with tab_hist:
    if user_hist.empty:
        st.markdown('<div class="banner cold"><b>Este usuário ainda não tem histórico.</b> '
                    'As recomendações usarão os livros mais populares (cold-start). Avalie livros '
                    'na aba ⭐ para ver a personalização começar.</div>', unsafe_allow_html=True)
    else:
        liked = user_hist[user_hist["rating"] >= relevant]
        c = st.columns(3)
        c[0].markdown(kpi(len(user_hist), "livros avaliados"), unsafe_allow_html=True)
        c[1].markdown(kpi(f"{user_hist['rating'].mean():.2f}", "nota média dada"), unsafe_allow_html=True)
        c[2].markdown(kpi(len(liked), f"livros de que gostou (nota ≥ {relevant:g})"), unsafe_allow_html=True)
        st.write("")
        left, right = st.columns([3, 2])
        with left:
            st.subheader("Livros avaliados")
            for _, r in user_hist.head(30).iterrows():
                st.markdown(book_card(ds, r["item"], side=f'<span class="big">★ {r["rating"]:g}</span>'),
                            unsafe_allow_html=True)
            if len(user_hist) > 30:
                st.caption(f"Mostrando 30 de {len(user_hist)} livros (os mais bem avaliados).")
        with right:
            st.subheader("Como o usuário avalia")
            dist = user_hist["rating"].round().astype(int).value_counts().reindex(range(1, 11), fill_value=0)
            st.bar_chart(dist, color=ACCENT)
            cats = user_hist[user_hist["item"].map(ds.extra).ne("")]["item"].map(ds.extra).value_counts().head(6)
            if not cats.empty:
                st.subheader("Categorias mais lidas")
                st.bar_chart(cats, horizontal=True, color=ACCENT)

# ---------- Recomendações ----------
with tab_recs:
    if user_hist.empty:
        st.markdown('<div class="banner cold"><b>Usuário sem histórico (cold-start):</b> exibindo os '
                    'livros mais populares, ordenados pela média bayesiana das notas. Nenhuma '
                    'personalização é possível ainda.</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="banner ok"><b>{escape(who)}</b> avaliou {len(user_hist)} livros. '
                    'Os livros abaixo <b>não estão no histórico dele</b>.</div>', unsafe_allow_html=True)
    compare = st.toggle("Comparar com o baseline de popularidade", value=False,
                        disabled=model_kind == "Popularidade" or user_hist.empty)
    if compare and not user_hist.empty and model_kind != "Popularidade":
        a, b = st.columns(2)
        with a:
            st.subheader(model_kind)
            mine = show_recs(model, user_id, n_recs, user_hist, compact=True)
        with b:
            st.subheader("Popularidade")
            base = show_recs(get_model("Popularidade", version), user_id, n_recs, user_hist, compact=True)
        st.caption(f"Livros em comum entre as duas listas: {len(set(mine) & set(base))} de {n_recs}. "
                   "Quanto menor, mais a filtragem colaborativa personaliza.")
    else:
        st.subheader(f"Top {n_recs} para {'novo usuário' if user_id is None else who.lower()}")
        show_recs(model, user_id, n_recs, user_hist)


# ---------- Avaliar livros ----------
def save_rating(item_id, rating):
    """Callback do botão: grava a nota; usuário novo ganha um id e passa a ser o selecionado."""
    uid = user_id
    if uid is None:
        uid = next_user_id(ds)
        st.session_state["user_select"] = uid
    append_rating(cfg, uid, item_id, rating)
    st.session_state["saved_msg"] = f"Avaliação de «{ds.title(item_id)}» salva para o usuário {uid}."


with tab_rate:
    st.subheader("Avaliar um livro")
    if user_id is None:
        st.markdown('<div class="banner cold">Você está como <b>novo usuário</b>. Ao salvar a '
                    'primeira avaliação, um usuário é criado e as recomendações passam a ser '
                    'personalizadas.</div>', unsafe_allow_html=True)
    if msg := st.session_state.pop("saved_msg", None):
        st.success(msg + " Veja as novas recomendações na aba ✨.")
    query = st.text_input("Buscar por título ou autor", placeholder="ex.: harry potter, stephen king, dune...")
    found = ds.items.iloc[0:0]
    if query:
        hit = (ds.items["title"].str.contains(query, case=False, regex=False)
               | ds.items["author"].str.contains(query, case=False, regex=False))
        found = ds.items[hit & ds.items.index.isin(ds.item_ids)].head(20)
        if found.empty:
            st.warning("Nenhum livro encontrado no catálogo tratado.")
    if not found.empty:
        item_id = st.selectbox("Livro", found.index.tolist(), format_func=ds.title)
        st.markdown(book_card(ds, item_id), unsafe_allow_html=True)
        lo, hi = cfg["dataset"]["rating_scale"]
        already = user_hist[user_hist["item"] == item_id]["rating"]
        default = float(already.iloc[0]) if len(already) else float(round((lo + hi) / 2))
        rating = st.slider("Sua nota", float(lo), float(hi), default, 1.0)
        if len(already):
            st.caption(f"Você já avaliou este livro com {already.iloc[0]:g}; salvar substitui a nota.")
        st.button("Salvar avaliação", type="primary", on_click=save_rating, args=(item_id, rating))

# ---------- Resultados ----------
with tab_res:
    metrics_path = ROOT / "results" / "metrics.csv"
    if metrics_path.exists():
        res = pd.read_csv(metrics_path, index_col="Modelo")
        k = cfg["evaluation"]["k"]
        ndcg, rec_, prec = f"NDCG@{k}", f"Recall@{k}", f"Precision@{k}"
        best = res[ndcg].idxmax()
        pop = res.loc["Popularidade", ndcg] if "Popularidade" in res.index else None
        st.subheader("Avaliação offline dos modelos")
        st.caption(f"Cada leitor teve {int(cfg['evaluation']['test_fraction'] * 100)}% das avaliações "
                   f"escondidas (sorteio com semente fixa). O modelo tenta recuperar, no top-{k}, os "
                   f"livros que ele avaliou com nota ≥ {relevant:g}.")
        c = st.columns(4)
        c[0].markdown(kpi(best, "melhor modelo"), unsafe_allow_html=True)
        c[1].markdown(kpi(f"{res.loc[best, ndcg]:.3f}", f"{ndcg} do melhor"), unsafe_allow_html=True)
        c[2].markdown(kpi(f"{res.loc[best, rec_]:.1%}", f"{rec_}: livros relevantes achados"),
                      unsafe_allow_html=True)
        if pop:
            c[3].markdown(kpi(f"{res.loc[best, ndcg] / pop:.1f}×", f"{ndcg} vs. popularidade"),
                          unsafe_allow_html=True)
        st.write("")
        st.dataframe(res.style.format("{:.4f}", na_rep="—").highlight_max(axis=0, color="#4a3a15"),
                     width="stretch")
        left, right = st.columns(2)
        with left:
            st.markdown("**Qualidade do ranking** (quanto maior, melhor)")
            st.bar_chart(res[[prec, rec_, f"MAP@{k}", ndcg]].T, color=["#6C7A89", "#4F8FBA", "#F7C873", ACCENT][:len(res)])
        with right:
            st.markdown("**Erro da nota prevista** (quanto menor, melhor)")
            err = res[["RMSE", "MAE"]].dropna()
            st.bar_chart(err.T, color=["#6C7A89", "#4F8FBA", ACCENT][:len(err)])
        with st.expander("O que cada métrica mede"):
            st.markdown(
                f"- **Precision@{k}**: dos {k} livros recomendados, quantos o leitor de fato gostou.\n"
                f"- **Recall@{k}**: dos livros que ele gostou, quantos apareceram no top-{k}.\n"
                f"- **MAP@{k}** e **{ndcg}**: como {prec} e {rec_}, mas premiam acertos nas primeiras posições.\n"
                "- **RMSE / MAE**: erro médio entre a nota prevista e a nota real (menor é melhor).\n"
                "- **Cobertura**: fração do catálogo que aparece em alguma recomendação.\n"
                "- **Diversidade**: quão diferentes entre si são os livros de uma mesma lista.")
        with st.expander("Como ler estes números"):
            st.markdown(
                "Os valores absolutos são baixos porque a matriz tem **99,65% de células vazias** e o "
                "teste só reconhece como acerto o que o leitor *de fato avaliou*. O que importa é a "
                "comparação entre modelos: o item-based CF acerta várias vezes mais que o baseline de "
                "popularidade, que recomenda quase sempre os mesmos livros (cobertura próxima de zero).")
        ex_path = ROOT / "results" / "exemplos_5_usuarios.md"
        if ex_path.exists():
            with st.expander("Exemplos para 5 usuários de teste"):
                st.markdown(ex_path.read_text(encoding="utf-8"))
    else:
        st.info("Rode `python scripts/evaluate.py` para gerar as métricas.")
