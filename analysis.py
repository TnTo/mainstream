# %%
import modin.pandas as pandas
from tqdm.auto import tqdm
import matplotlib.pyplot as plt

# %%
# MASS TO BE CONSERVED IN CLEANING

# %%
d = pandas.read_sql("SELECT * FROM doc_len", "sqlite:///clean.db")
w = pandas.read_sql("SELECT * FROM word_count", "sqlite:///clean.db")
dmeta = pandas.read_sql("document", "sqlite:///clean.db")

# %%
dq = d[["len"]].quantile(q=[x / 20 for x in range(0, 21, 1)])
wq = w[["freq"]].quantile(q=[x / 20 for x in range(0, 21, 1)])
s = w.freq.sum()
l = len(w)
for i in list(range(1, 11)) + list(range(10, 201, 10)):
    w0 = w[(w.word.str.len() > j) & (w.freq >= i)]
    print(
        f"min_freq: {i}; min_len: {j+1}; preserved vocab: {len(w0) / l}; preserved mass: {w0.freq.sum() / s}"
    )

# %%
# SECOND ATTEMPT
g = pandas.read_sql_query(
    "SELECT word_id, COUNT(DISTINCT document_id) AS n_docs, SUM(count) AS freq FROM graph GROUP BY word_id",
    f"sqlite:///clean.db",
)
M = g.freq.sum()
W = len(g.word_id.unique())
D = len(dmeta)

print("Min_doc", "Preserved Mass", "Preserved Voc")
for i in range(0, 51, 5):
    tmp = g[g.n_docs >= i]
    print(
        i, "{:.4f}".format((tmp.freq.sum()) / M), "{0:.3f}".format(len(tmp.word_id) / W)
    )

print("Max_doc", "Preserved Mass", "Preserved Voc")
for i in range(0, 101, 5):
    tmp = g[g.n_docs < D * (1 - (i / 100))]
    print(
        f"{100-i}% of docs",
        "{:.0f}".format(D * (1 - (i / 100))),
        "{:.3f}".format((tmp.freq.sum()) / M),
        "{0:.5f}".format(len(tmp.word_id) / W),
    )

# %%
# STOPWORDS

w0 = pandas.read_sql("SELECT * FROM word_count", f"sqlite:///clean.db")

g = pandas.read_sql_query(
    "SELECT word_id, COUNT(DISTINCT document_id) AS n_docs FROM graph GROUP BY word_id",
    f"sqlite:///clean.db",
)

w = w0.merge(
    g[(9000 > g.n_docs) & (g.n_docs >= 35)].word_id,
    how="inner",
    on="word_id",
)

print("Preserved Voc:", len(w) / len(w0))
print("Preserved Mass:", w.freq.sum() / w0.freq.sum())

import mainstream

sw = pandas.read_csv("old/latex_sw.csv")
sw_st = sw.word.apply(mainstream.clean_and_stem)
print("# SW", sw_st.isin(w.word).sum())
print(sw_st[sw_st.isin(w.word)])

sw_st2 = pandas.read_csv("latex.csv", header=False)
print("# SW", sw_st2[0].isin(w.word).sum())
print(sw_st2[sw_st2[0].isin(w.word)])

# %%
# DOC LEN

w = pandas.read_sql("SELECT * FROM word_count", f"sqlite:///clean.db")

min_doc_occurencies = 35  # 1/1.000 of docs
max_doc_occurencies = 9000  # ~25% of docs
sw = "latex.csv"

g = pandas.read_sql_query(
    "SELECT word_id, COUNT(DISTINCT document_id) AS n_docs FROM graph GROUP BY word_id",
    f"sqlite:///clean.db",
)
if not min_doc_occurencies:
    min_doc_occurencies = 0
if not max_doc_occurencies:
    max_doc_occurencies = 0
w = w.merge(
    g[(g.n_docs < max_doc_occurencies) & (g.n_docs >= min_doc_occurencies)].word_id,
    how="inner",
    on="word_id",
)
w = w[~(w.word.isin(pandas.read_csv(sw, header=None)[0]))]

g = pandas.read_sql("graph", f"sqlite:///clean.db")
g = g[g.word_id.isin(w.word_id)]
d = g.groupby("document_id")["count"].sum()
dmeta = pandas.read_sql("document", "sqlite:///clean.db")

docs = dmeta.join(d)[
    ["authors", "title", "journal", "year", "lenght", "count", "pagestart", "pageend"]
]
print(docs.sort_values("count", ascending=False).head(10))
print(docs.sort_values("count", ascending=True).head(10))

for i in range(0, 501, 10):
    print(i, "{0:.3f}".format((d < i).sum() / len(d)))

print(docs[docs.lenght >= 1500].sort_values("count"))
print(len(docs[docs.lenght >= 1500]) / len(docs))

dp = dmeta[dmeta.lenght < 1500]
dp.year.hist()
dp.groupby("journal").count().id.plot.bar()
