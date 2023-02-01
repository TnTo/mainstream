# %%
import pickle
import graph_tool.all as gt
import pandas
from multiprocessing import Pool
import sklearn.metrics
import scipy.sparse
import numpy
import numpy.random
import math
import os
import seaborn


# %%
df[(df.kind == 1)].groupby("seed").nunique()[["group"]]

# %%
doc_group = [
    df[(df.kind == 0) & (df.seed == 1000 + i)].group.to_numpy() for i in range(3)
] + [
    numpy.random.randint(
        math.ceil(df[(df.kind == 0)].groupby("seed").nunique()[["group"]].mean()),
        size=df[df.kind == 0].max().id + 1,
    )
]
for i in range(3 + 1):
    for j in range(3 + 1):
        print(
            i,
            j,
            sklearn.metrics.normalized_mutual_info_score(doc_group[i], doc_group[j]),
        )

# %%
word_group = [
    df[(df.kind == 1) & (df.seed == 1000 + i)].group.to_numpy() for i in range(3)
] + [
    numpy.random.randint(
        math.ceil(df[(df.kind == 1)].groupby("seed").nunique()[["group"]].mean()),
        size=df[df.kind == 1].max().id + 1,
    )
]
for i in range(3 + 1):
    for j in range(3 + 1):
        print(
            i,
            j,
            sklearn.metrics.normalized_mutual_info_score(word_group[i], word_group[j]),
        )
# %%
df[(df.seed == 1002)].groupby("kind").nunique()[["group"]]

#%%
m = scipy.sparse.load_npz("sparse.npz")
words = pandas.read_sql("vocabulary", "sqlite:///data.db").sort_values("id")
docs = pandas.read_sql("document", "sqlite:///data.db").sort_values("id")
try:
    os.mkdir("out")
except:
    pass
# %%
for s in seeds:
    try:
        os.mkdir(f"out/{s}")
    except:
        pass
    topics = (
        df[(df.seed == s) & (df.kind == 1)].groupby("group").id.agg(list).reset_index()
    )
    topics.index = topics.index.rename("topic")

    groups = (
        df[(df.seed == s) & (df.kind == 0)].groupby("group").id.agg(list).reset_index()
    )
    # Word-Topic
    word_topic = (
        df[(df.seed == s) & (df.kind == 1)]
        .merge(words, on="id")
        .merge(topics["group"].reset_index(), on="group")
        .sort_values("id")
    )
    word_topic["freq"] = m.sum(axis=0).T.astype(int)
    word_topic = word_topic.sort_values("freq", ascending=False)

    T = word_topic["topic"].nunique()
    G = groups.group.nunique()

    for i in range(T):
        word_topic.query("topic == @i")[["word", "freq"]].to_csv(
            f"out/{s}/topic_{i}.csv", index=False
        )
    # Word-Group
    for i in range(G):
        words.assign(
            freq=(m[groups.iloc[i].id, :].sum(axis=0) / m[groups.iloc[i].id, :].sum()).T
        ).sort_values("freq", ascending=False)[["word", "freq"]].to_csv(
            f"out/{s}/group_words_{i}.csv", index=False
        )
    # Topic-Group
    tg = numpy.zeros((G, T))
    for t in range(word_topic["topic"].nunique()):
        for g in range(groups.group.nunique()):
            tg[g, t] = m[groups.iloc[g].id, :][:, topics.iloc[t].id].sum()
    tg = tg / m.sum()
    pandas.DataFrame(tg).to_csv(f"out/{s}/group_topic.csv")

    # %%
