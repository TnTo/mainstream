# %%
import pickle
import graph_tool.all as gt
import pandas
from multiprocessing import Pool
import sklearn.metrics
import scipy.sparse
import numpy.random
import math
import os

# %%
numpy.random.seed(8686)

# %%
seeds = [1000, 1001, 1002]


def load(s):
    return pickle.load(open(f"state_{s}.pkl", "rb"))


with Pool(4) as p:
    states = p.map(load, seeds)
states

# %%
S = [state.entropy() for state in states]
S

# %%
levels = [
    states[0].levels[3],
    states[1].levels[3],
    states[2].levels[3],
]
levels

# %%
lS = [l.entropy() for l in levels]
lS

# %%
pickle.dump(levels, open("levels.pkl", "wb"), -1)

# %%
g = gt.load_graph("graph.gt.gz")

# %%
def get_groups(state, seed):
    level = state.levels[3]
    b = gt.contiguous_map(level.b)
    label_map = {}
    for v in g.vertices():
        label_map[level.b[v]] = b[v]
    return pandas.DataFrame(
        {
            "id": g.vp["id"].a,
            "kind": g.vp["kind"].a,
            "group": [label_map[b] for b in state.project_level(3).get_blocks()],
            "seed": seed,
        }
    )


df = pandas.concat([get_groups(state, 1000 + i) for i, state in enumerate(states)])


# %%
df.to_csv("groups.csv", index=False)

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

    for i in range(word_topic["topic"].nunique()):
        word_topic.query("topic == @i")[["word", "freq"]].to_csv(
            f"out/{s}/topic_{i}.csv", index=False
        )
    # Word-Group
    for i in range(groups.group.nunique()):
        words.assign(
            freq=(m[groups.iloc[i].id, :].sum(axis=0) / m[groups.iloc[i].id, :].sum()).T
        ).sort_values("freq", ascending=False)[["word", "freq"]].to_csv(
            f"out/{s}/group_words_{i}.csv", index=False
        )
    # %%
