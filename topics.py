# %%
import pandas
import seaborn
import numpy as np
import matplotlib.pyplot as plt
from tabulate import tabulate
import sqlite3
import scipy.stats
import scipy.sparse

# %%
seaborn.set_style("whitegrid")
seaborn.set_context("paper")

# %%
seeds = [1000, 1001, 1002, 1003, 1004]
idx = 3
l = 3

# %%
l = 0
while True:
    try:
        print(
            l,
            int(open(f"results_{seeds[idx]}/{l}/Bw").read()),
            int(open(f"results_{seeds[idx]}/{l}/Bd").read()),
        )
        l += 1
    except:
        break

# %%
ls = [3, 2]

# %%
docs = pandas.read_sql_table("document", "sqlite:///data.db", index_col="id")
words = pandas.read_sql_table("vocabulary", "sqlite:///data.db", index_col="id")
# %%
for l in ls:
    groups = np.load(f"results_{seeds[idx]}/{l}/Ps.npz")["p_td_d"]
    fig, ax1 = plt.subplots()
    ax1.plot(np.sort(groups.sum(axis=1)), ls="", marker="D")
    ax2 = ax1.twinx()
    ax2.plot(np.sort(groups.sum(axis=1)).cumsum())
    fig.tight_layout()
    plt.show()

# %%
l = 3
dt = (
    docs[["year"]]
    .join(
        pandas.DataFrame(
            np.load(f"results_{seeds[idx]}/{l}/Ps.npz")["p_td_d"].T, dtype=int
        )
    )
    .groupby("year")
    .sum()
)
dt = dt.div(dt.sum(axis=1), axis=0)
dt.plot()
dt.rolling(window=10).mean().dropna().plot()

# %%
twd = (
    docs[["year"]]
    .join(pandas.DataFrame(np.load(f"results_{seeds[idx]}/{l}/Ps.npz")["p_tw_d"].T))
    .groupby("year")
    .sum()
)
twd = twd.div(twd.sum(axis=1), axis=0)
twd.plot()
twd.rolling(window=10).mean().dropna().plot()


# %%
pwtw = np.load(f"results_{seeds[idx]}/{l}/Ps.npz")["p_w_tw"]
topicTopWords = np.argsort(pwtw, axis=0)[-20:, :].T
topicTopDocs = np.argsort(
    np.load(f"results_{seeds[idx]}/{l}/Ps.npz")["p_tw_d"], axis=1
)[:, -5:]
with open("topics.txt", "w") as f:
    for i, (w, d) in enumerate(zip(topicTopWords, topicTopDocs)):
        w = topicTopWords[i, :]
        d = topicTopDocs[i, :]
        print("Topic: ", i, file=f)
        print(tabulate(reversed(list(zip(words.loc[w, :].word, pwtw.T[i, w])))), file=f)
        print(
            tabulate(
                reversed(
                    list(
                        docs.loc[d, :][
                            ["title", "authors", "journal", "year"]
                        ].itertuples(index=False, name=None)
                    )
                ),
                maxcolwidths=40,
            ),
            file=f,
        )
        print("\n\n", file=f)

# %%
pandas.DataFrame(
    np.load(f"results_{seeds[idx]}/{l}/Ps.npz")["p_td_d"].argmax(axis=0),
    dtype=int,
    columns=["DocGroup"],
).to_sql(
    "Groups",
    "sqlite:///model.db",
    index=True,
    index_label="document_id",
    chunksize=10000,
)
pandas.DataFrame(
    np.load(f"results_{seeds[idx]}/{l}/Ps.npz")["p_w_tw"].argmax(axis=1),
    dtype=int,
    columns=["Topic"],
).to_sql(
    "Topics",
    "sqlite:///model.db",
    index=True,
    index_label="word_id",
    chunksize=10000,
)
# %%
with sqlite3.connect("model.db") as db:
    cur = db.cursor()
    cur.executescript(
        f"""
            ATTACH DATABASE "data.db" AS data;

            CREATE TABLE w_td AS SELECT word_id, docgroup, SUM(count) AS count FROM (Groups JOIN DATA.graph AS graph USING (document_id)) GROUP BY docgroup, word_id;
            """
    )
# %%
with sqlite3.connect("model.db") as db:
    cur = db.cursor()
    cur.executescript(
        f"""
            ATTACH DATABASE "data.db" AS data;

            CREATE TABLE tw_td AS SELECT topic, docgroup, SUM(count) AS count FROM ((Groups JOIN DATA.graph AS graph USING (document_id)) JOIN Topics USING (word_id)) GROUP BY docgroup, topic;
            """
    )
# %%
pwtd = pandas.read_sql_table("w_td", "sqlite:///model.db")
pw = pandas.DataFrame(
    np.load(f"results_{seeds[idx]}/{l}/Ps.npz")["p_w"], columns=["pw"]
)

# %%
btest = pwtd.merge(
    pw.reset_index(), left_on="word_id", right_on="index", how="left"
).merge(pwtd.groupby("DocGroup").sum(), on="DocGroup", how="left")[
    ["word_id_x", "DocGroup", "count_x", "count_y", "pw"]
]
btest.columns = ["word_id", "DocGroup", "word_count", "group_count", "pw"]
# %%
btest["pvalue"] = np.vectorize(
    lambda k, n, p: scipy.stats.binomtest(k, n, p, "greater").pvalue
)(btest["word_count"], btest["group_count"], btest["pw"])

# %%
btest[["word_id", "DocGroup", "word_count", "pvalue"]].to_sql(
    "pvalue", "sqlite:///model.db", index=False, chunksize=10000
)
# %%
btest = pandas.read_sql_table("pvalue", "sqlite:///model.db")

# %%
with open("topics_by_doc.txt", "w") as f:
    for i in range(btest.DocGroup.max() + 1):
        print("Topic: ", i, file=f)
        print(
            tabulate(
                (
                    btest.loc[
                        (btest.DocGroup == i).squeeze(),
                    ]
                )
                .sort_values(["pvalue", "word_count"], ascending=[True, False])
                .merge(words, on="word_id")
                .head(20)[["word", "word_count", "pvalue"]]
            ),
            file=f,
        )
        print("\n\n", file=f)

# %%
twtd = pandas.read_sql("tw_td", "sqlite:///model.db")
twtd = scipy.sparse.coo_array((twtd["count"], (twtd.Topic, twtd.DocGroup))).toarray()
# %%
Group_by_Topic = (
    (twtd / twtd.sum(axis=0)) - (twtd.sum(axis=1) / twtd.sum())[:, None]
).T
seaborn.heatmap(Group_by_Topic)

# %%
TopicLabels = pandas.read_csv("topics.csv")


# %%
with open("topics_by_group.txt", "w") as f:
    for i in range(Group_by_Topic.shape[0]):
        print("Topic: ", i, file=f)
        print(
            tabulate(
                pandas.DataFrame(
                    {
                        "Topic": np.argsort(-Group_by_Topic[i, :]),
                        "P_over_exp": np.flip(np.sort(Group_by_Topic[i, :])),
                    }
                )
                .query("P_over_exp > 0")
                .merge(TopicLabels, left_on="Topic", right_on="Number")[
                    ["Topic_y", "P_over_exp"]
                ],
                showindex=False,
                headers=["Topic", "Probability over Expected"]
            ),
            file=f,
        )
        print("\n\n", file=f)

# %%
