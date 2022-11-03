# %%
import pandas as pandas
from tqdm.auto import tqdm
import seaborn
import matplotlib
import matplotlib.pyplot as plt

# %%
# JOURNALS PROPORTIONS

# %%
df = pandas.read_sql_table(
    "document", "sqlite:///mainstream.db", index_col="id", parse_dates=["date"]
)

# %%
journals = [
    "Econometrica",
    "The American Economic Review",
    "The Review of Economic Studies",
    "The Quarterly Journal of Economics",
    "The Review of Economics and Statistics",
    "Journal of Political Economy",
    "The Economic Journal",
    "Economica",
]

# %%
df2 = df[
    (df.authors != "null")
    & (df.provider == "jstor")
    & (df.subtype == "research-article")
    & df.unigrams
    & df.journal.isin(journals)
    & (df.year >= 1946)
    & (df.year <= 2016)
]
df2

# %%
ax = seaborn.histplot(
    x="year",
    data=df2[["year", "journal"]],
    hue="journal",
    hue_order=journals,
    multiple="stack",
    bins=range(1947, 2018),
)
seaborn.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))

# %%
journal_year = (
    df2.groupby(["year", "journal"])
    .count()
    .type.reset_index()
    .pivot(columns="journal", values="type", index="year")
)

journal_year = journal_year.div(journal_year.sum(axis=1), axis=0)[
    list(reversed(journals))
]

ax = journal_year.plot(
    kind="bar",
    stacked=True,
    rot=0,
    legend="reverse",
    align="center",
    width=1,
    color=list(reversed(seaborn.color_palette(n_colors=len(journals)))),
    edgecolor="black",
    linewidth=0.8,
)
ax.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(10))
ax.legend(reversed(ax.legend().legendHandles), journals, bbox_to_anchor=(1.0, 1.0))

# %%
# MASS TO BE CONSERVED IN CLEANING

# %%
d = pandas.read_sql("SELECT * FROM doc_len;", "sqlite:///clean.db")
w = pandas.read_sql("SELECT * FROM word_count;", "sqlite:///clean.db")
dmeta = pandas.read_sql("document", "sqlite:///clean.db")

# %%
dq = d[["len"]].quantile(q=[x / 20 for x in range(0, 21, 1)])
wq = w[["freq"]].quantile(q=[x / 20 for x in range(0, 21, 1)])
s = w.freq.sum()
l = len(w)
for i in range(1, 11):
    for j in range(1, 3):
        w0 = w[(w.word.str.len() > j) & (w.freq >= i)]
        print(
            f"min_freq: {i}; min_len: {j+1}; preserved vocab: {len(w0) / l}; preserved mass: {w0.freq.sum() / s}"
        )

# %%
# IMPACT FACTOR

# %%
def get_data(y):
    df = pandas.read_csv(
        f"https://www.scimagojr.com/journalrank.php?category=2002&area=2000&type=j&year={y}&out=xls",
        sep=";",
        decimal=",",
    )
    df["year"] = y
    return df


df = pandas.concat([get_data(y) for y in range(1999, 2022)])
# %%
data = df[["Title", "year", "SJR", "H index", "Cites / Doc. (2years)"]]
data.columns = ["title", "y", "SJR", "H", "IF2"]

data["general"] = ~(
    data.title.str.contains("Financ")
    | data.title.str.contains("Marketing")
    | data.title.str.contains("Account")
    | data.title.str.contains("Business")
    | data.title.str.contains("Entrepreneur")
    | data.title.str.contains("Consumer")
)

data = data[data.general]
data = data.drop(columns="general")

# %%
min_data = data.groupby("title").min()
mean_data = data.groupby("title").mean()


def process_data(d, i, l):
    df = d[i].sort_values(ascending=False).reset_index().reset_index()
    df = df.rename(columns={"index": f"{i}_{l}_rank", f"{i}": f"{i}_{l}"})
    df[f"{i}_{l}_rank"] += 1
    return df[["title", f"{i}_{l}", f"{i}_{l}_rank"]]


datas = [process_data(min_data, i, "min") for i in ["SJR", "H", "IF2"]] + [
    process_data(mean_data, i, "avg") for i in ["SJR", "H", "IF2"]
]
# %%
res = datas[0]
for n in range(1, len(datas)):
    res = res.merge(datas[n], on="title")
# %%
res_rank = res[
    ["title"] + [f"{i}_{m}_rank" for i in ["SJR", "H", "IF2"] for m in ["min", "avg"]]
]
res_rank = res_rank.merge(
    res_rank.set_index("title").min(axis=1).rename("min_rank").reset_index(), on="title"
)
res_rank = res_rank.merge(
    res_rank.set_index("title").max(axis=1).rename("max_rank").reset_index(), on="title"
)

res_rank["blue"] = res_rank.title.isin(
    [
        "American Economic Review",
        "Econometrica",
        "International Economic Review",
        "Journal of Economic Theory",
        "Journal of Political Economy",
        "Quarterly Journal of Economics",
        "Review of Economic Studies",
        "Review of Economics and Statistics",
    ]
)

# %%
res_rank.sort_values("min_rank")[
    ["title", "blue"]
    + [f"{i}_{m}_rank" for i in ["SJR", "H", "IF2"] for m in ["min", "avg"]]
].reset_index(drop=True).query("blue")

# %%
res_rank.sort_values("min_rank")[
    ["title", "blue"]
    + [f"{i}_{m}_rank" for i in ["SJR", "H", "IF2"] for m in ["min", "avg"]]
].reset_index(drop=True).head(20)


# %%
res_rank.sort_values("max_rank")[
    ["title", "blue"]
    + [f"{i}_{m}_rank" for i in ["SJR", "H", "IF2"] for m in ["min", "avg"]]
].reset_index(drop=True).query("blue")
# %%
res_rank.sort_values("max_rank")[
    ["title", "blue"]
    + [f"{i}_{m}_rank" for i in ["SJR", "H", "IF2"] for m in ["min", "avg"]]
].reset_index(drop=True).head(20)

# %%
def plot(
    state, filename=None, nedges=1000, hide_h=0, h_v_size=5.0, h_e_size=1.0, **kwargs
):
    """
    Plot the graph and group structure.
    optional:
    - filename, str; where to save the plot. if None, will not be saved
    - nedges, int; subsample  to plot (faster, less memory)
    - hide_h, int; wether or not to hide the hierarchy
    - h_v_size, float; size of hierarchical vertices
    - h_e_size, float; size of hierarchical edges
    - **kwargs; keyword arguments passed to self.state.draw method (https://graph-tool.skewed.de/static/doc/draw.html#graph_tool.draw.draw_hierarchy)
    """
    state.draw(
        layout="bipartite",
        output=filename,
        subsample_edges=nedges,
        hshortcuts=1,
        hide=hide_h,
        hvprops={"size": h_v_size},
        heprops={"pen_width": h_e_size},
        **kwargs,
    )
