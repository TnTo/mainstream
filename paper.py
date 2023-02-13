# %%
## IMPORT
import glob
import os
import sqlite3

import matplotlib
import pandas
import seaborn
import numpy
import scipy.sparse
import sklearn.metrics

seaborn.set_style("whitegrid")
seaborn.set_context("paper")
seaborn.set_palette("colorblind")
numpy.random.seed(8686)


def mkdir(p):
    try:
        os.mkdir(p)
    except:
        pass


# %%
m = scipy.sparse.load_npz("sparse.npz")
words = pandas.read_sql("vocabulary", "sqlite:///data.db").sort_values("id")
docs = pandas.read_sql(
    "document", "sqlite:///data.db", parse_dates=["date"]
).sort_values("id")
model = pandas.read_sql("model", "sqlite:///model.db")
model.kind = model.kind.replace({0: "D", 1: "W"})
graph = pandas.read_sql("graph", "sqlite:///data.db")

# %%
## JOURNAL CHOICE

# %%
### IMPACT FACTOR
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
data.columns = ["title", "year", "SJR", "H", "IF2"]

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

# data[i] = data.groupby("year")[i].rank(ascending=False, method="min")
df2 = pandas.concat(
    [
        data.groupby("title")[[i]]
        .mean()
        .sort_values(i, ascending=False)
        .head(20)
        .reset_index()
        for i in ["SJR", "H", "IF2"]
    ],
    axis=1,
)
df2.H = df2.H.astype(int)

# %%
with pandas.option_context("max_colwidth", 1000):
    open("paper/src/IF.tex", "w").write(
        df2.to_latex(
            index=False,
            float_format="%.2f",
            sparsify=False,
            column_format="@{}Yr|Yr|Yr@{}",
            header=[
                "SCImago Journal Rank",
                "",
                "H-index",
                "",
                "2-year Impact Factor",
                "",
            ],
            label="tab:IF",
            position="p",
            caption=(
                'The 20 journals with the highest average indicators in the period 1999-2021 for three different bibliometric indicators. Journals from the subject "Economics, Econometrics and Finance". Journals containing the words "Financ", "Marketing", "Account", "Business", "Entrepreneur" or "Consumer" are excluded. Sourced from \\url{https://www.scimagojr.com/journalrank.php}.',
                "Bibliometrics indicator for top journals",
            ),
            multicolumn=False,
        )
        .replace("{tabular}", "{tabularx}")
        .replace("\\begin{tabularx}", "\\begin{tabularx}{\\hsize}")
    )

# %%
# DATA SHRINKING
with sqlite3.connect("data.db") as db:
    cur = db.cursor()
    cur.execute("SELECT SUM(count) FROM graph")
    t2 = cur.fetchall()[0][0]
    cur.execute("SELECT COUNT(*) FROM vocabulary")
    w2 = cur.fetchall()[0][0]
    cur.execute("SELECT COUNT(*) FROM document")
    d2 = cur.fetchall()[0][0]

with sqlite3.connect("clean.db") as db:
    cur = db.cursor()
    cur.execute("SELECT SUM(count) FROM graph")
    t1 = cur.fetchall()[0][0]
    cur.execute("SELECT COUNT(*) FROM vocabulary")
    w1 = cur.fetchall()[0][0]
    cur.execute("SELECT COUNT(*) FROM document")
    d1 = cur.fetchall()[0][0]

with sqlite3.connect("raw.db") as db:
    cur = db.cursor()
    cur.execute("SELECT SUM(count) FROM graph")
    t0 = cur.fetchall()[0][0]
    cur.execute("SELECT COUNT(*) FROM vocabulary")
    w0 = cur.fetchall()[0][0]
    cur.execute("SELECT COUNT(*) FROM document")
    d0 = cur.fetchall()[0][0]

print("First cleaning phase:")
print(f"{d1} of {d0} initial documents ({round(d1/d0*100)}%)")
print(f"{w1} of {w0} initial words ({round(w1/w0*100)}%)")
print(f"{t1} of {t0} initial tokens ({round(t1/t0*100)}%)")
print("")

print("Second cleaning phase:")
print(f"{d2} of {d1} initial documents ({round(d2/d1*100)}%)")
print(f"{w2} of {w1} initial words ({round(w2/w1*100)}%)")
print(f"{t2} of {t1} initial tokens ({round(t2/t1*100)}%)")
print("")

print("All cleaning process:")
print(f"{d2} of {d0} initial documents ({round(d2/d0*100)}%)")
print(f"{w2} of {w0} initial words ({round(w2/w0*100)}%)")
print(f"{t2} of {t0} initial tokens ({round(t2/t0*100)}%)")
print("")

# %%
pandas.DataFrame(
    [
        ("Original dataset", d0, w0, t0),
        ("After first cleaning", d1, w1, t1),
        ("Final dataset", d2, w2, t2),
    ],
    columns=["", "Documents", "Words", "Tokens"],
).to_latex(
    "paper/src/dataset.tex",
    index=False,
    sparsify=False,
    column_format="l|rrr",
    label="tab:dataset",
    position="tb",
    caption=(
        "Number of documents, unique words and tokens in the dataset at three different stages of the cleaning process",
        "Dataset size",
    ),
    multicolumn=False,
)

# %%
# JOURNALS PROPORTIONS
df = docs.set_index("id")

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
fig, (ax1, ax2) = matplotlib.pyplot.subplots(1, 2, figsize=(7.5, 3))

df2 = df[
    (df.authors != "null")
    & (df.provider == "jstor")
    & (df.subtype == "research-article")
    & df.unigrams
    & df.journal.isin(journals)
    & (df.year >= 1946)
    & (df.year <= 2016)
]
seaborn.histplot(
    x="year",
    data=df2[["year", "journal"]],
    hue="journal",
    hue_order=journals,
    multiple="stack",
    bins=range(1947, 2018),
    edgecolor="black",
    linewidth=0.01,
    alpha=1,
    ax=ax1,
)

journal_year = (
    df2.groupby(["year", "journal"])
    .count()
    .type.reset_index()
    .pivot(columns="journal", values="type", index="year")
)
journal_year = journal_year.div(journal_year.sum(axis=1), axis=0)[
    list(reversed(journals))
]
journal_year.plot(
    kind="bar",
    stacked=True,
    rot=0,
    legend="reverse",
    align="center",
    width=1,
    color=list(reversed(seaborn.color_palette()[: len(journals)])),
    edgecolor="black",
    ax=ax2,
    linewidth=0.01,
    ylabel="Frequence"
)
ax2.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(10))

handles, labels = ax2.get_legend_handles_labels()
ax1.get_legend().remove()
ax2.get_legend().remove()
lgd = fig.legend(
    handles[::-1], labels[::-1], loc="center left", bbox_to_anchor=(1, 0.5)
)
matplotlib.pyplot.tight_layout()
matplotlib.pyplot.savefig(
    "paper/src/journals.pdf",
    transparent=True,
    bbox_extra_artists=(lgd,),
    bbox_inches="tight",
)


# %%
## MODEL SELECTION

# %%
open("paper/src/levels.tex", "w").write(
    model.groupby(["seed", "level"])
    .group.nunique()
    .replace({1: numpy.nan})
    .reset_index()
    .pivot(index="seed", columns="level", values="group")
    .astype("Int64")[list(range(7))]
    .to_latex(
        index=True,
        sparsify=False,
        column_format="l|rrrrrrrr",
        label="tab:levels",
        position="tb",
        na_rep="",
        caption=(
            "Number of groups (both words and documents) in each level of the hierarchy for each random seed",
            "Number of groups per level and seed",
        ),
        multicolumn=False,
    )
    .replace("<NA>", "")
)

# %%
# MI
Egroup = model[model.level == 3].groupby("kind").group.mean().apply(numpy.ceil)
ED = int(Egroup.D)
EW = int(Egroup.W)

data = model[model.level == 3].pivot(
    index=["id", "kind"], columns="seed", values="group"
)

idx = data.reset_index()[["id", "kind"]]
Didx = idx[idx.kind == "D"]
Didx["RNG"] = numpy.random.randint(ED, size=len(Didx))
Widx = idx[idx.kind == "W"]
Widx["RNG"] = numpy.random.randint(EW, size=len(Widx)) + ED
data = data.merge(
    pandas.concat([Didx, Widx]).set_index(["id", "kind"]),
    left_index=True,
    right_index=True,
)
mi = numpy.zeros((len(data.columns), len(data.columns)))
for i in range(len(data.columns)):
    for j in range(len(data.columns)):
        mi[i, j] = sklearn.metrics.normalized_mutual_info_score(
            data.iloc[:, i].values, data.iloc[:, j].values
        )

labels = model.seed.unique().tolist() + ["RND"]

matplotlib.pyplot.figure(figsize=(3.5, 3))
seaborn.heatmap(mi, annot=True, xticklabels=labels, yticklabels=labels, fmt=".2f")
matplotlib.pyplot.savefig(
    "paper/src/mi.pdf",
    transparent=True,
    bbox_inches="tight",
)

# %%
S = pandas.read_csv("entropy.csv")
S[S.level.isna()].sort_values("entropy")[["seed", "entropy"]].merge(
    S[S.level == 3].sort_values("entropy")[["seed", "entropy"]], on="seed"
).rename(
    columns={"entropy_x": "Model Entropy", "entropy_y": "Level 3 Entropy"}
).sort_values(
    "seed"
).to_latex(
    "paper/src/entropy.tex",
    index=False,
    label="tab:entropy",
    position="tb",
    caption=(
        "The table reports the entropy of the inferred partition for the whole model and for the level of intersed, per each random seed (lower is better)",
        "Model and level entropy",
    ),
)


# %%
## MODEL INTERPRETATION

# %%
labels = {
    0: "Industrial Organization",  # D
    1: "Labour",  # W
    2: "Game Theory",  # D
    3: "Applied Microeconomics - Labour",  # D
    4: "Labour",  # D
    5: "Production",  # W
    6: "Mathematics",  # W
    7: "International - Development",  # D
    8: "Microdata",  # W
    9: "Macro - Trade - Growth",  # D
    10: "Game Theory",  # W
    11: "Applied Microeconomics",  # D
    12: "Macroeconomics",  # D
    13: "Credit",  # W
    14: "Mathematics",  # D
    15: "Industrial Organization",  # W
    16: "StopWords1",  # W
    17: "Econometrics - Time",  # W
    18: "Macroeconomics",  # W
    19: "Econometrics",  # W
    20: "StopWords2",  # W
}

# %%
def dump(s, l, labels=None):
    mkdir("out")
    mkdir(f"out/{s}")
    mkdir(f"out/{s}/{l}")
    mkdir(f"out/{s}/{l}/W")
    mkdir(f"out/{s}/{l}/D")

    df = model[(model.seed == s) & (model.level == l)]

    T = (
        df[df.kind == "W"]
        .merge(words, on="id")[["group", "id", "word"]]
        .merge(
            pandas.Series(numpy.asarray(m.sum(axis=0)).squeeze(), name="n").astype(int),
            left_on="id",
            right_index=True,
        )
    )
    T["freq"] = T.n / T.groupby("group").n.transform("sum")
    for g in T.group.unique():
        print("Topic: ", g)
        if labels:
            print(labels[g])
        print(
            T[T.group == g][["word", "freq"]]
            .sort_values("freq", ascending=False)
            .head(10)
        )
        T[T.group == g][["word", "freq"]].sort_values("freq", ascending=False).to_csv(
            f"out/{s}/{l}/W/{g}.csv", index=False
        )

    G = df[df.kind == "D"].groupby("group").id.agg(list)
    for g in G.index:
        print("Group: ", g)
        if labels:
            print(labels[g])
        print(
            words.merge(
                pandas.Series(
                    numpy.asarray(
                        m[G.loc[g], :].sum(axis=0) / m[G.loc[g], :].sum()
                    ).squeeze(),
                    name="freq",
                ),
                left_on="id",
                right_index=True,
            )[["word", "freq"]]
            .sort_values("freq", ascending=False)
            .head(10)
        )
        words.merge(
            pandas.Series(
                numpy.asarray(
                    m[G.loc[g], :].sum(axis=0) / m[G.loc[g], :].sum()
                ).squeeze(),
                name="freq",
            ),
            left_on="id",
            right_index=True,
        )[["word", "freq"]].sort_values("freq", ascending=False).to_csv(
            f"out/{s}/{l}/D/{g}.csv", index=False
        )


dump(1000, 3, labels)

# %%
def tab_d(s, l, labels):
    with pandas.option_context("max_colwidth", 1000):
        open("paper/src/groups.tex", "w").write(
            pandas.DataFrame(
                [
                    (
                        labels[int(path.split("/")[-1].split(".")[0])],
                        ", ".join(pandas.read_csv(path).head(20).word.to_list()),
                    )
                    for path in glob.glob(f"out/{s}/{l}/D/*.csv")
                ],
                columns=["label", "words"],
            )
            .sort_values("label")
            .to_latex(
                header=False,
                index=False,
                column_format="lY",
                label="tab:groups",
                position="tb",
                caption=(
                    "The nine documents-group and the twenty most frequent words in each of them",
                    "Groups",
                ),
            )
            .replace("{tabular}", "{tabularx}")
            .replace("\\begin{tabularx}", "\\begin{tabularx}{\\hsize}")
        )


tab_d(1000, 3, labels)

# %%
def tab_w(s, l, labels):
    with pandas.option_context("max_colwidth", 1000):
        open("paper/src/topics.tex", "w").write(
            pandas.DataFrame(
                [
                    (
                        labels[int(path.split("/")[-1].split(".")[0])],
                        ", ".join(pandas.read_csv(path).head(20).word.to_list()),
                    )
                    for path in glob.glob(f"out/{s}/{l}/W/*.csv")
                ],
                columns=["label", "words"],
            )
            .sort_values("label")
            .to_latex(
                header=False,
                index=False,
                column_format="lY",
                label="tab:topics",
                position="tb",
                caption=(
                    "The twelve topics (words-group) and the twenty most frequent words for each of them",
                    "Topics",
                ),
            )
            .replace("{tabular}", "{tabularx}")
            .replace("\\begin{tabularx}", "\\begin{tabularx}{\\hsize}")
        )


tab_w(1000, 3, labels)

# %%
### G-T Composition
def gt(s, l, labels=None):
    df = model[(model.seed == s) & (model.level == l)]
    if labels:
        df.group = df.group.replace(labels)
    G = df[df.kind == "D"].groupby("group").id.agg(list)
    T = df[df.kind == "W"].groupby("group").id.agg(list)
    data = pandas.DataFrame(
        [(g, t, m[G.loc[g], :][:, T.loc[t]].sum()) for g in G.index for t in T.index],
        columns=["Group", "Topic", "N"],
    )
    data.N = data.N / data.groupby("Group").N.transform("sum")
    data = data.sort_values("N", ascending=False)
    out = []
    for g in data.Group.unique():
        buffer = ""
        for t in data[data.Group == g].itertuples():
            if t.N >= 0.1:
                buffer += f" {t.N:.2f}*{t.Topic.replace(' ', '~')} +"
        out.append((g, buffer + " ..."))
    with pandas.option_context("max_colwidth", 1000):
        open("paper/src/gt.tex", "w").write(
            pandas.DataFrame(out, columns=["group", "topics"])
            .sort_values("group")
            .to_latex(
                header=False,
                index=False,
                column_format="lY",
                label="tab:gt",
                position="tb",
                caption=(
                    "The composition of each group as a mixture of topics. Only the topics which represent at least 10\\% of the group are listed",
                    "Groups-Topics",
                ),
                escape=False,
            )
            .replace("{tabular}", "{tabularx}")
            .replace("\\begin{tabularx}", "\\begin{tabularx}{\\hsize}")
        )


gt(1000, 3, labels)

# %%
### Most similar papers
def similar_papers(s, l, labels=None, n=5):
    df = model[(model.seed == s) & (model.level == l)]
    if labels:
        df.group = df.group.replace(labels)
    G = df[df.kind == "D"].groupby("group").id.agg(list)

    simil = (
        pandas.DataFrame(
            sklearn.metrics.pairwise.cosine_similarity(
                m,
                numpy.array(
                    [
                        numpy.asarray(m[G.loc[g], :].sum(axis=0)).squeeze()
                        for g in G.index
                    ]
                ),
            ),
            columns=G.index,
        )
        .reset_index()
        .melt(id_vars=("index",))
    )
    simil.columns = ["id", "group", "sim"]
    simil = (
        simil.set_index("id")
        .groupby("group")
        .sim.nlargest(n)
        .reset_index()
        .rename(columns={"level_1": "id"})
        .merge(docs, on="id")
    ).sort_values(["group", "sim"], ascending=[True, False])[
        ["group", "title", "authors", "year", "journal"]
    ]
    simil.title = simil.title.str.replace("&", "\&")
    simil.authors = simil.apply(lambda row: ", ".join(eval(row.authors)), axis=1)
    col_fmt = "sYsrs"
    label = "tab:papers"
    pos = "p"
    with pandas.option_context("max_colwidth", 1000):
        open("paper/src/papers.tex", "w").write(
            simil.to_latex(
                header=False,
                index=False,
                column_format=col_fmt,
                label=label,
                position=pos,
                caption=(
                    "The five papers most representative of each group, measured by the cosine similarity with the word count in the topic as whole",
                    "Representative papers",
                ),
                escape=False,
            )
            .replace("\\centering", "")
            .replace(f"\\begin{{tabular}}{{{col_fmt}}}", "")
            .replace("\\toprule", "")
            .replace("\\bottomrule", "")
            .replace("\\end{tabular}", "")
            .replace(
                f"\\begin{{table}}[{pos}]",
                f"\\begin{{xltabular}}[{pos}]{{\\hsize}}{{{col_fmt}}}",
            )
            .replace(
                f"\\label{{{label}}}",
                f"\\label{{{label}}} \\\\ \\toprule \\endhead \\hline \\multicolumn{{5}}{{r@{{}}}}{{\\textit{{continues on  next page}}}}\\\\ \\hline \\endfoot \\hline \\endlastfoot",
            )
            .replace("\\end{table}", "\\end{xltabular}")
        )


similar_papers(1000, 3, labels)

# %%
def plot(s, l, labels=None):

    df = (
        model[(model.seed == s) & (model.level == l) & (model.kind == "D")]
        .merge(docs, on="id")
        .groupby(["year", "group"])
        .count()
        .id.rename("N")
        .reset_index()
    )

    if labels:
        df.group = df.group.replace(labels)

    matplotlib.pyplot.figure(figsize=(7.5, 3))

    g = seaborn.lineplot(
        data=df.sort_values("year")
        .set_index("year")
        .groupby("group")
        .rolling(10, center=True)
        .N.mean()
        .reset_index()
        .astype({"group": "category"}),
        x="year",
        y="N",
        hue="group",
        color="colorblind",
    )

    seaborn.move_legend(g, loc="center left", bbox_to_anchor=(1, 0.5))
    matplotlib.pyplot.tight_layout()
    matplotlib.pyplot.savefig(
        "paper/src/groups.pdf",
        transparent=True,
        bbox_inches="tight",
    )


plot(1000, 3, labels)


# %%
# Topic over time
def plot_topics(s, l, labels=None):
    df = model[(model.seed == s) & (model.level == l)]
    if labels:
        df.group = df.group.replace(labels)
    Y = df[df.kind == "D"].merge(docs, on="id").groupby("year").id.agg(list)
    T = df[df.kind == "W"].groupby("group").id.agg(list)
    data = pandas.DataFrame(
        [(y, t, m[Y.loc[y], :][:, T.loc[t]].sum()) for y in Y.index for t in T.index],
        columns=["Year", "Topic", "N"],
    )
    data.N = data.N / data.groupby("Year").N.transform("sum")

    matplotlib.pyplot.figure(figsize=(7.5, 3))

    g = seaborn.lineplot(
        data=data.sort_values("Year")
        .set_index("Year")
        .groupby("Topic")
        .rolling(10, center=True)
        .N.mean()
        .reset_index()
        .astype({"Topic": "category"}),
        x="Year",
        y="N",
        hue="Topic",
        color="colorblind",
    )

    seaborn.move_legend(g, loc="center left", bbox_to_anchor=(1, 0.5))
    matplotlib.pyplot.tight_layout()
    matplotlib.pyplot.savefig(
        "paper/src/topics.pdf",
        transparent=True,
        bbox_inches="tight",
    )


plot_topics(1000, 3, labels)

# %%
def subgroups(s, l, labels):
    df = model[(model.seed == s) & (model.level == l) & (model.kind == "D")]
    if labels:
        df.group = df.group.replace(labels)
    df = df.merge(
        model[(model.seed == s) & (model.level == l - 1) & (model.kind == "D")], on="id"
    )
    df.groupby("group_x").group_y.nunique().sort_values(ascending=False).to_latex(
        "paper/src/subgroups.tex",
        index=True,
        index_names=False,
        header=False,
        label="tab:subgroups",
        caption=(
            "The number of groups in the more detailed level of the hierarchy for each group in the level of interest",
            "Subgroups",
        ),
        position="tb",
    )


subgroups(1000, 3, labels)

###########################################

# %%
# def plot_word_N(s, l, labels=None):
#     df = model[(model.seed == s) & (model.level == l) & (model.kind == "D")]
#     if labels:
#         df.group = df.group.replace(labels)
#     df = df.merge(docs, on="id")
#     df = graph.merge(df[["year", "group", "id"]], left_on="document_id", right_on="id")
#     data1 = (
#         df.groupby(["year", "group"])
#         .word_id.nunique()
#         .reset_index()
#         .sort_values("year")
#         .set_index("year")
#         .groupby("group")
#         .rolling(10, center=True)
#         .word_id.mean()
#         .reset_index()
#     )  # n unique words per year and group
#     data2 = (
#         docs.groupby("year")
#         .id.count()
#         .reset_index()
#         .sort_values("year")
#         .set_index("year")
#         .rolling(10, center=True)
#         .id.mean()
#         .reset_index()
#     )  # n document per year

#     matplotlib.pyplot.figure(figsize=(7.5, 3))

#     g = seaborn.lineplot(
#         data=data1,
#         x="year",
#         y="word_id",
#         hue="group",
#     )
#     g.set(ylabel="unique words")
#     ax2 = matplotlib.pyplot.twinx()
#     seaborn.lineplot(data=data2, x="year", y="id", ax=ax2, dashes=(2, 2))
#     ax2.set(ylabel="n docs")

#     seaborn.move_legend(g, loc="center left", bbox_to_anchor=(1.1, 0.5))
#     matplotlib.pyplot.tight_layout()
#     matplotlib.pyplot.savefig(
#         "paper/src/uwords_yg_ndoc.pdf",
#         transparent=True,
#         bbox_inches="tight",
#     )


# plot_word_N(1000, 3, labels)

# %%
def plot_word_y():

    data = graph.merge(
        docs[["year", "id"]], left_on="document_id", right_on="id"
    ).groupby("year")

    fig, (ax1, ax2, ax3) = matplotlib.pyplot.subplots(1, 3, figsize=(7.5, 3))

    uw = (
        data.word_id.nunique()
        .reset_index()
        .set_index("year")
        .word_id.reset_index()
        .sort_values("year")
        .set_index("year")
        .rolling(10, center=True)
        .mean()
        .rename(columns={"word_id": "Unique Words"})
    )
    tw = (
        data["count"]
        .sum()
        .reset_index()
        .set_index("year")["count"]
        .reset_index()
        .sort_values("year")
        .set_index("year")
        .rolling(10, center=True)
        .mean()
        .rename(columns={"count": "Total Words"})
    )
    nd = (
        docs.groupby("year")
        .id.count()
        .reset_index()
        .sort_values("year")
        .set_index("year")
        .rolling(10, center=True)
        .id.mean()
        .rename("N Documents")
    )

    df = (
        pandas.merge(uw, tw, left_index=True, right_index=True)
        .merge(nd, left_index=True, right_index=True)
        .reset_index()
    )

    seaborn.lineplot(
        df,
        x="year",
        y="Unique Words",
        color=seaborn.color_palette(n_colors=3)[0],
        ax=ax1,
    )
    ax1.set(ylabel="", title="Unique Words")
    seaborn.lineplot(
        df,
        x="year",
        y="Total Words",
        color=seaborn.color_palette(n_colors=3)[1],
        ax=ax2,
    )
    ax2.set(ylabel="", title="Total Words")
    seaborn.lineplot(
        df,
        x="year",
        y="N Documents",
        color=seaborn.color_palette(n_colors=3)[2],
        ax=ax3,
    )
    ax3.set(ylabel="", title="N Documents")

    matplotlib.pyplot.tight_layout()
    matplotlib.pyplot.savefig(
        "paper/src/uwords.pdf",
        transparent=True,
        bbox_inches="tight",
    )


plot_word_y()

# %%
