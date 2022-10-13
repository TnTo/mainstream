# %%
import pandas as pandas
from tqdm.auto import tqdm
import seaborn
import matplotlib
import matplotlib.pyplot as plt

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
