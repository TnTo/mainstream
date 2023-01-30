# %%
## IMPORT
import matplotlib
import pandas
import seaborn

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

for i in ["SJR", "H", "IF2"]:
    data[i] = data.groupby("year")[i].rank(ascending=False, method="min")

data = data.melt(["title", "year"])

data["blue"] = data.title.isin(
    [
        "American Economic Review",
        "Econometrica",
        # "International Economic Review",
        # "Journal of Economic Theory",
        "Journal of Political Economy",
        "Quarterly Journal of Economics",
        "Review of Economic Studies",
        "Review of Economics and Statistics",
    ]
)

seaborn.FacetGrid(data=data[data.blue], col="variable", sharey=False).map_dataframe(
    seaborn.lineplot, x="year", y="value", hue="title"
).add_legend()

# print(data[(data.year == 2000) & (data.variable == "SJR")].sort_values("value").head(20))
# print(data[(data.year == 2018) & (data.variable == "IF2")].sort_values("value").head(20))

# %%
# JOURNALS PROPORTIONS
# df = pandas.read_sql_table(
#     "document", "sqlite:///raw.db", index_col="id", parse_dates=["date"]
# )
df2 = pandas.read_sql_table(
    "document", "sqlite:///data.db", index_col="id", parse_dates=["date"]
)

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
def plot_journals(df):
    df2 = df[
        (df.authors != "null")
        & (df.provider == "jstor")
        & (df.subtype == "research-article")
        & df.unigrams
        & df.journal.isin(journals)
        & (df.year >= 1946)
        & (df.year <= 2016)
    ]
    ax = seaborn.histplot(
        x="year",
        data=df2[["year", "journal"]],
        hue="journal",
        hue_order=journals,
        multiple="stack",
        bins=range(1947, 2018),
    )
    seaborn.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
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
# plot_journals(df)
plot_journals(df2)

# %%
