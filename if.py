# %%
import pandas

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
