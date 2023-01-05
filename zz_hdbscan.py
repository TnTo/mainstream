# %%
import numpy as np
import hdbscan
import matplotlib.pylab as plt
import pandas
import seaborn

# %%
path = "uh/stem_sw_None.npz"
# %%
e = np.load(path)["e"]
# %%
plt.scatter(e[:, 0], e[:, 1], s=3)
# %%
df = pandas.DataFrame(
    [
        (
            lambda l: (
                i,
                j,
                l.max() + 1,
                np.round(1 - ((l == -1).sum() / l.shape[0]), 3),
            )
        )(
            hdbscan.HDBSCAN(
                min_cluster_size=i,
                min_samples=j,
            ).fit_predict(e)
        )
        for i in range(10, 1001, 50)
        for j in [1, 10, 100, 250, 500, None]
    ],
    columns=["min_cluster_size", "min_sample", "n_cluster", "pct_classified"],
)
# %%
df["log_n_cluster"] = np.log10(df.n_cluster)
df.min_sample = df.min_sample.fillna("None")
# %%
seaborn.FacetGrid(
    data=df.melt(id_vars=["min_cluster_size", "min_sample"]),
    row="variable",
    hue="min_sample",
    sharey=False,
).map(seaborn.lineplot, "min_cluster_size", "value").add_legend()

# %%
seaborn.scatterplot(data=df, y="log_n_cluster", x="pct_classified", hue="min_sample")

# %%
df2 = (
    df.sort_values(["min_sample", "min_cluster_size"])
    .set_index(["min_cluster_size", "min_sample"])
    .groupby("min_sample")
    .diff(periods=1, axis=0)
    .reset_index()
)
# %%
df2[
    (df2.pct_classified < 0.1)
    & (np.abs(df2.n_cluster) >= 20)
    & (df2.min_sample.isin([1.0, 10.0]))
]
# %%
