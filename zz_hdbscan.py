# %%
import hdbscan
import matplotlib.pylab as plt
import numpy as np
import pandas
import scipy.sparse
import seaborn
import umap

# %%
m = scipy.sparse.load_npz("sparse.npz")
u = umap.UMAP(
    n_components=2,
    metric="cosine",
)
e = u.fit_transform(m)
# %%
plt.scatter(e[:, 0], e[:, 1], s=3)
# %%
cs = 400
for r in [100, 50, 25, 10, 1]:
    print(cs)
    df = pandas.DataFrame(
        [
            (
                lambda l: (
                    i,
                    l.max() + 1,
                    np.round(1 - ((l == -1).sum() / l.shape[0]), 3),
                )
            )(
                hdbscan.HDBSCAN(
                    min_cluster_size=i,
                    min_samples=1,
                ).fit_predict(e)
            )
            for i in range(max(2, cs - 10 * r), min(3000, cs + 10 * r), r)
        ],
        columns=["min_cluster_size", "n_cluster", "pct_classified"],
    )
    print(
        df[(df.n_cluster <= 10) & (df.n_cluster >= 5)].sort_values(
            "pct_classified", ascending=False
        )
    )
    cs = int(
        df[(df.n_cluster <= 10) & (df.n_cluster >= 5)]
        .sort_values("pct_classified", ascending=False)
        .iloc[0]
        .min_cluster_size
    )

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
