# %%
import scipy.sparse
import umap
import hdbscan
import umap.plot

# %%
m = scipy.sparse.load_npz("sparse_stem_sw.npz")
u = umap.UMAP(
    n_components=2,
    metric="cosine",
)
e = u.fit_transform(m)
# %%
c = hdbscan.HDBSCAN(
    min_cluster_size=min_cluster_size,
    min_samples=min_samples,
)
l = c.fit_predict(e)

####
# Stem + SW
# min_cluster_size, min_samples
# 155,155 #47 clusters
# 380,1 #25 clusters

# %%
umap.plot.points(u, labels=l)

# %%
c.condensed_tree_.plot(select_clusters=True)

# %%
c.single_linkage_tree_.plot()

# %%
