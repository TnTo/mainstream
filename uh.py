# %%
import mainstream

# %%
u, e, c, l = mainstream.infer_uh_model(
    n_neighbors=30, min_dist=0.01, min_cluster_size=1000, seeds=[None]
)

# %%
umap.plot.points(u, labels=l)

# %%
c.condensed_tree_.plot(selected_clusters=True)
c.single_linkage_tree_.plot(selected_clusters=True)
