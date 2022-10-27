# %%
import graph_tool.all as gt
import numpy as np
import pickle

# %%
print("Load...")
g = gt.load_graph("graph.gt.gz")
print("Loaded!")

# %%
np.random.seed(42)
gt.seed_rng(42)

n_initial = 5
label = g.vp["kind"]
mdl = np.inf

for i in range(n_initial):
    print(f"Run {i+1} of {n_initial}")
    state_tmp = gt.minimize_nested_blockmodel_dl(
        g,
        state_args=dict(base_type=gt.OverlapBlockState, clabel=label),
        multilevel_mcmc_args=dict(verbose=True),
    )
    print(state_tmp)
    if mdl_tmp := state_tmp.entropy() < mdl:
        mdl = 1.0 * mdl_tmp
        state = state_tmp.copy()

load("Saving...")
pickle.dump(state, "state.pkl")
load("Saved!")
# %%
