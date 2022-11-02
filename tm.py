# %%
import graph_tool.all as gt
import numpy as np
import pickle

# %%
suffix = "_small"

# %%
print("Load...")
g = gt.load_graph(f"graph{suffix}.gt.gz")
print("Loaded!")

# %%
n_initial = 5
overlap = False
verbose = True

label = g.vp["kind"]
mdl = np.inf

state_args = {"clabel": label}
if not overlap:
    state_args["pclabel"] = label
    state_args["eweight"] = g.ep.count

# %%
for i in range(n_initial):
    np.random.seed(1000 + i)
    gt.seed_rng(1000 + i)

    print(f"Run {i+1} of {n_initial}")
    base_type = gt.OverlapBlockState if overlap else gt.BlockState
    state = gt.minimize_nested_blockmodel_dl(
        g,
        state_args=dict(base_type=base_type, **state_args),
        multilevel_mcmc_args=dict(verbose=verbose),
    )
    L = 0
    for s in state.levels:
        L += 1
        if s.get_nonempty_B() == 2:
            break
    state = state.copy(bs=state.get_bs()[:L] + [np.zeros(1)])
    print(state)
    print("Saving...")
    with open(f"state_{i}{suffix}.pkl", 'wb') as f:
        pickle.dump(state, f)
    print("Saved!")
    del state

# %%



 