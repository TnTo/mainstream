# %%
from mainstream import *

# %%
suffix = ""
verbose = True
seeds = [1000]
overlap = False

# %%
print("json2sql")
# json2sql()

# %%
print("clean raw")
clean_raw(
    output=f"clean{suffix}.db",
    tmp=f"merge{suffix}.db",
    min_word_len=3,
    clean_fn=clean_and_stem,
)

# %%
print("clean data")
clean_data(
    input=f"clean{suffix}.db",
    output=f"data{suffix}.db",
    min_doc_len=1500,  # ~3/4 pp
    min_doc_occurencies=35,  # 1/1.000 of docs
    max_doc_occurencies=9000,  # ~25% of docs
    min_word_freq=0,
    min_word_len=0,
    sw="latex.csv",
)

# %%
print("create graph")
osuffix = f"{suffix}_o" if overlap else suffix
create_graph(input=f"data{osuffix}.db", output=f"graph{osuffix}.gt.gz", overlap=overlap)

# %%
print("create matrix")
create_sparse(input=f"data{suffix}.db", output=f"sparse{suffix}.npz")

# %%
print("infer hsbm topic model")
osuffix = f"{suffix}_o" if overlap else suffix
infer_hsbm_tm(
    input=f"hsbm/graph{osuffix}.gt.gz",
    output_prefix=f"hsbm/state{osuffix}",
    verbose=verbose,
    seeds=seeds,
    overlap=overlap,
)

# %%
print("dump hsbm topic model")
dump_hsbm_tm(
    graph_input=f"hsbm/graph{suffix}.gt.gz",
    input_prefix=f"hsbm/state{suffix}",
    output_prefix=f"hsbm/results{suffix}",
    seeds=seeds,
)
