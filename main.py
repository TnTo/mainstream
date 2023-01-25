# %%
from mainstream import *

# %%
suffix = ""
verbose = True
seeds = [1000, 1001, 1002]
overlap = True

# %%
print("json2sql")
# json2sql()

# %%
print("clean raw")
# clean_raw(
#     output=f"clean{suffix}.db",
#     tmp=f"merge{suffix}.db",
#     min_word_len=3,
#     clean_fn=clean_and_stem,
# )

# %%
print("clean data")
# clean_data(
#     input=f"clean{suffix}.db",
#     output=f"data{suffix}.db",
#     min_doc_len=1500,  # ~3/4 pp
#     min_doc_occurencies=35,  # 1/1.000 of docs
#     max_doc_occurencies=9000,  # ~25% of docs
#     min_word_freq=0,
#     min_word_len=0,
#     sw="latex.csv",
# )

# %%
print("create graph")
# create_graph(input=f"data{suffix}.db", output=f"graph{osuffix}.gt.gz")

# %%
print("create matrix")
# create_sparse(input=f"data{suffix}.db", output=f"sparse{suffix}.npz")

# %%
print("infer hsbm topic model")
# infer_hsbm_tm(
#     input=f"graph{suffix}.gt.gz",
#     output_prefix=f"state{suffix}",
#     verbose=verbose,
#     seeds=seeds,
# )
