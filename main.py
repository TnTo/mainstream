# %%
from mainstream import *

# %%
min_year = 1946
max_year = 2016
suffix = "_stem_sw"
# verbose = False
# seeds = [1000, 1001, 1002, 1003, 1004]

# %%
print("json2sql")
# json2sql()

# %%
print("clean raw")
clean_raw(
    min_year=min_year,
    max_year=max_year,
    output=f"clean{suffix}.db",
    tmp=f"merge{suffix}.db",
    min_word_len=3,
    clean_fn=clean_and_stem,
)

# %%
print("clean data")
# clean_data(
#    input=f"clean{suffix}.db",
#    output=f"data{suffix}.db",
#    min_doc_occurencies=5,
#    max_doc_occurencies=20000,
# )

# %%
print("create graph")
# create_graph(input=f"data{suffix}.db", output=f"hsbm/graph{suffix}.gt.gz", overlap=False)

# %%
print("create matrix")
# create_sparse(input=f"data{suffix}.db", output=f"sparse{suffix}.npz")

# %%
print("infer hsbm topic model")
# infer_hsbm_tm(
#    input=f"hsbm/graph{suffix}.gt.gz",
#    output_prefix=f"hsbm/state{suffix}",
#    verbose=verbose,
#    seeds=seeds,
# )

# %%
print("dump hsbm topic model")
# dump_hsbm_tm(
#    graph_input=f"hsbm/graph{suffix}.gt.gz",
#    input_prefix=f"hsbm/state{suffix}",
#    output_prefix=f"hsbm/results{suffix}",
#    seeds=seeds,
# )
