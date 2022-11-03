# %%
from mainstream import *

# %%
min_year = 1946  # 1946
max_year = 1949  # 2016
suffix = "_small"  # ""

# %%
print("json2sql")
# json2sql()

# %%
print("clean raw")
# clean_raw(
#    min_year=min_year,
#    max_year=max_year,
#    output=f"clean{suffix}.db",
#    tmp=f"merge{suffix}.db",
# )

# %%
print("clean data")
# clean_data(input=f"clean{suffix}.db", output=f"data{suffix}.db")

# %%
print("create graph")
# create_graph(input=f"data{suffix}.db", output=f"graph{suffix}.gt.gz", overlap=False)

# %%
print("infer topic model")
infer_tm(
    input=f"graph{suffix}.gt.gz",
    output_prefix=f"state{suffix}",
    verbose=True,
    seeds=[1000],
)


# %%
