# %%
from mainstream import *

# %%
# json2sql()

# %%
clean_raw(min_year=2007, output="clean_small.db", tmp="merge_small.db")

# %%
clean_data(input="clean_small.db", output="data_small.db")

# %%
create_graph(input="data_small.db", output="graph_small.gt.gz")
