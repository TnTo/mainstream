# %%
import numpy
import pandas
import mainstream

# %%

# %%
df = pandas.read_sql_table("vocabulary", "sqlite:///data.db").loc[
    numpy.load("hsbm/results_1003/3/Ps.npz")["p_tw_w"].argmax(axis=0) == 31, "word"
]
# %%
df.to_csv("latex_sw.csv", index=False)
# %%
df.apply(mainstream.clean_and_stem).to_csv("latex_sw_stem.csv", index=False)
# %%
