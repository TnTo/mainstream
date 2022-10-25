# %%
import pandas as pandas
import sqlite3

# %%
d = pandas.read_sql("SELECT * FROM doc_len;", "sqlite:///clean.db")
w = pandas.read_sql("SELECT * FROM word_count;", "sqlite:///clean.db")
dmeta = pandas.read_sql("document", "sqlite:///clean.db")

# %%
dq = d[["len"]].quantile(q=[x / 20 for x in range(0, 21, 1)])
wq = w[["freq"]].quantile(q=[x / 20 for x in range(0, 21, 1)])
s = w.freq.sum()
l = len(w)
for i in range(1, 11):
    for j in range(1, 3):
        w0 = w[(w.word.str.len() > j) & (w.freq >= i)]
        print(
            f"min_freq: {i}; min_len: {j+1}; preserved vocab: {len(w0) / l}; preserved mass: {w0.freq.sum() / s}"
        )

# %%
d[d.len >= 1000].reset_index(drop=True)[["document_id"]].merge(
    dmeta, how="inner", left_on="document_id", right_on="id"
).drop(columns=["id"]).to_sql(
    "document", "sqlite:///data.db", index=True, index_label="id"
)

#%%
w[(w.word.str.len() > 2) & (w.freq >= 10)][["word_id", "word"]].reset_index(
    drop=True
).to_sql("vocabulary", "sqlite:///data.db", index=True, index_label="id")

# %%
with sqlite3.connect("data.db") as db:
    cur = db.cursor()
    cur.executescript(
        """
ATTACH DATABASE "clean.db" AS CleanDB;

CREATE TABLE graph AS SELECT document_id, id AS word_id, count FROM (
		(
            SELECT id AS document_id, word_id, count FROM (
			    CleanDB.graph AS graph INNER JOIN document USING (document_id)
		    )
        ) AS graph INNER JOIN vocabulary USING (word_id)
);
    """
    )

# %%
