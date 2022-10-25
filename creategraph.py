# %%
import sqlite3
import graph_tool.all as gt
from collections import defaultdict

# %%
g = gt.Graph(directed=False)
id = g.vp["id"] = g.new_vp("int")
kind = g.vp["kind"] = g.new_vp("int")
ecount = g.ep["count"] = g.new_ep("int")

docs_add = defaultdict(lambda: g.add_vertex())
words_add = defaultdict(lambda: g.add_vertex())

with sqlite3.connect("data.db") as db:
    cur = db.cursor()
    max_doc_id = cur.execute("SELECT MAX(document_id) FROM graph").fetchone()[0]
    max_word_id = cur.execute("SELECT MAX(word_id) FROM graph").fetchone()[0]
    lag = max_doc_id + 1

    for i_d in range(max_doc_id + 1):
        d = docs_add[i_d]
        id[d] = i_d
        kind[d] = 0

    for i_w in range(max_doc_id + 1):
        w = docs_add[i_w]
        id[w] = i_w
        kind[w] = 1

    for i in cur.execute("SELECT * FROM graph").fetchall():
        i_d, i_w, c = i
        e = g.add_edge(i_d, lag + i_w)
        ecount[e] = c

g.save("graph.gt.gz")

# %%
