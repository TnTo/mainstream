import glob
import json
import os
import pickle
import re
import sqlite3
from collections import defaultdict
from multiprocessing import Pool

import graph_tool.all as gt
import hdbscan
import nltk
import numpy as np
import pandas
import scipy.sparse
import umap

### modin ###
# import modin.pandas as pandas
# import ray


# def my_to_sql(self, *args, **kwargs):
#    return pandas.dataframe.DataFrame._to_pandas(self).to_sql(*args, **kwargs)


# pandas.base.BasePandasDataset.to_sql = my_to_sql

# ray.init(
#    ignore_reinit_error=True,
#    runtime_env={"env_vars": {"__MODIN_AUTOIMPORT_PANDAS__": "1"}},
# )
### end modin ###

stemmer = nltk.stem.SnowballStemmer(language="english")

# Get Constallate Data and Tabulate them into SQLite
def json2sql(input="data", output="raw.db"):
    paths = glob.glob(f"{input}/*.jsonl")
    with sqlite3.connect(f"{output}") as db:
        cur = db.cursor()

        cur.execute("PRAGMA foreign_keys = ON;")
        cur.execute(
            """
            CREATE TABLE document(
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                authors TEXT,
                date TEXT,
                year INTEGER,
                type TEXT,
                subtype TEXT,
                sourceid TEXT,
                url TEXT,
                doi TEXT,
                journal TEXT,
                title TEXT,
                lenght INTEGER,
                unigrams BOOLEAN,
                provider TEXT,
                volume INTEGER,
                pagestart INTEGER,
                pageend INTEGER,
                CHECK (unigrams IN (0,1))
            );
        """
        )
        cur.execute(
            """
            CREATE TABLE vocabulary(
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                word TEXT UNIQUE
            );
        """
        )
        cur.execute(
            """
            CREATE TABLE graph(
                document_id INTEGER,
                word_id INTEGER,
                count INTEGER,
                PRIMARY KEY (document_id, word_id),
                FOREIGN KEY (document_id) REFERENCES document(id),
                FOREIGN KEY (word_id) REFERENCES vocabulary(id)
            )
            WITHOUT ROWID;
        """
        )
        for p in paths:
            with open(p, "r") as f:
                while l := f.readline():
                    j = json.loads(l)
                    unigram = 1 if "unigram" in j.get("outputFormat") else 0
                    cur.execute(
                        """
                        INSERT INTO document(
                            authors, date, year, type, subtype, 
                            sourceid, url, doi, journal, 
                            title, lenght, unigrams, provider, 
                            volume, pagestart, pageend
                        ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                        """,
                        (
                            json.dumps(j.get("creator")),
                            j.get("datePublished"),
                            j.get("publicationYear"),
                            j.get("docType"),
                            j.get("docSubType"),
                            j.get("id"),
                            j.get("url"),
                            (
                                [
                                    i
                                    for i in (j.get("identifier") or [])
                                    if i.get("name") == "local_doi"
                                ]
                                or [dict()]
                            )[0].get("value"),
                            j.get("isPartOf"),
                            j.get("title"),
                            j.get("wordCount"),
                            unigram,
                            j.get("provider"),
                            j.get("volumeNumber"),
                            j.get("pageStart"),
                            j.get("pageEnd"),
                        ),
                    )
                    doc_id = cur.lastrowid
                    if unigram:
                        for w, c in j["unigramCount"].items():
                            cur.execute(
                                "INSERT OR IGNORE INTO vocabulary(word) VALUES (?);",
                                (w,),
                            )
                            cur.execute(
                                "SELECT id FROM vocabulary WHERE word = ?;", (w,)
                            )
                            word_id = cur.fetchone()[0]
                            cur.execute(
                                "INSERT INTO graph(document_id, word_id, count) VALUES (?,?,?);",
                                (doc_id, word_id, c),
                            )


def clean_str(s: str) -> str:
    return re.sub(r"[^a-zA-Z]", "", s).lower()


def clean_and_stem(s: str) -> str:
    return stemmer.stem(clean_str(s))


# Perform basic cleaning operation
# WORDS
# Delete non alphabetical (plus -) characters
# Drop single character words
# DOCS
# require author, jstor, research-article, and timespan
def clean_raw(
    input="raw.db",
    output="clean.db",
    tmp="merge.db",
    journals=[
        "Econometrica",
        "The American Economic Review",
        "The Review of Economic Studies",
        "The Quarterly Journal of Economics",
        "The Review of Economics and Statistics",
        "Journal of Political Economy",
        "The Economic Journal",
        "Economica",
    ],
    min_year=1946,
    max_year=2016,
    clean_fn=clean_str,
    min_word_len=2,
):
    docs = (
        pandas.read_sql_table(
            "document", f"sqlite:///{input}", index_col="id", parse_dates=["date"]
        )
        .query(
            f'(authors != "null") \
        & (provider == "jstor") \
        & (subtype == "research-article") \
        & unigrams \
        & journal.isin(@journals) \
        & (year >= {min_year}) \
        & (year <= {max_year})'
        )
        .reset_index()
        .rename(columns={"id": "old_id"})
    )
    docs.drop(columns="old_id").to_sql(
        "document", f"sqlite:///{output}", index=True, index_label="id"
    )
    docs[["old_id"]].copy().reset_index().rename(columns={"index": "id"}).set_index(
        "old_id"
    ).to_sql("docs", f"sqlite:///{tmp}", index=True, index_label="old_id")

    del docs

    words = pandas.read_sql("vocabulary", f"sqlite:///{input}", index_col="id")
    words = words.assign(word_clean=words.word.apply(clean_fn))
    vocabulary = pandas.Series(
        words.query("word_clean.str.len() >= @min_word_len").word_clean.unique(),
        name="word",
    )
    vocabulary.to_sql(
        "vocabulary",
        f"sqlite:///{output}",
        index=True,
        index_label="id",
        chunksize=10000,
    )

    words.reset_index().merge(
        vocabulary.reset_index(), how="right", left_on="word_clean", right_on="word"
    )[["id", "index"]].rename(columns={"id": "old_id", "index": "id"}).set_index(
        "old_id"
    )[
        ["id"]
    ].to_sql(
        "words", f"sqlite:///{tmp}", index=True, index_label="old_id", chunksize=10000
    )

    del vocabulary
    del words

    with sqlite3.connect(output) as db:
        cur = db.cursor()
        cur.executescript(
            f"""
            ATTACH DATABASE "{input}" AS RawDB;
            ATTACH DATABASE "{tmp}" AS MergeDB;

            CREATE TABLE graph AS SELECT document_id, id AS word_id, count FROM (
                    (
                        SELECT id AS document_id, word_id, count FROM (
                            RawDB.graph AS graph INNER JOIN MergeDB.docs AS docs ON graph.document_id = docs.old_id
                        )
                    ) AS graph INNER JOIN MergeDB.words AS words ON graph.word_id = words.old_id
            );
            """
        )
        cur.execute(
            "CREATE VIEW word_count AS SELECT vocabulary.word, freqs.word_id, freqs.freq FROM ((SELECT word_id, sum(count) AS freq FROM graph GROUP BY word_id) AS freqs JOIN vocabulary on freqs.word_id = vocabulary.id);"
        )
        cur.execute(
            "CREATE VIEW doc_len AS SELECT lens.document_id, lens.len FROM ((SELECT document_id, sum(count) AS len FROM graph GROUP BY document_id) AS lens JOIN document on lens.document_id = document.id);"
        )


def clean_data(
    input="clean.db",
    output="data.db",
    min_doc_len=1500,
    min_word_len=3,
    min_word_freq=10,
    min_doc_occurencies=None,
    max_doc_occurencies=None,
    sw=None,
):
    d = pandas.read_sql("SELECT * FROM doc_len", f"sqlite:///{input}")
    w = pandas.read_sql("SELECT * FROM word_count", f"sqlite:///{input}")
    dmeta = pandas.read_sql("document", f"sqlite:///{input}")

    if min_doc_occurencies or max_doc_occurencies:
        g = pandas.read_sql_query(
            "SELECT word_id, COUNT(DISTINCT document_id) AS n_docs FROM graph GROUP BY word_id",
            f"sqlite:///{input}",
        )
        if not min_doc_occurencies:
            min_doc_occurencies = 0
        if not max_doc_occurencies:
            max_doc_occurencies = 0
        w = w.merge(
            g[
                (g.n_docs < max_doc_occurencies) & (g.n_docs >= min_doc_occurencies)
            ].word_id,
            how="inner",
            on="word_id",
        )

    if min_word_len or min_word_freq:
        w = w[(w.word.str.len() >= min_word_len) & (w.freq >= min_word_freq)]

    if sw:
        w = w[~(w.word.isin(pandas.read_csv(sw, header=None)[0]))]

    w[["word_id", "word"]].reset_index(drop=True).to_sql(
        "vocabulary", f"sqlite:///{output}", index=True, index_label="id"
    )

    d.reset_index(drop=True)[["document_id"]].merge(
        dmeta[dmeta.lenght >= min_doc_len],
        how="inner",
        left_on="document_id",
        right_on="id",
    ).drop(columns=["id"]).to_sql(
        "document", f"sqlite:///{output}", index=True, index_label="id"
    )

    with sqlite3.connect(output) as db:
        cur = db.cursor()
        cur.executescript(
            f"""
            ATTACH DATABASE "{input}" AS CleanDB;

            CREATE TABLE graph AS SELECT document_id, id AS word_id, count FROM (
                    (
                        SELECT id AS document_id, word_id, count FROM (
                            CleanDB.graph AS graph INNER JOIN document USING (document_id)
                        )
                    ) AS graph INNER JOIN vocabulary USING (word_id)
            );
        """
        )


def create_graph(input="data.db", output="graph.gt.gz", overlap=False):
    g = gt.Graph(directed=False)
    id = g.vp["id"] = g.new_vp("int")
    kind = g.vp["kind"] = g.new_vp("int")
    if not overlap:
        count = g.ep["count"] = g.new_ep("int")

    docs_add = defaultdict(lambda: g.add_vertex())
    words_add = defaultdict(lambda: g.add_vertex())

    with sqlite3.connect(input) as db:
        cur = db.cursor()
        max_doc_id = cur.execute("SELECT MAX(document_id) FROM graph").fetchone()[0]
        max_word_id = cur.execute("SELECT MAX(word_id) FROM graph").fetchone()[0]
        lag = max_doc_id + 1

        for i_d in range(max_doc_id + 1):
            d = docs_add[i_d]
            id[d] = i_d
            kind[d] = 0

        for i_w in range(max_word_id + 1):
            w = words_add[i_w]
            id[w] = i_w
            kind[w] = 1

        for i in cur.execute("SELECT * FROM graph").fetchall():
            i_d, i_w, c = i
            if overlap:
                for _ in range(c):
                    g.add_edge(i_d, lag + i_w)
            else:
                e = g.add_edge(i_d, lag + i_w)
                count[e] = c

    g.save(output)


def create_sparse(input="data.db", output="sparse.npz"):
    with sqlite3.connect(input) as db:
        cur = db.cursor()
        ND = cur.execute("SELECT MAX(document_id) FROM graph").fetchone()[0] + 1
        NW = cur.execute("SELECT MAX(word_id) FROM graph").fetchone()[0] + 1
    m = scipy.sparse.coo_array((ND, NW))
    data = pandas.read_sql("graph", f"sqlite:///{input}", chunksize=500000)
    for d in data:
        m = m + scipy.sparse.coo_array(
            (d["count"], (d.document_id, d.word_id)), shape=(ND, NW)
        )
    scipy.sparse.save_npz(output, m)


def infer_hsbm_tm(
    input="graph.gt.gz",
    output_prefix="state",
    verbose=False,
    seeds=[1000, 1001, 1002, 1003, 1004],
):
    print("Load...")
    g = gt.load_graph(input)
    print("Loaded!")

    label = g.vp["kind"]

    state_args = {"clabel": label}
    state_args["pclabel"] = label
    state_args["eweight"] = g.ep.count

    for s in seeds:
        np.random.seed(s)
        gt.seed_rng(s)

        print(f"Seed {s}")
        state = gt.minimize_nested_blockmodel_dl(
            g,
            state_args=dict(base_type=gt.BlockState, **state_args),
            multilevel_mcmc_args=dict(verbose=verbose),
        )
        L = 0
        for l in state.levels:
            L += 1
            if l.get_nonempty_B() == 2:
                break
        state = state.copy(bs=state.get_bs()[:L] + [np.zeros(1)])
        print(state)
        print("Saving...")
        with open(f"{output_prefix}_{s}.pkl", "wb") as f:
            pickle.dump(state, f, -1)
        print("Saved!")
        del state


# def infer_hsbm_tm_soft(
#     input_prefix="state",
#     output_prefix="state_o",
#     niter=100,
#     seeds=[1000, 1001, 1002, 1003, 1004],
# ):
#     for s in seeds:
#         print(f"Seed {s}")
#         np.random.seed(s)
#         gt.seed_rng(s)
#         state = pickle.load(open(f"{input_prefix}_{s}.pkl", "rb")).copy(
#             state_args=dict(overlap=True)
#         )
#         print("Loaded!")
#         dS, nmoves = 0, 0
#         for i in range(niter):
#             if i % 10 == 0:
#                 print(i, "of", niter)
#             ret = state.multiflip_mcmc_sweep(niter=10, verbose=True)
#             dS += ret[0]
#             nmoves += ret[1]

#         print("Change in description length:", dS)
#         print("Number of accepted vertex moves:", nmoves)
#         with open(f"{output_prefix}_{s}.pkl", "wb") as f:
#             pickle.dump(state, f)
#         print("Saved!")


def dump_hsbm_tm(
    graph_input="graph.gt.gz",
    input_prefix="state",
    model_output="model.db",
    entropy_output="entropy.csv",
    seeds=[1000, 1001, 1002, 1003, 1004],
):
    print("Load Graph...")
    g = gt.load_graph(graph_input)
    print("Loaded!")

    print("Load States...")

    def load(s):
        return pickle.load(open(f"{input_prefix}_{s}.pkl", "rb"))

    with Pool(4) as p:
        states = p.map(load, seeds)
    print("Loaded!")

    S = []
    for i, s in enumerate(seeds):
        S.append((s, None, states[i].entropy()))
        for j, l in enumerate(states[i].get_levels()):
            S.append((s, j, l.entropy()))
    pandas.DataFrame(S, columns=["seed", "level", "entropy"]).to_csv(
        entropy_output, index=False
    )

    def get_groups(state, seed, l):
        level = state.levels[l]
        b = gt.contiguous_map(level.b)
        label_map = {}
        for v in g.vertices():
            label_map[level.b[v]] = b[v]
        return pandas.DataFrame(
            {
                "seed": seed,
                "level": l,
                "id": g.vp["id"].a,
                "kind": g.vp["kind"].a,
                "group": [label_map[b] for b in state.project_level(l).get_blocks()],
            }
        )

    df = pandas.concat(
        [
            get_groups(states[i], s, l)
            for i, s in enumerate(seeds)
            for l, _ in enumerate(states[i].get_levels())
        ]
    )
    df.to_sql("model", f"sqlite:///{model_output}", index=False)


def infer_uh_model(
    input="sparse.npz",
    output_prefix="uh",
    n_neighbors=15,
    min_dist=0.1,
    n_components=2,
    metric="cosine",
    min_cluster_size=5,
    min_samples=1,
    cluster_selection_method="eom",
    seeds=[1000, 1001, 1002, 1003, 1004],
):
    for s in seeds:
        m = scipy.sparse.load_npz(input)
        u = umap.UMAP(
            random_state=s,
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            n_components=n_components,
            metric=metric,
        )
        e = u.fit_transform(m)
        c = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            cluster_selection_method=cluster_selection_method,
        )
        l = c.fit_predict(e)
        np.savez_compressed(f"uh/{output_prefix}_{s}.npz", e=e, l=l)
        return u, e, c, l
