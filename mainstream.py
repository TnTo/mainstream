import glob
import json
import os
import pickle
import re
import sqlite3
from collections import defaultdict

import graph_tool.all as gt
import hdbscan
import nltk
import numpy as np
import pandas
import scipy.sparse
import umap

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
    return re.sub(r"[^a-zA-Z\-]", "", s).lower()


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
    min_doc_len=1000,
    min_word_len=3,
    min_word_freq=10,
    min_doc_occurencies=None,
    max_doc_occurencies=None,
):
    d = pandas.read_sql("SELECT * FROM doc_len;", f"sqlite:///{input}")
    w = pandas.read_sql("SELECT * FROM word_count;", f"sqlite:///{input}")
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
    w[["word_id", "word"]].reset_index(drop=True).to_sql(
        "vocabulary", f"sqlite:///{output}", index=True, index_label="id"
    )

    d[d.len >= min_doc_len].reset_index(drop=True)[["document_id"]].merge(
        dmeta, how="inner", left_on="document_id", right_on="id"
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
    overlap=False,
    verbose=False,
    seeds=[1000, 1001, 1002, 1003, 1004],
):
    print("Load...")
    g = gt.load_graph(input)
    print("Loaded!")

    label = g.vp["kind"]

    state_args = {"clabel": label}
    if not overlap:
        state_args["pclabel"] = label
        state_args["eweight"] = g.ep.count

    for s in seeds:
        np.random.seed(s)
        gt.seed_rng(s)

        print(f"Seed {s}")
        base_type = gt.OverlapBlockState if overlap else gt.BlockState
        state = gt.minimize_nested_blockmodel_dl(
            g,
            state_args=dict(base_type=base_type, **state_args),
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
            pickle.dump(state, f)
        print("Saved!")
        del state


def dump_level(g, V, D, state, root_results_dir, l, overlap):
    print(f"Level {l}")
    results_dir = os.path.join(root_results_dir, f"{l}")
    os.mkdir(results_dir)

    state_l = state.project_level(l).copy(overlap=True)
    b = gt.contiguous_map(state_l.b)
    label_map = {}
    for v in g.vertices():
        label_map[state_l.b[v]] = b[v]

    with open(os.path.join(results_dir, "label_map"), "wb") as f:
        pickle.dump(label_map, f)
    del label_map

    state_l = state_l.copy(b=b)

    state_l_edges = state_l.get_edge_blocks()  ## labeled half-edges

    ## count labeled half-edges, group-memberships
    B = state_l.get_nonempty_B()

    n_wb = np.zeros(
        (V, B)
    )  ## number of half-edges incident on word-node w and labeled as word-group tw
    n_db = np.zeros(
        (D, B)
    )  ## number of half-edges incident on document-node d and labeled as document-group td
    n_dbw = np.zeros(
        (D, B)
    )  ## number of half-edges incident on document-node d and labeled as word-group tw
    n_td_tw = np.zeros(
        (B, B)
    )  ## number of half-edges labeled as document-group td and as word-group tw

    if not overlap:
        eweight = g.ep["count"]
    else:
        eweight = g.new_ep("int", 1)

    ze = gt.ungroup_vector_property(state_l_edges, [0, 1])
    for v1, v2, z1, z2, w in g.get_edges([ze[0], ze[1], eweight]):
        n_db[v1, z1] += w
        n_dbw[v1, z2] += w
        n_wb[v2 - D, z2] += w
        n_td_tw[z1, z2] += w

    p_w = np.sum(n_wb, axis=1) / float(np.sum(n_wb))

    ind_d = np.where(np.sum(n_db, axis=0) > 0)[0]
    Bd = len(ind_d)
    n_db = n_db[:, ind_d]

    ind_w = np.where(np.sum(n_wb, axis=0) > 0)[0]
    Bw = len(ind_w)
    n_wb = n_wb[:, ind_w]

    ind_w2 = np.where(np.sum(n_dbw, axis=0) > 0)[0]
    n_dbw = n_dbw[:, ind_w2]

    n_td_tw = n_td_tw[:Bd, Bd:]

    ## group-membership distributions
    # group membership of each word-node P(t_w | w)
    p_tw_w = (n_wb / np.sum(n_wb, axis=1)[:, np.newaxis]).T

    # group membership of each doc-node P(t_d | d)
    p_td_d = (n_db / np.sum(n_db, axis=1)[:, np.newaxis]).T

    ## topic-distribution for words P(w | t_w)
    p_w_tw = n_wb / np.sum(n_wb, axis=0)[np.newaxis, :]

    ## Mixture of word-groups into documetns P(t_w | d)
    p_tw_d = (n_dbw / np.sum(n_dbw, axis=1)[:, np.newaxis]).T

    ## Group-Group matrix
    p_td_tw = n_td_tw / np.sum(n_td_tw)

    with open(os.path.join(results_dir, "Bd"), "w") as f:
        f.write(f"{Bd}")

    with open(os.path.join(results_dir, "Bw"), "w") as f:
        f.write(f"{Bw}")

    np.savez_compressed(
        os.path.join(results_dir, "Ps"),
        p_w=p_w,
        p_tw_w=p_tw_w,
        p_td_d=p_td_d,
        p_w_tw=p_w_tw,
        p_tw_d=p_tw_d,
        p_td_tw=p_td_tw,
    )


def dump_hsbm_tm(
    graph_input="graph.gt.gz",
    input_prefix="state",
    output_prefix="results",
    seeds=[1000, 1001, 1002, 1003, 1004],
    overlap=False,
):
    print("Load Graph...")
    g = gt.load_graph(graph_input)
    print("Loaded!")

    V = int(np.sum(g.vp["kind"].a == 1))
    D = int(np.sum(g.vp["kind"].a == 0))

    for s in seeds:
        print(f"Seed {s}")
        with open(f"{input_prefix}_{s}.pkl", "rb") as f:
            state = pickle.load(f)
        root_results_dir = f"{output_prefix}_{s}"
        os.mkdir(root_results_dir)

        with open(os.path.join(root_results_dir, "mdl"), "w") as f:
            f.write(f"{state.entropy()}")

        for l in range(len(state.levels)):
            dump_level(g, V, D, state, root_results_dir, l, overlap)


def infer_uh_model(
    input="sparse.npz",
    output_prefix="uh",
    n_neighbors=15,
    min_dist=0.1,
    n_components=2,
    metric="cosine",
    min_cluster_size=5,
    min_samples=1,
    cluster_selection_method='eom',
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
