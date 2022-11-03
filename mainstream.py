import glob
import json
import re
import sqlite3
from collections import defaultdict

import graph_tool.all as gt
import numpy as np
import pandas


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

    # %%
    words = pandas.read_sql("vocabulary", f"sqlite:///{input}", index_col="id")
    words = words.assign(word_clean=words.word.apply(clean_str))
    vocabulary = pandas.Series(
        words.query("word_clean.str.len() > 1").word_clean.unique(),
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

    # %%
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
):
    d = pandas.read_sql("SELECT * FROM doc_len;", f"sqlite:///{input}")
    w = pandas.read_sql("SELECT * FROM word_count;", f"sqlite:///{input}")
    dmeta = pandas.read_sql("document", f"sqlite:///{input}")

    d[d.len >= min_doc_len].reset_index(drop=True)[["document_id"]].merge(
        dmeta, how="inner", left_on="document_id", right_on="id"
    ).drop(columns=["id"]).to_sql(
        "document", f"sqlite:///{output}", index=True, index_label="id"
    )

    w[(w.word.str.len() >= min_word_len) & (w.freq >= min_word_freq)][
        ["word_id", "word"]
    ].reset_index(drop=True).to_sql(
        "vocabulary", f"sqlite:///{output}", index=True, index_label="id"
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


def infer_tm(
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
