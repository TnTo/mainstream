# %%
import glob
import json
import sqlite3
from tqdm.auto import tqdm

# %%
paths = glob.glob("data/*.jsonl")

# %%
db = sqlite3.connect("raw.db")
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


# %%
with tqdm(total=(len(paths) - 1) * 25000) as pbar:
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
                            "INSERT OR IGNORE INTO vocabulary(word) VALUES (?);", (w,)
                        )
                        cur.execute("SELECT id FROM vocabulary WHERE word = ?;", (w,))
                        word_id = cur.fetchone()[0]
                        cur.execute(
                            "INSERT INTO graph(document_id, word_id, count) VALUES (?,?,?);",
                            (doc_id, word_id, c),
                        )
                pbar.update()
# %%
db.commit()
db.close()
