# %%
import pandas as pandas
import re
from tqdm import tqdm
import sqlite3

tqdm.pandas()

# %%
journals = [
    "Econometrica",
    "The American Economic Review",
    "The Review of Economic Studies",
    "The Quarterly Journal of Economics",
    "The Review of Economics and Statistics",
    "Journal of Political Economy",
    "The Economic Journal",
    "Economica",
]


def clean_str(s: str) -> str:
    return re.sub(r"[^a-zA-Z\-]", "", s).lower()


# %%
docs = (
    pandas.read_sql_table(
        "document", "sqlite:///raw.db", index_col="id", parse_dates=["date"]
    )
    .query(
        '(authors != "null") \
    & (provider == "jstor") \
    & (subtype == "research-article") \
    & unigrams \
    & journal.isin(@journals) \
    & (year >= 1946) \
    & (year <= 2016)'
    )
    .reset_index()
    .rename(columns={"id": "old_id"})
)
docs.drop(columns="old_id").to_sql(
    "document", "sqlite:///clean.db", index=True, index_label="id"
)
docs[["old_id"]].copy().reset_index().rename(columns={"index": "id"}).set_index(
    "old_id"
).to_sql("docs", "sqlite:///merge.db", index=True, index_label="old_id")

del docs

# %%
words = pandas.read_sql("vocabulary", "sqlite:///raw.db", index_col="id")
words = words.assign(word_clean=words.word.progress_apply(clean_str))
vocabulary = pandas.Series(
    words.query("word_clean.str.len() > 1").word_clean.unique(),
    name="word",
)
vocabulary.to_sql(
    "vocabulary", "sqlite:///clean.db", index=True, index_label="id", chunksize=10000
)

words.reset_index().merge(
    vocabulary.reset_index(), how="right", left_on="word_clean", right_on="word"
)[["id", "index"]].rename(columns={"id": "old_id", "index": "id"}).set_index("old_id")[
    ["id"]
].to_sql(
    "words", "sqlite:///merge.db", index=True, index_label="old_id", chunksize=10000
)

del vocabulary
del words

# %%
with sqlite3.connect("clean.db") as db:
    cur = db.cursor()
    cur.executescript(
        """
ATTACH DATABASE "raw.db" AS RawDB;
ATTACH DATABASE "merge.db" AS MergeDB;

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
