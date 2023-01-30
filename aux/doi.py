# %%
import json
import os
import re
import time
import urllib.parse

import dotenv
import modin.pandas as pandas
import pybliometrics.scopus
import requests

dotenv.load_dotenv()

# %%
### SemanticScholar
# def get_doi(doi):
#     r = requests.get(
#         f"https://api.semanticscholar.org/graph/v1/paper/{doi}?fields=citationCount"
#     )
#     match r.status_code:
#         case 200:
#             return json.loads(r.text)["citationCount"]
#         case 404:
#             return None
#         case 429:
#             time.sleep(250)
#             return get_doi(doi)
#         case 403:
#             return get_doi(doi)
#         case _:
#             return f"ERROR {r.status_code}"


# df = pandas.read_sql(
#     "SELECT id, authors, title, journal, year, doi FROM document", "sqlite:///data.db"
# )
# df["citations"] = df.apply(lambda row: get_doi(row.doi), axis=1)
# df.to_csv("citations.csv", index=False)
# df[["id", "citations"]].set_index("id").to_sql("citations", "sqlite:///data.db")

# %%
## SCOPUS
key = os.environ["SCOPUS_API_KEY"]
journals = [
    "economic AND journal",
    "journal AND political",
    "review AND economic AND statistics",
    "review AND economic AND studies",
    "quarterly AND journal AND economics",
    "economica",
    "american AND economic AND review",
    "econometrica",
]
dates = [
    (1945, 1951),
    (1950, 1956),
    (1955, 1961),
    (1960, 1966),
    (1965, 1971),
    (1970, 1976),
    (1975, 1981),
    (1980, 1986),
    (1985, 1991),
    (1990, 1996),
    (1995, 1999),
    (1998, 2002),
    (2001, 2005),
    (2004, 2007),
    (2006, 2009),
    (2008, 2010),
    (2009, 2011),
    (2010, 2012),
    (2011, 2013),
    (2012, 2014),
    (2013, 2015),
    (2014, 2016),
    (2015, 2017),
]
responses = []


def retrive(query):
    try:
        return pybliometrics.scopus.ScopusSearch(query, subscriber=False)
    except requests.ReadTimeout:
        return retrive(query)


for j in journals:
    for s, e in dates:
        query = f"SRCTITLE({j}) AND (PUBYEAR > {s} AND PUBYEAR < {e})"
        # url = f"https://api.elsevier.com/content/search/scopus?&apiKey={key}&query={query}&date=1946-2016&count=25"
        # r = requests.get(url)
        # print(r.status_code)
        # responses += r.json()["search-results"]["entry"]
        p = retrive(query)
        responses += p.results or []
# %%
df = pandas.DataFrame(responses)
df = df[
    df.publicationName.isin(
        [
            "Economic Journal",
            "Economic Journal (Conference Papers)",
            "Journal of Political Economy",
            "The journal of political economy",
            "Review of Economic &amp; Statistics",
            "Review of Economic and Statistics",
            "Review of Economic Studies",
            "The Review of economic studies",
            "Quarterly Journal of Economics",
            "Economica",
            "ECONOMICA",
            "American Economic Review",
            "American Economic Review, Papers and Proceedings",
            "The American economic review",
            "Econometrica : journal of the Econometric Society",
            "Econometrica",
        ]
    )
]
df.to_csv("scopus.csv", index=False)

# %%
