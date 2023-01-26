# %%
import json
import time
import re
import urllib.parse

import dotenv

import modin.pandas as pandas
import requests
import os

dotenv.load_dotenv()

# %%
### SemanticScholar
def get_doi(doi):
    r = requests.get(
        f"https://api.semanticscholar.org/graph/v1/paper/{doi}?fields=citationCount"
    )
    match r.status_code:
        case 200:
            return json.loads(r.text)["citationCount"]
        case 404:
            return None
        case 429:
            time.sleep(250)
            return get_doi(doi)
        case 403:
            return get_doi(doi)
        case _:
            return f"ERROR {r.status_code}"


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
    "journal AND political AND economy",
    "review AND economic AND statistics",
    "review AND economic AND studies",
    "quarterly AND journal AND economics",
    "economica",
    "american AND economic AND review",
    "econometrica",
]
responses = []
for j in journals:
    query = urllib.parse.quote_plus(f"SRCTITLE({j})", safe="")
    url = f"https://api.elsevier.com/content/search/scopus?apiKey={key}&query={query}"
    r = requests.get(url)
    print(r.status_code)
    responses += r.json()["search-results"]["entry"]
pandas.DataFrame(responses).to_csv("scopus.csv", index=False)
# %%
