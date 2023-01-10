# %%
import json
import pandas
import requests

# %%
def get_doi(doi):
    r = requests.get(
        f"https://api.semanticscholar.org/graph/v1/paper/{doi}?fields=citationCount"
    )
    match r.status_code:
        case 200:
            return json.loads(r.text)["citationCount"]
        case 404:
            return None
        case _:
            return f"ERROR {r.status_code}"


# %%
df = pandas.read_sql("SELECT doi FROM document", "sqlite:///raw.db")
df["citations"] = df.apply(lambda row: get_doi(row.doi), axis=1)
df.to_sql("citations", "sqlite:///citations.db", index=False)
