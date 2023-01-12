# %%
import json
import time
import re
import urllib.parse

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from fake_useragent import UserAgent

import modin.pandas as pandas
import requests


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


df = pandas.read_sql(
    "SELECT id, authors, title, journal, year, doi FROM document", "sqlite:///data.db"
)
df["citations"] = df.apply(lambda row: get_doi(row.doi), axis=1)
df.to_csv("citations.csv", index=False)
# df[["id", "citations"]].set_index("id").to_sql("citations", "sqlite:///data.db")

# %%
### Google Scholar
def get_jstor(url):
    options = Options()
    # options.headless = True
    ua = UserAgent()
    options.add_argument("--disable-blink-features=AutomationControlled")
    options.add_argument(f"user-agent={ua.random}")
    options.add_argument("--profile-directory=Default")
    driver = webdriver.Chrome(options=options)
    driver.get(
        f"https://scholar.google.com/scholar?hl=en&q={urllib.parse.quote(url, safe='')}"
    )
    html = driver.page_source
    driver.quit()
    if re.findall("recaptcha", html):
        return "ERROR CAPTCHA"  # always a capthca
    else:
        cit = re.compile(r"Cited[\s\n]+by[\s\n]+([0-9]+)", re.MULTILINE).findall(html)
        match len(cit):
            case 0:
                return "ERROR NOT FOUND"
            case 1:
                return cit[0]
            case _:
                return "ERROR MULTIPLE MATCHES"


# %%
