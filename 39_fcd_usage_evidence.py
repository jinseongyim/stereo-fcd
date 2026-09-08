"""
Step 39: how widely is FCD actually used?

Reviewer 2 asked what the general implications are and "how many active users
ChemNet has". The manuscript asserts that FCD is a standard metric without
evidence. This collects citable evidence.

Sources, all public and unauthenticated:
  - OpenAlex           : citation counts, looked up by DOI where one exists
  - PyPI JSON          : release metadata for the reference implementation
  - pypistats.org      : recent download counts

Semantic Scholar was tried first and rate-limited immediately; OpenAlex is used
instead because it answers unauthenticated requests reliably. Every number is
printed with an access date.
"""
import json
import os
import sys
import time
import urllib.parse
import urllib.request
from datetime import date
from pathlib import Path

HERE = Path(__file__).parent
OUT = HERE / "fcd_usage_evidence.json"
MAILTO = os.environ.get("OPENALEX_MAILTO", "you@example.com")  # OpenAlex raises the
# rate limit for requests that carry a contact address; set it in your environment
UA = {"User-Agent": f"stereo-fcd-usage/1.0 (mailto:{MAILTO})"}

# (label, DOI or None, fallback search string)
PAPERS = [
    ("FCD (Preuer et al. 2018)", "10.1021/acs.jcim.8b00234", None),
    ("GuacaMol (Brown et al. 2019)", "10.1021/acs.jcim.8b00839", None),
    ("MOSES (Polykovskiy et al. 2020)", "10.3389/fphar.2020.565644", None),
    ("MolT5 (Edwards et al. 2022)", None, "Translation between Molecules and Natural Language"),
    ("BioT5 (Pei et al. 2023)", None,
     "BioT5 Enriching Cross-modal Integration in Biology with Chemical Knowledge"),
]

PKGS = ["fcd", "fcd-torch"]


def get(url, tries=4, base=1.5):
    last = None
    for i in range(tries):
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=40) as r:
                return json.loads(r.read().decode())
        except Exception as e:
            last = e
            time.sleep(base * (2 ** i))
    raise last


def one_paper(label, doi, query):
    try:
        if doi:
            w = get(f"https://api.openalex.org/works/doi:{doi}")
        else:
            d = get("https://api.openalex.org/works?search="
                    + urllib.parse.quote(query) + "&per-page=1")
            hits = d.get("results") or []
            if not hits:
                return {"label": label, "error": "no match"}
            w = hits[0]
    except Exception as e:
        return {"label": label, "error": str(e)}
    return {"label": label,
            "title": w.get("title"),
            "year": w.get("publication_year"),
            "venue": ((w.get("primary_location") or {}).get("source") or {}).get("display_name"),
            "citations": w.get("cited_by_count"),
            "doi": (w.get("doi") or "").replace("https://doi.org/", "") or doi,
            "openalex": w.get("id")}


def packages():
    rows = []
    for name in PKGS:
        rec = {"package": name}
        try:
            meta = get(f"https://pypi.org/pypi/{name}/json")
            info = meta["info"]
            rels = sorted(meta["releases"].keys())
            rec.update({"version": info.get("version"), "summary": info.get("summary"),
                        "n_releases": len(rels), "first_release": rels[0] if rels else None})
        except Exception as e:
            rec["meta_error"] = str(e)
        try:
            st = get(f"https://pypistats.org/api/packages/{name}/recent")
            rec["downloads"] = st["data"]
        except Exception as e:
            rec["stats_error"] = str(e)
        rows.append(rec)
        time.sleep(1.0)
    return rows


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    accessed = date.today().isoformat()
    print(f"accessed {accessed}\n")

    print("=" * 84)
    print("Citation counts (OpenAlex)")
    print("=" * 84)
    papers = []
    for label, doi, q in PAPERS:
        r = one_paper(label, doi, q)
        papers.append(r)
        print(f"\n{label}")
        if "error" in r:
            print(f"  FAILED: {r['error']}")
        else:
            print(f"  matched : {r['title']}")
            print(f"  venue   : {r['venue']} ({r['year']})")
            print(f"  cited by: {r['citations']}")
            print(f"  DOI     : {r['doi']}")
        time.sleep(0.6)

    print("\n" + "=" * 84)
    print("Reference implementation (PyPI)")
    print("=" * 84)
    pkgs = packages()
    for r in pkgs:
        print(f"\n{r['package']}")
        print(f"  latest version : {r.get('version')}   releases: {r.get('n_releases')}")
        print(f"  summary        : {r.get('summary')}")
        d = r.get("downloads")
        if d:
            print(f"  downloads      : last day {d['last_day']:,} | "
                  f"last week {d['last_week']:,} | last month {d['last_month']:,}")
        else:
            print(f"  downloads      : unavailable ({r.get('stats_error')})")

    OUT.write_text(json.dumps({"accessed": accessed, "papers": papers, "packages": pkgs},
                              indent=2))
    print("\n" + "=" * 84)
    print("SENTENCE MATERIAL")
    print("=" * 84)
    for r in papers:
        if "citations" in r:
            print(f"  {r['label']:34s} cited {r['citations']:>6,} times")
    for r in pkgs:
        if r.get("downloads"):
            print(f"  PyPI {r['package']:12s} {r['downloads']['last_month']:>6,} "
                  f"downloads in the last month")
    print(f"\n  all figures accessed {accessed}")
    print(f"wrote {OUT.name}")
