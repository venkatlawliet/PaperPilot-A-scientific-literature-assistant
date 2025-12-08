#semantic scholar results paper url extraction
from __future__ import annotations
import os, requests
from typing import Optional, Dict, Any
UNPAYWALL_EMAIL = os.getenv("UNPAYWALL_EMAIL") #Here i used my mail
USE_UNPAYWALL = bool(UNPAYWALL_EMAIL and "@" in UNPAYWALL_EMAIL)
def resolve_pdf_url_from_s2_item(s2_item: Dict[str, Any]) -> Optional[str]:
    oa = (s2_item.get("openAccessPdf") or {}).get("url")
    if oa:
        return oa
    links = s2_item.get("paperLinks") or []
    for link in links:
        url = link.get("url", "")
        if "pdf" in url.lower():
            return url
    ext = s2_item.get("externalIds") or {}
    arxiv = ext.get("ArXiv") or ext.get("ARXIV") or ext.get("arXiv")
    if arxiv:
        return f"https://arxiv.org/pdf/{arxiv}.pdf"
    doi = ext.get("DOI") or ext.get("doi")
    if not doi:
        return None
    if USE_UNPAYWALL:
        try:
            r = requests.get(
                f"https://api.unpaywall.org/v2/{doi}",
                params={"email": UNPAYWALL_EMAIL}, 
                timeout=20,
            )
            if not r.ok:
                return None
            data = r.json()
            best = data.get("best_oa_location") or {}
            pdf = best.get("url_for_pdf")
            if pdf:
                return pdf
            for loc in data.get("oa_locations", []):
                if loc.get("url_for_pdf"):
                    return loc["url_for_pdf"]
        except Exception:
            return None
    return None