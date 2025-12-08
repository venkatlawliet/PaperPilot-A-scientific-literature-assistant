from __future__ import annotations
import os
import time
import requests
from typing import List, Dict, Any
S2_BASE = "https://api.semanticscholar.org/graph/v1" #to search papers on semantic scholar
S2_API_KEY = os.getenv("S2_API_KEY")
_last_call_ts = None  
def _respect_rate():
    global _last_call_ts
    now = time.time()
    if _last_call_ts is None:
        _last_call_ts = now
        return
    wait = 1.05 - (now - _last_call_ts)
    if wait > 0:
        time.sleep(wait)
    _last_call_ts = time.time()
def _headers() -> Dict[str, str]:
    if not S2_API_KEY:
        raise RuntimeError("S2_API_KEY environment variable is missing!")
    return {
        "x-api-key": S2_API_KEY,
        "User-Agent": "researchmcp/1.0 (https://semanticscholar.org)"
    }
def search_papers(query: str, limit: int = 5) -> List[Dict[str, Any]]:
    if not query or not query.strip():
        return []
    url = f"{S2_BASE}/paper/search"
    fields = ",".join([
        "title",
        "year",
        "venue",
        "paperId",
        "url",
        "authors",
        "externalIds",
        "isOpenAccess",
        "openAccessPdf",
        "citationCount",
        "referenceCount"
    ])
    params = {
        "query": query.strip(),
        "limit": max(1, min(limit, 20)),
        "offset": 0,
        "fields": fields,
    }
    for attempt in range(2):
        try:
            _respect_rate()
            response = requests.get(
                url,
                headers=_headers(),
                params=params,
                timeout=20
            )
            response.raise_for_status()
            data = response.json().get("data", [])
            return data if isinstance(data, list) else []
        except Exception as e:
            if attempt == 1:
                print(f"[S2 ERROR] {e}")
                return []
            time.sleep(1)