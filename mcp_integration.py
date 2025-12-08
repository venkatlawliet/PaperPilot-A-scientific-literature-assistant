import os
import requests
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
SERPAPI_API_KEY = os.environ.get("SERPAPI_API_KEY", "")
SERPAPI_ENDPOINT = "https://serpapi.com/search"
REQUEST_TIMEOUT = int(os.environ.get("SERPAPI_TIMEOUT", "10"))
@dataclass
class SearchResult:
    title: str
    url: str
    description: str
class WebSearchClient:
    def __init__(
        self,
        api_key: str = SERPAPI_API_KEY,
        endpoint: str = SERPAPI_ENDPOINT,
        timeout: int = REQUEST_TIMEOUT,
    ):
        self.api_key = api_key
        self.endpoint = endpoint
        self.timeout = timeout
    def search(self, query: str, count: int = 5) -> List[SearchResult]:
        if not self.api_key:
            print("[SERPAPI] ERROR: SERPAPI_API_KEY not set")
            return []
        if not query or not query.strip():
            return []
        params = {
            "q": query.strip(),
            "api_key": self.api_key,
            "engine": "google",
            "num": min(count + 2, 10), 
            "hl": "en",
            "gl": "us",
        }
        try:
            response = requests.get(
                self.endpoint,
                params=params,
                timeout=self.timeout,
            )
            response.raise_for_status()
            data = response.json()
            if "error" in data:
                print(f"[SERPAPI] API error: {data['error']}")
                return []
            results: List[SearchResult] = []
            answer_box = data.get("answer_box")
            if answer_box:
                answer_result = self._extract_answer_box(answer_box, data)
                if answer_result:
                    results.append(answer_result)
            organic_results = data.get("organic_results", [])
            for item in organic_results:
                if len(results) >= count:
                    break
                result = self._extract_organic_result(item)
                if result:
                    results.append(result)
            if not results:
                related = data.get("related_questions", [])
                for item in related[:2]:
                    snippet = item.get("snippet", "")
                    if snippet:
                        results.append(SearchResult(
                            title=item.get("question", "Related Answer"),
                            url=item.get("link", ""),
                            description=snippet,
                        ))
            return results[:count]
        except requests.Timeout:
            print("[SERPAPI] Request timed out")
            return []
        except requests.RequestException as e:
            print(f"[SERPAPI] Request error: {e}")
            return []
        except Exception as e:
            print(f"[SERPAPI] Unexpected error: {type(e).__name__}: {e}")
            return []
    def _extract_answer_box(
        self,
        answer_box: Dict[str, Any],
        full_data: Dict[str, Any],
    ) -> Optional[SearchResult]:
        box_type = answer_box.get("type", "")
        google_url = full_data.get("search_metadata", {}).get("google_url", "")
        if box_type == "currency_converter":
            return self._extract_currency_answer(answer_box, google_url)
        answer_text = (
            answer_box.get("answer")
            or answer_box.get("snippet")
            or answer_box.get("result")
        )
        if answer_text:
            title = answer_box.get("title", "Direct Answer")
            return SearchResult(
                title=title,
                url=google_url,
                description=str(answer_text),
            )
        highlighted = answer_box.get("snippet_highlighted_words")
        if highlighted and isinstance(highlighted, list):
            return SearchResult(
                title=answer_box.get("title", "Answer"),
                url=answer_box.get("link", google_url),
                description=" ".join(highlighted),
            )
        list_items = answer_box.get("list")
        if list_items and isinstance(list_items, list):
            description = "; ".join(str(item) for item in list_items[:5])
            return SearchResult(
                title=answer_box.get("title", "Answer"),
                url=google_url,
                description=description,
            )
        return None
    def _extract_currency_answer(
        self,
        answer_box: Dict[str, Any],
        google_url: str,
    ) -> SearchResult:
        converter = answer_box.get("currency_converter", {})
        from_data = converter.get("from", {})
        to_data = converter.get("to", {})
        if from_data and to_data:
            from_price = from_data.get("price", 1)
            from_currency = from_data.get("currency", "USD")
            to_price = to_data.get("price", "N/A")
            to_currency = to_data.get("currency", "")
            description = f"{from_price} {from_currency} = {to_price} {to_currency}"
        else:
            price = answer_box.get("price", "N/A")
            currency = answer_box.get("currency", "")
            result = answer_box.get("result", "")
            if result:
                description = result
            elif price and currency:
                description = f"1 USD = {price} {currency}"
            else:
                description = str(answer_box.get("result", "Currency data unavailable"))
        date = answer_box.get("date")
        if date:
            description += f" (as of {date})"
        return SearchResult(
            title="Currency Exchange Rate",
            url=google_url,
            description=description,
        )
    def _extract_organic_result(
        self,
        item: Dict[str, Any],
    ) -> Optional[SearchResult]:
        title = item.get("title", "").strip()
        url = item.get("link", "").strip()
        snippet = item.get("snippet", "").strip()
        if not title or not url:
            return None
        return SearchResult(
            title=title,
            url=url,
            description=snippet or title,
        )
def handle_tool_call_from_claude(
    tool_name: str,
    tool_params: Dict[str, Any],
) -> Dict[str, Any]:
    if tool_name == "web_search":
        query = (tool_params.get("query") or "").strip()
        if not query:
            return {"error": "No query provided for web_search."}
        results = WebSearchClient().search(query, count=5)
        return {"results": [asdict(r) for r in results]}
    return {"error": f"Unsupported tool: {tool_name}"}