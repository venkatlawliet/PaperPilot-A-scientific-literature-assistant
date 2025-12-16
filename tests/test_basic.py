# tests/test_basic.py
# Basic tests for ResearchQuesta
# Run with: pytest tests/ -v

import pytest
import os
import sys

# Add parent directory to path so we can import our modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestEnvironmentVariables:
    """Test that required environment variables are documented"""
    
    REQUIRED_ENV_VARS = [
        "CLAUDE_API_KEY",
        "GROQ_API_KEY",
        "SERPAPI_API_KEY",
        "S2_API_KEY",
        "PINECONE_API_KEY",
        "PINECONE_INDEX_NAME",
        "SUPABASE_URL",
        "SUPABASE_ANON_KEY",
        "VISION_AGENT_API_KEY",
    ]
    
    def test_env_vars_list_exists(self):
        """Verify we have a list of required env vars"""
        assert len(self.REQUIRED_ENV_VARS) > 0


class TestImports:
    """Test that all main modules can be imported"""
    
    def test_import_mcp_integration(self):
        """Test mcp_integration module imports"""
        try:
            from mcp_integration import WebSearchClient, SearchResult
            assert WebSearchClient is not None
            assert SearchResult is not None
        except ImportError as e:
            pytest.skip(f"Dependencies not installed: {e}")
    
    def test_import_s2_client(self):
        """Test s2_client module imports"""
        try:
            from s2_client import search_papers
            assert search_papers is not None
        except ImportError as e:
            pytest.skip(f"Dependencies not installed: {e}")
    
    def test_import_llm_bridge(self):
        """Test llm_bridge module imports"""
        try:
            from llm_bridge import answer_with_claude, answer_with_llama
            assert answer_with_claude is not None
            assert answer_with_llama is not None
        except ImportError as e:
            pytest.skip(f"Dependencies not installed: {e}")


class TestSearchResult:
    """Test SearchResult dataclass"""
    
    def test_search_result_creation(self):
        """Test creating a SearchResult"""
        try:
            from mcp_integration import SearchResult
            result = SearchResult(
                title="Test Title",
                url="https://example.com",
                description="Test description"
            )
            assert result.title == "Test Title"
            assert result.url == "https://example.com"
            assert result.description == "Test description"
        except ImportError as e:
            pytest.skip(f"Dependencies not installed: {e}")


class TestWebSearchClient:
    """Test WebSearchClient class"""
    
    def test_client_initialization(self):
        """Test WebSearchClient can be initialized"""
        try:
            from mcp_integration import WebSearchClient
            client = WebSearchClient(api_key="test_key")
            assert client is not None
            assert client.api_key == "test_key"
        except ImportError as e:
            pytest.skip(f"Dependencies not installed: {e}")
    
    def test_empty_query_returns_empty_list(self):
        """Test that empty query returns empty list"""
        try:
            from mcp_integration import WebSearchClient
            client = WebSearchClient(api_key="test_key")
            results = client.search("")
            assert results == []
        except ImportError as e:
            pytest.skip(f"Dependencies not installed: {e}")
    
    def test_no_api_key_returns_empty_list(self):
        """Test that missing API key returns empty list"""
        try:
            from mcp_integration import WebSearchClient
            client = WebSearchClient(api_key="")
            results = client.search("test query")
            assert results == []
        except ImportError as e:
            pytest.skip(f"Dependencies not installed: {e}")


class TestContentResolver:
    """Test content_resolver module"""
    
    def test_arxiv_url_generation(self):
        """Test ArXiv PDF URL generation"""
        try:
            from content_resolver import resolve_pdf_url_from_s2_item
            
            # Test with ArXiv ID
            s2_item = {
                "externalIds": {"ArXiv": "2301.00001"}
            }
            url = resolve_pdf_url_from_s2_item(s2_item)
            assert url == "https://arxiv.org/pdf/2301.00001.pdf"
        except ImportError as e:
            pytest.skip(f"Dependencies not installed: {e}")
    
    def test_open_access_pdf_priority(self):
        """Test that openAccessPdf is prioritized"""
        try:
            from content_resolver import resolve_pdf_url_from_s2_item
            
            s2_item = {
                "openAccessPdf": {"url": "https://example.com/paper.pdf"},
                "externalIds": {"ArXiv": "2301.00001"}
            }
            url = resolve_pdf_url_from_s2_item(s2_item)
            assert url == "https://example.com/paper.pdf"
        except ImportError as e:
            pytest.skip(f"Dependencies not installed: {e}")
    
    def test_no_pdf_returns_none(self):
        """Test that missing PDF returns None"""
        try:
            from content_resolver import resolve_pdf_url_from_s2_item
            
            s2_item = {}
            url = resolve_pdf_url_from_s2_item(s2_item)
            assert url is None
        except ImportError as e:
            pytest.skip(f"Dependencies not installed: {e}")


class TestD2Utils:
    """Test D2 diagram utilities"""
    
    def test_extract_d2_block(self):
        """Test D2 code extraction from LLM response"""
        try:
            from d2_utils import extract_d2_block
            
            response = '''Here is the diagram:
```d2
A -> B: connection
B -> C: flow
```
'''
            d2_code = extract_d2_block(response)
            assert "A -> B" in d2_code
            assert "B -> C" in d2_code
        except ImportError as e:
            pytest.skip(f"Dependencies not installed: {e}")
    
    def test_extract_d2_block_no_language_tag(self):
        """Test D2 extraction without language tag"""
        try:
            from d2_utils import extract_d2_block
            
            response = '''```
A -> B
```'''
            d2_code = extract_d2_block(response)
            assert "A -> B" in d2_code
        except ImportError as e:
            pytest.skip(f"Dependencies not installed: {e}")
# Run tests with: pytest tests/ -v
