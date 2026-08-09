"""Tests for Tavily domain-restricted search."""

import asyncio

from open_deep_research.configuration import Configuration
from open_deep_research.utils import tavily_search_async


class FakeTavilyClient:
    """Capture Tavily search options without making network requests."""

    calls = []

    def __init__(self, api_key):
        self.api_key = api_key

    async def search(self, query, **options):
        """Record a search call and return an empty result."""
        self.calls.append((query, options))
        return {"query": query, "results": []}


def test_include_domains_loads_from_environment(monkeypatch):
    """Load a comma-separated domain whitelist from the environment."""
    monkeypatch.setenv("INCLUDE_DOMAINS", "gov.cn,who.int")

    configuration = Configuration.from_runnable_config()

    assert configuration.include_domains == "gov.cn,who.int"


def test_tavily_search_forwards_include_domains(monkeypatch):
    """Pass the configured whitelist to every Tavily search request."""
    FakeTavilyClient.calls = []
    monkeypatch.setenv("TAVILY_API_KEY", "test-key")
    monkeypatch.setattr(
        "open_deep_research.utils.AsyncTavilyClient",
        FakeTavilyClient,
    )

    asyncio.run(
        tavily_search_async(
            ["first query", "second query"],
            include_domains=["gov.cn", "who.int"],
        )
    )

    assert len(FakeTavilyClient.calls) == 2
    assert all(
        options["include_domains"] == ["gov.cn", "who.int"]
        for _, options in FakeTavilyClient.calls
    )
