"""Tests for the shared webpage summarization concurrency limit."""

import asyncio

from open_deep_research.configuration import Configuration
from open_deep_research import utils


def test_summarization_limit_loads_from_environment(monkeypatch):
    """Load the summarization concurrency limit from the environment."""
    monkeypatch.setenv("MAX_CONCURRENT_SUMMARIZATIONS", "3")

    configuration = Configuration.from_runnable_config()

    assert configuration.max_concurrent_summarizations == 3


def test_summarization_limit_is_shared_across_batches(monkeypatch):
    """Never exceed the shared limit when multiple search calls overlap."""
    active = 0
    peak = 0

    async def fake_summarize(model, webpage_content):
        nonlocal active, peak
        active += 1
        peak = max(peak, active)
        try:
            await asyncio.sleep(0.01)
            return webpage_content
        finally:
            active -= 1

    async def run_batches():
        monkeypatch.setattr(utils, "summarize_webpage", fake_summarize)
        first_batch = [
            utils.summarize_webpage_with_limit(None, f"first-{index}", 3)
            for index in range(4)
        ]
        second_batch = [
            utils.summarize_webpage_with_limit(None, f"second-{index}", 3)
            for index in range(4)
        ]
        await asyncio.gather(*first_batch, *second_batch)

    asyncio.run(run_batches())

    assert peak == 3
