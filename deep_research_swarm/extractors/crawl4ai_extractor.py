"""Crawl4AI extractor — handles JS-rendered pages with BM25 filtering."""

from __future__ import annotations


async def extract_with_crawl4ai(url: str) -> str:
    """Extract content from URL using Crawl4AI.

    Returns cleaned markdown content, or empty string on failure.
    """
    try:
        from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig
        from crawl4ai.content_filter_strategy import BM25ContentFilter

        browser_config = BrowserConfig(headless=True)
        content_filter = BM25ContentFilter(user_query=None)
        run_config = CrawlerRunConfig(
            content_filter=content_filter,
            word_count_threshold=50,
        )

        async with AsyncWebCrawler(config=browser_config) as crawler:
            result = await crawler.arun(url=url, config=run_config)
            if result.success and result.markdown_v2:
                content = result.markdown_v2.fit_markdown or result.markdown_v2.raw_markdown
                if content and len(content.strip()) > 100:
                    return content.strip()

        return ""
    except Exception:
        return ""
