"""Internet search over privacy-friendly HTML frontends (no API keys).

Providers are tried in order; a provider that errors or returns nothing
just yields to the next one. Parsers are deliberately tolerant regexes —
these pages change, and a partial result beats an exception.
"""
from __future__ import annotations

import html
import re
import urllib.parse

import httpx

UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/126.0 Safari/537.36")

_TAG_RE = re.compile(r"<[^>]+>")

_DDG_ANCHOR_RE = re.compile(
    r'<a[^>]+class="[^"]*result__a[^"]*"[^>]+href="(?P<href>[^"]+)"[^>]*>(?P<title>.*?)</a>',
    re.S,
)
_DDG_SNIPPET_RE = re.compile(
    r'<a[^>]+class="[^"]*result__snippet[^"]*"[^>]*>(?P<snippet>.*?)</a>', re.S)
_MOJEEK_ANCHOR_RE = re.compile(
    r'<h2><a[^>]+href="(?P<href>https?://[^"]+)"[^>]*>(?P<title>.*?)</a></h2>', re.S)
_MOJEEK_SNIPPET_RE = re.compile(r'<p class="s">(?P<snippet>.*?)</p>', re.S)


def _clean(text: str) -> str:
    return html.unescape(_TAG_RE.sub("", text or "")).strip()


def _ddg_unwrap(href: str) -> str:
    """DDG's HTML frontend wraps targets as /l/?uddg=<urlencoded>."""
    if "uddg=" in href:
        qs = urllib.parse.parse_qs(urllib.parse.urlparse(href).query)
        if qs.get("uddg"):
            return qs["uddg"][0]
    return href


def _parse_results(page: str, anchor_re: re.Pattern, snippet_re: re.Pattern,
                   max_results: int, unwrap=lambda href: href) -> list[dict]:
    """Anchor matches delimit result blocks; the snippet (optional on real
    pages) is searched only inside its own block."""
    anchors = list(anchor_re.finditer(page))
    results = []
    for i, m in enumerate(anchors[:max_results]):
        block_end = anchors[i + 1].start() if i + 1 < len(anchors) else len(page)
        sm = snippet_re.search(page, m.end(), block_end)
        results.append({
            "title": _clean(m.group("title")),
            "url": unwrap(m.group("href")),
            "snippet": _clean(sm.group("snippet")) if sm else "",
        })
    return results


def parse_ddg(page: str, max_results: int) -> list[dict]:
    return _parse_results(page, _DDG_ANCHOR_RE, _DDG_SNIPPET_RE, max_results, _ddg_unwrap)


def parse_mojeek(page: str, max_results: int) -> list[dict]:
    return _parse_results(page, _MOJEEK_ANCHOR_RE, _MOJEEK_SNIPPET_RE, max_results)


class SearchEngine:
    def __init__(self, client: httpx.AsyncClient):
        self.client = client

    async def _ddg(self, query: str, max_results: int) -> list[dict]:
        resp = await self.client.get(
            "https://html.duckduckgo.com/html/",
            params={"q": query}, headers={"User-Agent": UA}, timeout=15.0,
        )
        resp.raise_for_status()
        return parse_ddg(resp.text, max_results)

    async def _mojeek(self, query: str, max_results: int) -> list[dict]:
        resp = await self.client.get(
            "https://www.mojeek.com/search",
            params={"q": query}, headers={"User-Agent": UA}, timeout=15.0,
        )
        resp.raise_for_status()
        return parse_mojeek(resp.text, max_results)

    async def search(self, query: str, max_results: int = 5) -> list[dict]:
        for provider in (self._ddg, self._mojeek):
            try:
                results = await provider(query, max_results)
            except httpx.HTTPError:
                continue
            if results:
                return results
        return []


def format_results(results: list[dict]) -> str:
    if not results:
        return "no results"
    return "\n".join(
        f"{i}. {r['title']}\n   {r['url']}\n   {r['snippet']}"
        for i, r in enumerate(results, 1)
    )
