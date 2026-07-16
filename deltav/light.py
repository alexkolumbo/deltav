"""Light client: verify the chain and your charges without trusting a node.

Three independent checks, from weakest trust assumption to strongest:

1. Header-chain integrity — starting from a trusted genesis, every header
   links by prev_hash, is signed by its stated proposer, and carries a
   valid RANDAO reveal. Pure cryptography, no state, no full blocks.
   (What it CANNOT prove alone: that the proposer was the *legitimate*
   stake-weighted choice — that needs the validator set, i.e. state, i.e.
   a committee. Documented, deferred to a future finality gadget.)

2. Quorum agreement — query K independent nodes; only trust a head hash /
   balance a majority agree on. A single lying gateway or node is caught.

3. Payment authorization — an INFERENCE_RECEIPT in a block carries the
   requester's signature over (request_hash, node, model, price_limit).
   A consumer verifies locally that every charge against THEIR key was
   actually authorized by their key — a gateway cannot fabricate charges.

`deltav verify` drives all three.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

import httpx

from .chain.block import Block, GENESIS_PROPOSER, ZERO_HASH, tx_root
from .chain.blockchain import build_genesis_block
from .chain.transaction import Tx, TxType, receipt_auth_bytes
from .config import Genesis
from .crypto import address_from_public, canonical_json, sha256_hex, verify_signature


@dataclass
class HeaderVerdict:
    ok: bool
    height: int
    checked: int
    error: str = ""


@dataclass
class Charge:
    receipt_hash: str
    height: int
    node: str
    model: str
    price_limit: int
    tokens: int
    authorized: bool  # requester signature verifies against the payer's key
    duplicate: bool = False  # a later charge reusing one (request_hash, node) auth


@dataclass
class AuditResult:
    address: str
    charges: list[Charge] = field(default_factory=list)

    @property
    def all_authorized(self) -> bool:
        # A duplicated authorization is NOT a clean charge, even if its
        # signature verifies — one auth should settle exactly once.
        return all(c.authorized and not c.duplicate for c in self.charges)

    @property
    def duplicate_charges(self) -> list[Charge]:
        return [c for c in self.charges if c.duplicate]

    @property
    def total_price_limit(self) -> int:
        return sum(c.price_limit for c in self.charges)


def _header_hash(h: dict) -> str:
    """Recompute a block hash from header fields. Uses tx_root directly
    (header-only dicts carry it) or derives it from txs (full blocks) — so
    verification works whether we fetched /chain/headers or /chain/blocks."""
    root = h.get("tx_root")
    if root is None:
        root = tx_root([Tx.from_dict(t) for t in h.get("txs", [])])
    header = {
        "height": int(h["height"]),
        "prev_hash": h["prev_hash"],
        "timestamp": float(h["timestamp"]),
        "proposer": h["proposer"],
        "slot": int(h.get("slot", 0)),
        "randao_reveal": h.get("randao_reveal", ""),
        "tx_root": root,
        "state_root": h.get("state_root", ""),
    }
    return sha256_hex(canonical_json(header))


def verify_header_chain(genesis: Genesis, headers: list[dict]) -> HeaderVerdict:
    """Check hash links, proposer signatures and RANDAO reveals from a
    trusted genesis. `headers` may be full block dicts or header-only dicts."""
    expected_genesis = build_genesis_block(genesis)
    if not headers:
        return HeaderVerdict(False, 0, 0, "no headers")
    if _header_hash(headers[0]) != expected_genesis.hash:
        return HeaderVerdict(False, 0, 0, "genesis hash mismatch — wrong chain")

    prev_hash = expected_genesis.hash
    prev_height = 0
    chain_id = genesis.params.chain_id
    checked = 1
    for h in headers[1:]:
        try:
            height = int(h["height"])
            block_hash = _header_hash(h)
        except (KeyError, ValueError) as exc:
            return HeaderVerdict(False, prev_height, checked, f"malformed header: {exc}")
        if height != prev_height + 1:
            return HeaderVerdict(False, prev_height, checked, f"height gap at {height}")
        if h["prev_hash"] != prev_hash:
            return HeaderVerdict(False, height, checked,
                                 f"prev_hash break at height {height}")
        if h.get("hash") and h["hash"] != block_hash:
            return HeaderVerdict(False, height, checked,
                                 f"header hash mismatch at {height}")
        proposer = h["proposer"]
        pubkey, sig = h.get("pubkey", ""), h.get("signature", "")
        if proposer == GENESIS_PROPOSER:
            if height != 0:
                return HeaderVerdict(False, height, checked, f"genesis proposer at {height}")
        else:
            if address_from_public(pubkey) != proposer or not verify_signature(
                pubkey, bytes.fromhex(block_hash), sig
            ):
                return HeaderVerdict(False, height, checked,
                                     f"bad proposer signature at {height}")
            if not verify_signature(pubkey, f"randao:{chain_id}:{height}".encode(),
                                    h.get("randao_reveal", "")):
                return HeaderVerdict(False, height, checked,
                                     f"bad randao reveal at {height}")
        prev_hash = block_hash
        prev_height = height
        checked += 1
    return HeaderVerdict(True, prev_height, checked)


def verify_charges(blocks: list[dict], payer_address: str,
                   payer_pubkey: str | None = None) -> AuditResult:
    """Extract every INFERENCE_RECEIPT that charged `payer_address` and
    check the requester's payment-authorization signature."""
    result = AuditResult(address=payer_address)
    # One signed authorization (request_hash + node) should settle exactly
    # once; a node reusing it with a different output_hash is a replayed drain.
    seen_auth: set[tuple[str, str]] = set()
    for bdict in blocks:
        for tdict in bdict.get("txs", []):
            if tdict.get("type") != TxType.INFERENCE_RECEIPT.value:
                continue
            p = tdict.get("payload", {})
            if p.get("requester") != payer_address:
                continue
            pubkey = payer_pubkey or p.get("requester_pubkey", "")
            ok = False
            if pubkey and address_from_public(pubkey) == payer_address:
                auth = receipt_auth_bytes(
                    p["request_hash"], tdict["sender"], p["model"], int(p["price_limit"]))
                ok = verify_signature(pubkey, auth, p.get("requester_sig", ""))
            auth_key = (p["request_hash"], tdict["sender"])
            is_dup = auth_key in seen_auth
            seen_auth.add(auth_key)
            receipt_hash = sha256_hex(canonical_json(
                {"request_hash": p["request_hash"], "node": tdict["sender"],
                 "output_hash": p["output_hash"]}))
            result.charges.append(Charge(
                receipt_hash=receipt_hash,
                height=int(bdict.get("height", 0)),
                node=tdict["sender"],
                model=p.get("model", ""),
                price_limit=int(p.get("price_limit", 0)),
                tokens=int(p.get("tokens_in", 0)) + int(p.get("tokens_out", 0)),
                authorized=ok,
                duplicate=is_dup,
            ))
    return result


class LightClient:
    def __init__(self, genesis: Genesis, node_urls: list[str],
                 client: httpx.AsyncClient | None = None):
        self.genesis = genesis
        self.node_urls = node_urls
        self.client = client or httpx.AsyncClient()
        self._owns = client is None

    async def close(self) -> None:
        if self._owns:
            await self.client.aclose()

    async def _get(self, url: str, path: str, **params):
        resp = await self.client.get(f"{url}{path}", params=params, timeout=15.0)
        resp.raise_for_status()
        return resp.json()

    async def quorum_head(self) -> tuple[str, int, int]:
        """Ask every node for its head; return the majority (hash, height,
        votes). Detects a node reporting a forged head."""
        votes: Counter = Counter()
        heights: dict[str, int] = {}
        for url in self.node_urls:
            try:
                head = await self._get(url, "/chain/head")
                votes[head["hash"]] += 1
                heights[head["hash"]] = int(head["height"])
            except (httpx.HTTPError, KeyError):
                continue
        if not votes:
            raise RuntimeError("no node reachable")
        top_hash, count = votes.most_common(1)[0]
        return top_hash, heights[top_hash], count

    async def verify_headers(self) -> HeaderVerdict:
        for url in self.node_urls:
            try:
                data = await self._get(url, "/chain/headers", start=0, count=100_000)
            except httpx.HTTPError:
                continue
            return verify_header_chain(self.genesis, data.get("headers", []))
        raise RuntimeError("no node served headers")

    async def quorum_balance(self, address: str) -> tuple[int, int]:
        """Majority-agreed balance and the vote count."""
        votes: Counter = Counter()
        for url in self.node_urls:
            try:
                acc = await self._get(url, f"/chain/account/{address}")
                votes[int(acc["balance"])] += 1
            except (httpx.HTTPError, KeyError):
                continue
        if not votes:
            raise RuntimeError("no node reachable")
        balance, count = votes.most_common(1)[0]
        return balance, count

    async def audit_charges(self, address: str,
                            pubkey: str | None = None) -> AuditResult:
        for url in self.node_urls:
            try:
                data = await self._get(url, "/chain/blocks", start=0, count=100_000)
            except httpx.HTTPError:
                continue
            return verify_charges(data.get("blocks", []), address, pubkey)
        raise RuntimeError("no node served blocks")
