"""Deterministic state machine of the Delta V chain.

State = accounts (balance / nonce / stake) + node registry + inference
receipts awaiting spot checks. Every rule that moves value lives here.
"""
from __future__ import annotations

import copy
from dataclasses import asdict, dataclass, field

from ..config import ChainParams
from ..crypto import address_from_public, canonical_json, sha256_hex, verify_signature
from .transaction import Tx, TxType, receipt_auth_bytes


class StateError(Exception):
    """Raised when a transaction violates state rules."""


@dataclass
class Account:
    balance: int = 0
    nonce: int = 0
    stake: int = 0
    # Unstaked funds waiting out the unbonding period; still slashable.
    unbonding: list = field(default_factory=list)  # [{"amount", "release_height"}]
    misses: int = 0        # consecutive missed proposer slots
    jailed_until: int = 0  # height until which the validator is jailed


@dataclass
class NodeInfo:
    address: str
    endpoint: str
    hardware: dict = field(default_factory=dict)  # vendor / device / vram_mb / backend
    models: list[str] = field(default_factory=list)
    # The node's asking price in udvt per token; 0 = network default.
    # Receipts pay THIS price (capped by the requester's signed limit),
    # so nodes compete on price and the router prefers cheaper ones.
    price_per_token: int = 0
    reputation: float = 0.5
    jobs_done: int = 0
    tokens_served: int = 0
    registered_at: int = 0
    last_seen: int = 0
    active: bool = True


@dataclass
class Receipt:
    receipt_hash: str
    node: str
    requester: str
    model: str
    request_hash: str
    output_hash: str
    seed: int
    tokens_in: int
    tokens_out: int
    price_paid: int
    height: int
    # Whether the producing backend claims bit-reproducible output.
    # True -> spot checks compare output hashes exactly;
    # False -> checkers fall back to fuzzy verification (token counts).
    deterministic: bool = True
    checked: bool = False
    check_ok: bool | None = None


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


class State:
    def __init__(self, params: ChainParams):
        self.params = params
        self.height = 0
        self.supply = 0
        # RANDAO accumulator: mixes every block's randao_reveal; feeds
        # proposer selection so the schedule can't be ground in advance.
        self.randao = ""
        self.accounts: dict[str, Account] = {}
        self.nodes: dict[str, NodeInfo] = {}
        self.receipts: dict[str, Receipt] = {}

    # ------------------------------------------------------------- helpers
    def account(self, address: str) -> Account:
        if address not in self.accounts:
            self.accounts[address] = Account()
        return self.accounts[address]

    def clone(self) -> "State":
        return copy.deepcopy(self)

    def to_dict(self) -> dict:
        return {
            "height": self.height,
            "supply": self.supply,
            "randao": self.randao,
            "accounts": {a: asdict(acc) for a, acc in sorted(self.accounts.items())},
            "nodes": {a: asdict(n) for a, n in sorted(self.nodes.items())},
            "receipts": {h: asdict(r) for h, r in sorted(self.receipts.items())},
        }

    def state_root(self) -> str:
        return sha256_hex(canonical_json(self.to_dict()))

    def validators(self) -> list[tuple[str, int]]:
        """Accounts eligible to propose blocks and spot-check, sorted.

        Jailed validators are excluded until their jail height passes.
        """
        return sorted(
            (a, acc.stake)
            for a, acc in self.accounts.items()
            if acc.stake >= self.params.min_validator_stake and acc.jailed_until <= self.height
        )

    def unchecked_receipts(self) -> list[Receipt]:
        return sorted(
            (r for r in self.receipts.values() if not r.checked),
            key=lambda r: (r.height, r.receipt_hash),
        )

    # ---------------------------------------------------------- transitions
    def apply_tx(self, tx: Tx, height: int) -> None:
        if not tx.verify():
            raise StateError("invalid signature")
        sender = self.account(tx.sender)
        if tx.nonce != sender.nonce:
            raise StateError(f"bad nonce: expected {sender.nonce}, got {tx.nonce}")
        if tx.fee < 0:
            raise StateError("negative fee")
        if sender.balance < tx.fee:
            raise StateError("cannot afford fee")

        handler = {
            TxType.TRANSFER.value: self._apply_transfer,
            TxType.REGISTER_NODE.value: self._apply_register_node,
            TxType.ANNOUNCE_MODELS.value: self._apply_announce_models,
            TxType.STAKE.value: self._apply_stake,
            TxType.UNSTAKE.value: self._apply_unstake,
            TxType.INFERENCE_RECEIPT.value: self._apply_receipt,
            TxType.SPOT_CHECK.value: self._apply_spot_check,
        }.get(tx.type)
        if handler is None:
            raise StateError(f"unknown tx type {tx.type!r}")

        handler(tx, height)
        # Charged only after the handler succeeded — failed txs are not included.
        sender.balance -= tx.fee
        sender.nonce += 1

    def _apply_transfer(self, tx: Tx, height: int) -> None:
        to = tx.payload.get("to", "")
        amount = int(tx.payload.get("amount", 0))
        if amount <= 0 or not to:
            raise StateError("transfer needs positive amount and recipient")
        sender = self.account(tx.sender)
        if sender.balance < amount + tx.fee:
            raise StateError("insufficient balance")
        sender.balance -= amount
        self.account(to).balance += amount

    def _apply_register_node(self, tx: Tx, height: int) -> None:
        endpoint = tx.payload.get("endpoint", "")
        if not endpoint:
            raise StateError("register_node needs an endpoint")
        hardware = dict(tx.payload.get("hardware", {}))
        models = list(tx.payload.get("models", []))
        price = max(0, int(tx.payload.get("price_per_token", 0)))
        existing = self.nodes.get(tx.sender)
        if existing is not None:
            existing.endpoint = endpoint
            existing.hardware = hardware
            if models:
                existing.models = models
            existing.price_per_token = price
            existing.last_seen = height
            existing.active = True
        else:
            self.nodes[tx.sender] = NodeInfo(
                address=tx.sender,
                endpoint=endpoint,
                hardware=hardware,
                models=models,
                price_per_token=price,
                registered_at=height,
                last_seen=height,
            )

    def _apply_announce_models(self, tx: Tx, height: int) -> None:
        node = self.nodes.get(tx.sender)
        if node is None:
            raise StateError("node not registered")
        node.models = list(tx.payload.get("models", []))
        if "price_per_token" in tx.payload:
            node.price_per_token = max(0, int(tx.payload["price_per_token"]))
        node.last_seen = height
        node.active = bool(tx.payload.get("active", True))

    def _apply_stake(self, tx: Tx, height: int) -> None:
        amount = int(tx.payload.get("amount", 0))
        if amount <= 0:
            raise StateError("stake needs positive amount")
        sender = self.account(tx.sender)
        if sender.balance < amount + tx.fee:
            raise StateError("insufficient balance to stake")
        sender.balance -= amount
        sender.stake += amount

    def _apply_unstake(self, tx: Tx, height: int) -> None:
        amount = int(tx.payload.get("amount", 0))
        sender = self.account(tx.sender)
        if amount <= 0 or sender.stake < amount:
            raise StateError("invalid unstake amount")
        sender.stake -= amount
        sender.unbonding.append({
            "amount": amount,
            "release_height": height + self.params.unbonding_blocks,
        })

    def _apply_receipt(self, tx: Tx, height: int) -> None:
        """A node claims payment for one inference job, authorized by the requester."""
        p = tx.payload
        node = self.nodes.get(tx.sender)
        if node is None:
            raise StateError("receipt from unregistered node")

        required = ("requester", "requester_pubkey", "requester_sig", "request_hash",
                    "output_hash", "model", "seed", "tokens_in", "tokens_out", "price_limit")
        if any(k not in p for k in required):
            raise StateError("receipt payload incomplete")

        requester = p["requester"]
        if address_from_public(p["requester_pubkey"]) != requester:
            raise StateError("requester pubkey does not match address")
        auth = receipt_auth_bytes(p["request_hash"], tx.sender, p["model"], int(p["price_limit"]))
        if not verify_signature(p["requester_pubkey"], auth, p["requester_sig"]):
            raise StateError("bad requester authorization signature")

        deterministic = bool(p.get("deterministic", True))
        tokens_in, tokens_out = int(p["tokens_in"]), int(p["tokens_out"])
        if tokens_in < 0 or tokens_out <= 0:
            raise StateError("invalid token counts")
        effective_price = node.price_per_token or self.params.price_per_token
        price = (tokens_in + tokens_out) * effective_price
        if price > int(p["price_limit"]):
            raise StateError("price exceeds requester's authorized limit")

        receipt_hash = sha256_hex(canonical_json(
            {"request_hash": p["request_hash"], "node": tx.sender, "output_hash": p["output_hash"]}
        ))
        if receipt_hash in self.receipts:
            raise StateError("duplicate receipt")

        requester_acc = self.account(requester)
        if requester_acc.balance < price:
            raise StateError("requester cannot afford the job")
        requester_acc.balance -= price
        node_acc = self.account(tx.sender)
        node_acc.balance += price + self.params.inference_reward
        self.supply += self.params.inference_reward

        node.jobs_done += 1
        node.tokens_served += tokens_in + tokens_out
        node.last_seen = height
        node.reputation = _clamp01(node.reputation * 0.95 + 0.05)

        self.receipts[receipt_hash] = Receipt(
            receipt_hash=receipt_hash,
            node=tx.sender,
            requester=requester,
            model=p["model"],
            request_hash=p["request_hash"],
            output_hash=p["output_hash"],
            seed=int(p["seed"]),
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            price_paid=price,
            height=height,
            deterministic=deterministic,
        )

    def _apply_spot_check(self, tx: Tx, height: int) -> None:
        """A validator re-executed a receipt's job and reports the verdict."""
        checker = self.account(tx.sender)
        if checker.stake < self.params.min_validator_stake:
            raise StateError("spot check requires validator stake")
        receipt = self.receipts.get(tx.payload.get("receipt_hash", ""))
        if receipt is None:
            raise StateError("unknown receipt")
        if receipt.checked:
            raise StateError("receipt already checked")
        if receipt.node == tx.sender:
            raise StateError("node cannot check its own receipt")

        ok = bool(tx.payload.get("ok"))
        receipt.checked = True
        receipt.check_ok = ok
        node = self.nodes.get(receipt.node)
        node_acc = self.account(receipt.node)

        if ok:
            if node is not None:
                node.reputation = _clamp01(node.reputation * 0.9 + 0.1)
        else:
            self.slash(receipt.node, self.params.slash_fraction)
            if node is not None:
                node.reputation = _clamp01(node.reputation * 0.5)

        checker.balance += self.params.check_reward
        self.supply += self.params.check_reward

    def slash(self, address: str, fraction: float) -> int:
        """Burn a fraction of everything at stake — bonded AND unbonding,
        so unstaking right before getting caught doesn't dodge the penalty."""
        acc = self.account(address)
        total = acc.stake + sum(u["amount"] for u in acc.unbonding)
        remaining = int(total * fraction)
        burned = remaining
        take = min(acc.stake, remaining)
        acc.stake -= take
        remaining -= take
        for entry in acc.unbonding:
            if remaining <= 0:
                break
            take = min(entry["amount"], remaining)
            entry["amount"] -= take
            remaining -= take
        acc.unbonding = [u for u in acc.unbonding if u["amount"] > 0]
        self.supply -= burned
        return burned

    def apply_block_effects(self, proposer: str, fees: int, missed: list[str],
                            height: int, randao_reveal: str = "") -> None:
        """Per-block bookkeeping: mix RANDAO, release matured unbondings,
        punish proposers that missed their slot, reward the actual proposer."""
        self.randao = sha256_hex((self.randao + randao_reveal).encode())
        for address in sorted(self.accounts):
            acc = self.accounts[address]
            if not acc.unbonding:
                continue
            matured = sum(u["amount"] for u in acc.unbonding if u["release_height"] <= height)
            if matured:
                acc.balance += matured
                acc.unbonding = [u for u in acc.unbonding if u["release_height"] > height]

        for address in missed:
            acc = self.account(address)
            acc.misses += 1
            if acc.misses >= self.params.jail_after_misses:
                acc.misses = 0
                acc.jailed_until = height + self.params.jail_blocks

        proposer_acc = self.account(proposer)
        proposer_acc.misses = 0
        proposer_acc.balance += self.params.block_reward + fees
        self.supply += self.params.block_reward
