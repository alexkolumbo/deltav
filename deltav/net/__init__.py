"""External connectivity: make any node reachable by the whole network.

Three layers, tried in order so a node needs *zero* network config:

1. **Self-endpoint discovery** — a joining node learns its own public
   address from a peer (`GET /whoami`, a STUN-equivalent with no external
   service) and proves it is directly reachable by asking a peer to call
   it back (`POST /net/reachcheck`).
2. **Circuit relay** — if the callback fails (NAT/CGNAT, no port-forward),
   the node opens an *outbound* HTTP long-poll to any public node that
   advertises `relay: true`, and is handed a public URL
   `https://<relay>/via/<node-id>`. Other nodes reach it through the relay
   with no inbound port on the node's side.
3. **Manual** — an operator that set `--endpoint` is always trusted as-is.

Everything is signed (endpoint ownership is bound to the node's chain key)
and speaks only HTTP, so it traverses any proxy and needs no WebSocket.
"""
from __future__ import annotations

from .reach import ReachResult, check_direct, mount_reach, probe_public_ip
from .relay import RelayClient, RelayServer, discover_relay

__all__ = [
    "ReachResult",
    "check_direct",
    "mount_reach",
    "probe_public_ip",
    "RelayClient",
    "RelayServer",
    "discover_relay",
]
