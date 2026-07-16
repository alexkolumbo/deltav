"""Companion — a persistent, per-user, self-improving agent on the network.

Inspired by Hermes-style stacks (a durable agent with a memory layer),
but built on Delta V's own pieces: the ReAct agent, tools, and
network-embedded vector memory. Three properties define it:

  * strict per-user isolation — a user's identity comes from their key,
    not from the request body, so one user can never reach another's
    memory or sessions (this is the network's standing policy, enforced in
    code);
  * a personal memory layer — facts, preferences and learnings kept per
    user, recalled to shape future turns;
  * self-improvement — after each turn the agent reflects and stores what
    it learned; feedback becomes a durable, high-priority learning. Since
    the network targets small models, getting better via accumulated
    per-user memory is more reliable than retraining.
"""
from .identity import Identity, resolve_identity
from .memory import UserMemory
from .agent import CompanionAgent, CompanionResult, CompanionStep

__all__ = [
    "Identity",
    "resolve_identity",
    "UserMemory",
    "CompanionAgent",
    "CompanionResult",
    "CompanionStep",
]
