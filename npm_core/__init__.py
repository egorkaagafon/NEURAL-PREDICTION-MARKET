"""Neural Prediction Market — core package."""

from npm_core.agents import AgentHead, AgentPool
from npm_core.market import MarketAggregator
from npm_core.capital import CapitalManager

__all__ = [
    "AgentHead",
    "AgentPool",
    "MarketAggregator",
    "CapitalManager",
]
