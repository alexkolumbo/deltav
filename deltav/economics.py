"""Delta V tokenomics: cost-anchored pricing.

The anchor: a node operator must recover electricity and earn a 50%
service margin. Everything else derives from three numbers:

    joules/token = watts / tokens_per_sec
    kWh per 1M tokens = watts / (tps * 3.6)
    USD per 1M tokens = kWh * electricity_price * (1 + margin)

Because the network is decentralized, the electricity coefficient is the
WORLD AVERAGE household price — no single node's tariff dictates the
network price, but every operator can run `deltav price` with their own
watts/tps/tariff and set `--price` accordingly (the phase-7 market does
the rest: cheaper electricity -> lower asking price -> more traffic).

The DVT reference peg is chosen so the network's DEFAULT price
(10 udvt/token = 10 DVT per 1M) exactly covers the reference node
(150 W system, 30 tok/s) at world-average electricity + 50% margin:
$0.32 per 1M tokens -> 1 DVT ~= $0.032.
"""
from __future__ import annotations

from dataclasses import dataclass

# ~Global average household electricity price (USD/kWh), 2025 ballpark.
WORLD_AVG_ELECTRICITY_USD_PER_KWH = 0.155
SERVICE_MARGIN = 0.50
# Reference peg: what 1 DVT is worth in USD for pricing purposes.
REFERENCE_USD_PER_DVT = 0.032
# The reference node the peg was calibrated against.
REFERENCE_WATTS = 150.0
REFERENCE_TPS = 30.0


def kwh_per_million_tokens(watts: float, tokens_per_sec: float) -> float:
    """1M tokens * (watts / tps) joules/token -> kWh."""
    if tokens_per_sec <= 0:
        raise ValueError("tokens_per_sec must be positive")
    return watts / (tokens_per_sec * 3.6)


def cost_usd_per_million(watts: float, tokens_per_sec: float,
                         electricity_usd_kwh: float = WORLD_AVG_ELECTRICITY_USD_PER_KWH) -> float:
    return kwh_per_million_tokens(watts, tokens_per_sec) * electricity_usd_kwh


def price_usd_per_million(watts: float, tokens_per_sec: float,
                          electricity_usd_kwh: float = WORLD_AVG_ELECTRICITY_USD_PER_KWH,
                          margin: float = SERVICE_MARGIN) -> float:
    return cost_usd_per_million(watts, tokens_per_sec, electricity_usd_kwh) * (1 + margin)


def suggested_price_udvt(watts: float, tokens_per_sec: float,
                         electricity_usd_kwh: float = WORLD_AVG_ELECTRICITY_USD_PER_KWH,
                         margin: float = SERVICE_MARGIN,
                         usd_per_dvt: float = REFERENCE_USD_PER_DVT) -> int:
    """The `--price` (udvt per token) that realizes cost + margin.

    udvt/token == USD-per-1M / USD-per-DVT (the 10^6 factors cancel).
    """
    usd_million = price_usd_per_million(watts, tokens_per_sec, electricity_usd_kwh, margin)
    return max(1, round(usd_million / usd_per_dvt))


@dataclass
class PriceReport:
    watts: float
    tokens_per_sec: float
    electricity_usd_kwh: float
    margin: float
    usd_per_dvt: float
    kwh_per_million: float
    cost_usd_per_million: float
    price_usd_per_million: float
    suggested_price_udvt: int

    def to_dict(self) -> dict:
        return self.__dict__.copy()


def price_report(watts: float = REFERENCE_WATTS,
                 tokens_per_sec: float = REFERENCE_TPS,
                 electricity_usd_kwh: float = WORLD_AVG_ELECTRICITY_USD_PER_KWH,
                 margin: float = SERVICE_MARGIN,
                 usd_per_dvt: float = REFERENCE_USD_PER_DVT) -> PriceReport:
    return PriceReport(
        watts=watts,
        tokens_per_sec=tokens_per_sec,
        electricity_usd_kwh=electricity_usd_kwh,
        margin=margin,
        usd_per_dvt=usd_per_dvt,
        kwh_per_million=round(kwh_per_million_tokens(watts, tokens_per_sec), 4),
        cost_usd_per_million=round(cost_usd_per_million(watts, tokens_per_sec, electricity_usd_kwh), 4),
        price_usd_per_million=round(price_usd_per_million(watts, tokens_per_sec, electricity_usd_kwh, margin), 4),
        suggested_price_udvt=suggested_price_udvt(watts, tokens_per_sec, electricity_usd_kwh,
                                                  margin, usd_per_dvt),
    )
