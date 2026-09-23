"""Measured bid/ask spread per symbol and UTC hour - replaces the flat default.

The ``spread_bps = 2.0`` default of ``RealismConfig`` (decision 29) was never measured. It
is now: Measurement 6 (2026-09-23) read the Dukascopy tick files directly, 60 files per
hour per symbol. The flat default turned out to be 3.3x to 7.2x TOO HIGH on the median
hour - and, at 21:00 UTC, TOO LOW for GBPUSD and USDJPY.

These are MEAN spreads, deliberately. A backtest *sums* costs over many fills, and a sum
converges on the mean, not the median. At 21:00 UTC the distribution changes shape, not
just level: a tenth of GBPUSD ticks exceed 5.70 bps and the p99 reaches 12.42, against
0.82 and 1.08 at 13:00. Dropping the worst one percent barely moves the mean (2.397 ->
2.268), so the tail is genuine book widening at the daily rollover, not a handful of bad
quotes. The median would systematically understate cost, and understating cost is what
makes a bad strategy look good.

This is the FLOOR of execution cost, not the cost. It is spread only. Slippage is not
observable in quote data, so it is NOT included here and must still be supplied through
``RealismConfig.slippage_bps``.

Provenance: ``reports/custo_horario_20260923.json``, produced by
``tools/medicoes/medir_tabela_de_custo_horario.py``; laudo in
``docs/MEDICAO_6_TABELA_DE_CUSTO_HORARIO_20260923.md``. Sampling seed 20260923, drawn
across 2023-2026 WITHOUT stratifying by year: the table treats spread as stationary in
time, which was not tested. Weekday is likewise not separated, so the Friday and Sunday
rollovers sit mixed into the same 24 hours.

Pure data and pure functions, stdlib only, no I/O: the table is embedded so this module
stays as deterministic as the rest of the execution layer.
"""

from __future__ import annotations

from src.execution_realism import RealismConfig

#: Mean bid/ask spread in basis points of the mid, indexed [0..23] by UTC hour.
MEAN_SPREAD_BPS: dict[str, tuple[float, ...]] = {
    "EURUSD": (
        0.3394, 0.3202, 0.3220, 0.3258,   # h00-h03
        0.2860, 0.3954, 0.2995, 0.2751,   # h04-h07
        0.3567, 0.3082, 0.2993, 0.2946,   # h08-h11
        0.2750, 0.2548, 0.2727, 0.2636,   # h12-h15
        0.2700, 0.2830, 0.2773, 0.2844,   # h16-h19
        0.4269, 1.2738, 1.1528, 0.4875,   # h20-h23
    ),
    "GBPUSD": (
        0.7026, 0.7053, 0.6698, 0.6722,   # h00-h03
        0.7082, 0.6557, 0.6305, 0.6060,   # h04-h07
        0.5673, 0.5557, 0.5828, 0.5701,   # h08-h11
        0.5957, 0.5674, 0.5908, 0.5933,   # h12-h15
        0.5770, 0.6060, 0.6108, 0.6013,   # h16-h19
        0.7045, 2.5421, 1.7244, 0.8913,   # h20-h23
    ),
    "USDJPY": (
        0.4256, 0.4302, 0.4726, 0.4396,   # h00-h03
        0.5292, 0.5392, 0.4133, 0.3978,   # h04-h07
        0.4073, 0.4399, 0.4263, 0.3909,   # h08-h11
        0.4381, 0.3985, 0.3950, 0.3973,   # h12-h15
        0.3727, 0.3731, 0.3729, 0.3828,   # h16-h19
        0.5421, 2.0751, 1.2089, 0.6032,   # h20-h23
    ),
}

#: The flat default this table replaces (decision 29).
LEGACY_FLAT_SPREAD_BPS = 2.0


def spread_bps(symbol: str, hour_utc: int) -> float:
    """Measured mean spread in bps for this symbol at this UTC hour.

    Raises KeyError for an unmeasured symbol rather than silently falling back to the flat
    default - a wrong-but-plausible cost is worse than a loud failure."""
    try:
        table = MEAN_SPREAD_BPS[symbol]
    except KeyError:
        raise KeyError(
            "no measured spread for {!r}; measured symbols: {}".format(
                symbol, ", ".join(sorted(MEAN_SPREAD_BPS))
            )
        ) from None
    if isinstance(hour_utc, bool) or not isinstance(hour_utc, int):
        raise ValueError("hour_utc must be an int in 0..23, got {!r}".format(hour_utc))
    if not 0 <= hour_utc <= 23:
        raise ValueError("hour_utc must be an int in 0..23, got {!r}".format(hour_utc))
    return table[hour_utc]


def config_for(symbol: str, hour_utc: int, slippage_bps: float = 1.0) -> RealismConfig:
    """A RealismConfig carrying the measured spread for that symbol and hour.

    ``slippage_bps`` is NOT measured here and keeps its caller-supplied value: quote data
    cannot observe it."""
    return RealismConfig(
        spread_bps=spread_bps(symbol, hour_utc),
        slippage_bps=slippage_bps,
    )


def costliest_hour(symbol: str) -> int:
    """The UTC hour with the widest measured mean spread (the rollover, for every pair)."""
    table = MEAN_SPREAD_BPS[symbol]
    return max(range(24), key=lambda h: table[h])
