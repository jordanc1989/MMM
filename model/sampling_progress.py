"""Thread-safe sampling progress for PyMC `pm.sample(..., callback=...)`.

PyMC's callback is invoked once per chain step during **native** NUTS
(`nuts_sampler="pymc"`). External samplers (e.g. nutpie/JAX) do not invoke this hook.

This app fits with **nutpie** for all runs: native PyMC NUTS can hit a PyTensor edge
case (`0**0` in geometric adstock when `adstock_alpha` touches 0) that JAX does not.
Refits therefore use an indeterminate progress bar instead of per-chain percentages.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Any, Literal

Phase = Literal["idle", "nuts", "ppc"]


@dataclass
class SamplingProgressTracker:
    """Updated from PyMC's sample callback; read from Dash poll callbacks."""

    _lock: threading.Lock = field(default_factory=threading.Lock)
    chains: int = 4
    tune: int = 0
    draws: int = 0
    phase: Phase = "idle"
    _per_chain: list[dict[str, Any]] = field(default_factory=list)
    nuts_indeterminate: bool = False

    def reset(self, *, chains: int, tune: int, draws: int, indeterminate: bool = False) -> None:
        """Call before `pm.sample` so totals match this run."""
        with self._lock:
            self.chains = max(1, int(chains))
            self.tune = int(tune)
            self.draws = int(draws)
            total = self.tune + self.draws
            self.phase = "nuts"
            self.nuts_indeterminate = bool(indeterminate)
            self._per_chain = [
                {
                    "current": 0,
                    "total": total,
                    "warmup": True,
                }
                for _ in range(self.chains)
            ]

    def set_phase(self, phase: Phase) -> None:
        with self._lock:
            self.phase = phase

    def pymc_callback(self, *, trace, draw) -> None:
        """Pass as `pm.sample(..., callback=tracker.pymc_callback)`."""
        chain = int(draw.chain)
        idx = int(draw.draw_idx)
        tuning = bool(draw.tuning)
        with self._lock:
            if not self._per_chain or chain < 0 or chain >= len(self._per_chain):
                return
            row = self._per_chain[chain]
            row["current"] = idx + 1
            row["warmup"] = tuning

    def snapshot(self) -> dict[str, Any]:
        """JSON-friendly dict for Dash `dcc.Store` / UI."""
        with self._lock:
            total = max(1, self.tune + self.draws)
            chains_out: list[dict[str, Any]] = []
            for i, row in enumerate(self._per_chain):
                cur = min(int(row["current"]), total)
                pct = 100.0 * cur / total
                chains_out.append(
                    {
                        "chain": i,
                        "current": cur,
                        "total": total,
                        "warmup": bool(row["warmup"]),
                        "pct": round(pct, 2),
                    }
                )
            overall = (
                sum(c["pct"] for c in chains_out) / max(len(chains_out), 1)
                if chains_out
                else 0.0
            )
            return {
                "phase": self.phase,
                "chains": chains_out,
                "overall_pct": round(overall, 2),
                "nuts_indeterminate": self.nuts_indeterminate,
            }
