"""Per-step sample emitted by ``simulate_iter`` and consumed by sinks.

A ``Sample`` is one snapshot of the simulation at one MATLAB-1-based sample
``k``. It carries:

  - ``state``    — the 33 named ODE states (post per-sample conversions:
                   pH from [H+] to pH units, Q from kJ to "kcal-ish")
  - ``controls`` — the 12 manipulated variables produced by the controller
                   that were used to integrate from t[k-1] to t[k]
  - ``raman``    — optional 2200-bin spectrum, populated only when
                   ``StreamConfig.raman_every`` and ``k > 10`` allow
  - ``offline``  — optional dict of lab measurements with the configured
                   delay applied; populated only on the configured cadence

Pacers stamp ``wall_time_s`` (seconds since stream start). Unpaced
streams leave it as ``None``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass(frozen=True)
class Sample:
    k: int                             # 1-based MATLAB-style sample index
    sim_time_h: float                  # batch time in hours
    wall_time_s: Optional[float]       # set by paced(); None when unpaced
    state: dict[str, float]            # 33 named ODE states
    controls: dict[str, float]         # 12 manipulated variables
    raman: Optional[list[float]] = None
    offline: Optional[dict[str, float]] = None
    # Recipe-driven runs populate these from the active phase; legacy
    # (recipe=None) runs leave them None and the state heartbeat falls
    # back to its static default.
    phase: Optional[str] = None
    phase_state: Optional[str] = None
    # Transitions that fired at this sample (empty on most samples; one
    # or more entries on a transition boundary). Kept as plain dicts so
    # the streaming layer has no dependency on recipe module types.
    phase_transitions: Optional[list[dict]] = None


@dataclass(frozen=True)
class StreamConfig:
    """How often to populate the optional fields.

    All defaults assume h=0.2h sample period.

    Attributes:
        raman_every: emit raman field every Nth sample (and only when k > 10
            per indpensim.m:341). Default 1 = every sample.
        lab_every: emit offline field every Nth sample. Default 60 = every
            60×0.2 = 12h, matching MATLAB Off_line_m.
        lab_delay_samples: lab field carries values from k - delay (matches
            MATLAB Off_line_delay = 4h = 20 samples at h=0.2).
        sample_period_s: emit a state sample every N simulated seconds
            (must divide h). None = one sample per control step. Extra
            samples share the step's k and controls; raman/offline/phase
            events only appear on the step-end sample.
    """
    raman_every: int = 1
    lab_every: int = 60
    lab_delay_samples: int = 20
    sample_period_s: float | None = None
