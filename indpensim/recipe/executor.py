"""Stateful Recipe executor — consumed by ``controller_step``.

One executor per batch. ``.step(k, history)`` must be called with
strictly increasing k starting at 1, the same cadence as the outer
simulation loop. It:

  1. If the active phase is RUNNING, checks its transition trigger
     against time-in-phase and (optionally) a state threshold. If the
     trigger fires, advances to the next phase (or marks COMPLETE if
     this was the last one). Multiple transitions can fire in a single
     call — e.g. a phase with ``max_hours=0``.
  2. Resolves the 7 feed setpoints from the active phase's
     ``SetpointProfile`` using first-match / fall-through-to-last
     semantics identical to the legacy ``controller._recipe_lookup``.

Mutator hooks (``pause``/``resume``/``advance_phase``/``abort``) let a
second thread — e.g. an operator console driving a live ``simulate_iter``
stream via its ``.control`` handle — reach into an in-flight batch between
``step()`` calls. A single internal lock serializes each hook against
``step()``'s own check-and-mutate section; this is enough to avoid a torn
read/write for the expected one-control-thread/one-sim-thread usage, not a
general concurrency framework. Each hook raises ``RuntimeError`` (naming
the attempted operation and the actual state) when called from a state it
doesn't accept — never a silent no-op.
"""
from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import NamedTuple

from indpensim.control.history import BatchHistory
from indpensim.recipe.types import (
    Phase,
    PhaseState,
    Recipe,
    SetpointSchedule,
    TransitionTrigger,
)


class ResolvedSetpoints(NamedTuple):
    """The 7 feed setpoints resolved at one sample k, plus optional
    per-phase ``T_sp``/``pH_sp`` overrides.

    ``Fdischarge`` is positive; the controller applies the sign.
    ``Fwater`` mirrors the legacy ``Fw``.

    ``T_sp`` (Kelvin) and ``pH_sp`` are ``None`` when the active phase
    does not override them; the controller falls back to
    ``ControlFlags.T_sp`` / ``ControlFlags.pH_sp`` in that case. New
    fields appended at the end for backwards-compatible positional
    consumers.
    """
    Fs: float
    Foil: float
    Fg: float
    pressure: float
    Fdischarge: float
    Fwater: float
    Fpaa: float
    T_sp: float | None = None
    pH_sp: float | None = None


@dataclass(frozen=True)
class PhaseTransitionLog:
    from_phase: str
    to_phase: str | None          # None when the last phase completes
    at_k: int
    at_time_h: float
    reason: str                    # "trigger" | "advance" | "abort" | "complete"


def _lookup(k: int, schedule: SetpointSchedule, default: float = 0.0) -> float:
    """First-match / fall-through-to-last, matching ``_recipe_lookup``."""
    if not schedule:
        return default
    for bp, sp in schedule:
        if k <= bp:
            return float(sp)
    return float(schedule[-1][1])


def _trigger_fires(
    trig: TransitionTrigger,
    time_in_phase_h: float,
    history: BatchHistory,
    k: int,
) -> bool:
    if trig.max_hours is not None and time_in_phase_h >= trig.max_hours:
        return True
    if trig.state_var is not None and trig.state_value is not None:
        # Read the most recent populated history slot.
        idx = max(k - 1, 1)
        val = history.y(trig.state_var, idx)
        if trig.state_op == ">=" and val >= trig.state_value:
            return True
        if trig.state_op == "<=" and val <= trig.state_value:
            return True
    return False


@dataclass
class RecipeExecutor:
    recipe: Recipe
    h: float                                # sample period, hours
    _phase_idx: int = 0
    _phase_start_k: int = 1
    _phase_state: PhaseState = PhaseState.RUNNING
    _transitions: list[PhaseTransitionLog] = field(default_factory=list)
    _drained: int = 0                        # cursor for streaming drain
    _held_samples: int = 0                   # samples elapsed while HELD, this phase
    _last_k: int = 0                         # most recent k seen by step()
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False, compare=False)

    @property
    def current_phase(self) -> Phase:
        return self.recipe.phases[self._phase_idx]

    @property
    def phase_state(self) -> PhaseState:
        return self._phase_state

    @property
    def transitions(self) -> tuple[PhaseTransitionLog, ...]:
        return tuple(self._transitions)

    def drain_new_transitions(self) -> list[PhaseTransitionLog]:
        """Return transitions logged since the last drain (streaming hook)."""
        new = self._transitions[self._drained :]
        self._drained = len(self._transitions)
        return list(new)

    # ------------------------------------------------------------------
    def step(self, k: int, history: BatchHistory) -> ResolvedSetpoints:
        # Advance phases while triggers fire (normally at most once per step,
        # but a chain of zero-duration phases could cascade).
        with self._lock:
            self._last_k = k
            if self._phase_state == PhaseState.HELD:
                self._held_samples += 1
            while self._phase_state == PhaseState.RUNNING:
                # _held_samples excludes any wall-time spent paused earlier
                # in this phase from counting toward max_hours.
                time_in_phase = (k - self._phase_start_k - self._held_samples) * self.h
                phase = self.current_phase
                if not _trigger_fires(phase.transition, time_in_phase, history, k):
                    break
                self._advance(k, reason="trigger")

        sp = self.current_phase.setpoints
        return ResolvedSetpoints(
            Fs=_lookup(k, sp.Fs),
            Foil=_lookup(k, sp.Foil),
            Fg=_lookup(k, sp.Fg),
            pressure=_lookup(k, sp.pressure),
            Fdischarge=_lookup(k, sp.Fdischarge),
            Fwater=_lookup(k, sp.Fwater),
            Fpaa=_lookup(k, sp.Fpaa),
            T_sp=sp.T_sp,
            pH_sp=sp.pH_sp,
        )

    # ------------------------------------------------------------------
    def _advance(self, k: int, *, reason: str) -> None:
        # Only ever called from within a section already holding self._lock
        # (step()'s while-loop, advance_phase()) — threading.Lock isn't
        # reentrant, so this method must not acquire it itself.
        self._held_samples = 0
        from_name = self.current_phase.name
        if self._phase_idx + 1 < len(self.recipe.phases):
            to_name = self.recipe.phases[self._phase_idx + 1].name
            self._phase_idx += 1
            self._phase_start_k = k
            self._phase_state = PhaseState.RUNNING
            self._transitions.append(PhaseTransitionLog(
                from_phase=from_name, to_phase=to_name,
                at_k=k, at_time_h=k * self.h, reason=reason,
            ))
        else:
            self._phase_state = PhaseState.COMPLETE
            self._transitions.append(PhaseTransitionLog(
                from_phase=from_name, to_phase=None,
                at_k=k, at_time_h=k * self.h, reason="complete",
            ))

    # ---- Mutator hooks — live control from outside the sim loop. --------
    def pause(self) -> None:
        """Hold the current phase — stop automatic phase transitions.

        The active phase's setpoint schedule keeps resolving by absolute
        ``k`` as normal; only the transition-trigger check is skipped.
        """
        with self._lock:
            if self._phase_state != PhaseState.RUNNING:
                raise RuntimeError(
                    f"cannot pause(): executor is {self._phase_state.name}, expected RUNNING"
                )
            self._phase_state = PhaseState.HELD

    def resume(self) -> None:
        """Resume automatic phase transitions after ``pause()``."""
        with self._lock:
            if self._phase_state != PhaseState.HELD:
                raise RuntimeError(
                    f"cannot resume(): executor is {self._phase_state.name}, expected HELD"
                )
            self._phase_state = PhaseState.RUNNING

    def advance_phase(self, reason: str | None = None) -> None:
        """Force an immediate transition to the next phase, bypassing its
        trigger. Accepted from RUNNING or HELD — forcing an advance
        implicitly un-pauses. Uses the ``k`` of the most recent ``step()``
        call as the transition's timestamp.
        """
        with self._lock:
            if self._phase_state not in (PhaseState.RUNNING, PhaseState.HELD):
                raise RuntimeError(
                    f"cannot advance_phase(): executor is {self._phase_state.name}, "
                    "expected RUNNING or HELD"
                )
            self._advance(self._last_k, reason=reason or "advance")

    def abort(self, reason: str | None = None) -> None:
        """End the batch early. The streaming layer (``SampleStream``) stops
        yielding further samples once it observes ``ABORTED``.
        """
        with self._lock:
            if self._phase_state in (PhaseState.ABORTED, PhaseState.COMPLETE):
                raise RuntimeError(
                    f"cannot abort(): executor is already {self._phase_state.name}"
                )
            from_name = self.current_phase.name
            self._phase_state = PhaseState.ABORTED
            self._transitions.append(PhaseTransitionLog(
                from_phase=from_name, to_phase=None,
                at_k=self._last_k, at_time_h=self._last_k * self.h,
                reason=reason or "abort",
            ))
