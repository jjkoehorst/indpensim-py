"""Pacing strategies for streaming Sample iterators.

Three modes:

  - ``as_fast_as_possible()`` — no sleep, max throughput. Same wall-clock
    behavior as ``simulate()``. For backfilling ML training data.
  - ``fixed_interval(seconds)`` — emit one Sample every N wall-seconds
    regardless of the simulation's h. For dashboards and SCADA-rate replay.
  - ``accelerated(factor)`` — sleep so wall_elapsed ≈ sim_elapsed / factor.
    factor=1 → real-time (12 minutes per sample at h=0.2). factor=360 →
    one sample every 2 wall-seconds (matches learning-sim's 2s scan rate).

``paced(samples, pacer)`` wraps any Sample iterator, sleeping appropriately
between yields and stamping ``wall_time_s`` on each yielded Sample.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, replace
from typing import Iterable, Iterator, Protocol

from indpensim.streaming.sample import Sample


class Pacer(Protocol):
    """Strategy: given the previous sample's sim_time_h and this sample's
    sim_time_h, decide how long to sleep before emitting this one (relative
    to the stream's wall-clock start).
    """
    def wait_until(self, sample: Sample, *, t0_wall: float, t0_sim_h: float,
                   index: int = 0) -> float:
        """Block until the sample should be emitted; return wall_time_s
        (seconds since stream start) at the moment of emission."""
        ...


@dataclass(frozen=True)
class _AsFastAsPossible:
    def wait_until(self, sample: Sample, *, t0_wall: float, t0_sim_h: float,
                   index: int = 0) -> float:
        return time.monotonic() - t0_wall


@dataclass(frozen=True)
class _FixedInterval:
    seconds: float

    def wait_until(self, sample: Sample, *, t0_wall: float, t0_sim_h: float,
                   index: int = 0) -> float:
        # Count emitted samples, not k: sub-step samples share a k.
        target_offset = index * self.seconds
        sleep_for = (t0_wall + target_offset) - time.monotonic()
        if sleep_for > 0:
            time.sleep(sleep_for)
        return time.monotonic() - t0_wall


@dataclass(frozen=True)
class _Accelerated:
    factor: float    # >1 = faster than real-time

    def wait_until(self, sample: Sample, *, t0_wall: float, t0_sim_h: float,
                   index: int = 0) -> float:
        # Target wall offset = (sim_time_h - t0_sim_h) * 3600 / factor
        target_offset_s = (sample.sim_time_h - t0_sim_h) * 3600.0 / self.factor
        sleep_for = (t0_wall + target_offset_s) - time.monotonic()
        if sleep_for > 0:
            time.sleep(sleep_for)
        return time.monotonic() - t0_wall


class Pacing:
    """Factory for the three pacers."""

    @staticmethod
    def as_fast_as_possible() -> Pacer:
        return _AsFastAsPossible()

    @staticmethod
    def fixed_interval(seconds: float) -> Pacer:
        if seconds < 0:
            raise ValueError(f"interval must be >= 0, got {seconds}")
        return _FixedInterval(seconds=float(seconds))

    @staticmethod
    def accelerated(factor: float) -> Pacer:
        if factor <= 0:
            raise ValueError(f"factor must be > 0, got {factor}")
        return _Accelerated(factor=float(factor))


def paced(samples: Iterable[Sample], pacer: Pacer) -> Iterator[Sample]:
    """Wrap a Sample iterator with a pacer. Sleeps before each yield and
    stamps ``wall_time_s`` on the emitted Sample."""
    t0_wall: float | None = None
    t0_sim_h: float | None = None
    for index, sample in enumerate(samples):
        if t0_wall is None:
            t0_wall = time.monotonic()
            t0_sim_h = sample.sim_time_h
        wall_t = pacer.wait_until(sample, t0_wall=t0_wall, t0_sim_h=t0_sim_h, index=index)
        yield replace(sample, wall_time_s=wall_t)


def parse_pace_spec(spec: str) -> Pacer:
    """Parse CLI ``--pace`` syntax: ``fast | fixed:<seconds> | accelerated:<factor>``."""
    if spec == "fast":
        return Pacing.as_fast_as_possible()
    if ":" not in spec:
        raise ValueError(f"unknown pace spec: {spec!r}")
    kind, _, value = spec.partition(":")
    if kind == "fixed":
        return Pacing.fixed_interval(float(value))
    if kind in ("accel", "accelerated"):
        return Pacing.accelerated(float(value))
    raise ValueError(f"unknown pace kind: {kind!r}")
