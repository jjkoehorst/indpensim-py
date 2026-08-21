"""Streaming module tests.

Five focused tests:
  1. simulate_iter yields the right number of samples
  2. simulate_iter trajectory matches simulate() trajectory (regression gate
     against the refactor)
  3. paced fixed_interval respects timing within slack
  4. UNS topic + payload format matches learning-sim's schema
  5. StreamConfig gates raman field correctly

MQTT round-trip is intentionally NOT tested here (requires a live broker).
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pytest

REF_DIR = Path(__file__).resolve().parents[1] / "data" / "matlab_reference"
INITCONDS_MAT = REF_DIR / "batch_seed42_b01_initconds.mat"


# ---------------------------------------------------------------------------
# 1 + 2: simulate_iter contract
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def short_spec():
    """Use the python-RNG path with a short batch length to keep tests fast."""
    from indpensim.driver import (
        BatchConfig, CampaignConfig, batch_spec_from_python_rng,
    )
    rng = np.random.default_rng(0)
    cfg = CampaignConfig(optimum_T=10)            # 50 samples at h=0.2
    bcfg = BatchConfig(raman_spec=1)
    return batch_spec_from_python_rng(rng, batch_no=1, campaign=cfg, batch=bcfg)


def test_simulate_iter_yields_correct_count(short_spec):
    from indpensim.simulation import simulate_iter
    samples = list(simulate_iter(short_spec))
    expected = int(round(short_spec.T / short_spec.h))    # = 50
    assert len(samples) == expected
    assert samples[0].k == 1
    assert samples[-1].k == expected
    assert samples[0].wall_time_s is None       # unpaced


def test_simulate_iter_control_is_none_without_recipe(short_spec):
    """Legacy (no-Recipe) runs expose .control as None but stream normally."""
    from indpensim.simulation import simulate_iter
    stream = simulate_iter(short_spec)
    assert stream.control is None
    expected = int(round(short_spec.T / short_spec.h))
    assert len(list(stream)) == expected


def test_sample_stream_next_matches_manual_while_loop(short_spec):
    """.next() supports a manual while-loop driver, equivalent to next()."""
    from indpensim.simulation import simulate_iter

    stream = simulate_iter(short_spec)
    expected = int(round(short_spec.T / short_spec.h))
    samples = []
    while True:
        try:
            samples.append(stream.next())
        except StopIteration:
            break
    assert len(samples) == expected
    assert samples[0].k == 1
    assert samples[-1].k == expected


def test_simulate_iter_matches_simulate(short_spec):
    """Regression gate: collect all yielded samples, reconstruct trajectory,
    compare to simulate()'s SimulationResult."""
    from indpensim.simulation import simulate, simulate_iter

    iter_samples = list(simulate_iter(short_spec))
    bulk = simulate(short_spec)

    # Spot-check a few channels at random samples.
    # simulate_iter yields converted state (pH already log-converted, Q/1000),
    # bulk's history.channels has the same convention after finalize().
    rng = np.random.default_rng(7)
    for _ in range(10):
        idx = int(rng.integers(0, len(iter_samples)))
        s = iter_samples[idx]
        for tag in ("S", "P", "V", "Wt", "T", "pH", "Q"):
            iter_v = s.state[tag]
            hist_v = bulk.history.channels[tag][s.k]
            assert np.isclose(iter_v, hist_v, rtol=1e-9, atol=1e-9), (
                f"channel {tag} at k={s.k}: iter={iter_v}, simulate={hist_v}"
            )


# ---------------------------------------------------------------------------
# 3: paced timing
# ---------------------------------------------------------------------------

def test_paced_fixed_interval_respects_timing():
    from indpensim.streaming.pacing import Pacing, paced
    from indpensim.streaming.sample import Sample

    # Synth a tiny stream of 3 samples
    def fake():
        for k in range(1, 4):
            yield Sample(k=k, sim_time_h=k * 0.2, wall_time_s=None,
                         state={}, controls={})

    pacer = Pacing.fixed_interval(0.05)    # 50 ms
    t0 = time.monotonic()
    out = list(paced(fake(), pacer))
    elapsed = time.monotonic() - t0
    assert len(out) == 3
    # Deltas should be ~0.05s. Allow generous slack for CI noise.
    assert 0.08 < elapsed < 0.30, f"elapsed {elapsed:.3f}s out of expected band"
    assert out[0].wall_time_s is not None
    assert out[2].wall_time_s > out[0].wall_time_s


def test_pacing_accelerated_factor_zero_or_negative_rejected():
    from indpensim.streaming.pacing import Pacing
    with pytest.raises(ValueError):
        Pacing.accelerated(0)
    with pytest.raises(ValueError):
        Pacing.accelerated(-1)


def test_parse_pace_spec():
    from indpensim.streaming.pacing import parse_pace_spec
    assert parse_pace_spec("fast") is not None
    assert parse_pace_spec("fixed:1.5") is not None
    assert parse_pace_spec("accelerated:60") is not None
    assert parse_pace_spec("accel:30") is not None
    with pytest.raises(ValueError):
        parse_pace_spec("garbage")


# ---------------------------------------------------------------------------
# 4: UNS topic + payload format
# ---------------------------------------------------------------------------

def test_uns_build_messages_format():
    from indpensim.streaming.sample import Sample
    from indpensim.streaming.uns import UnsConfig, build_messages

    sample = Sample(
        k=42, sim_time_h=8.4, wall_time_s=None,
        state={"T": 298.0, "pH": 6.5, "P": 12.3, "X": 30.0,
               "S": 1.2, "DO2": 14.0, "V": 60000.0, "Wt": 65000.0,
               "Viscosity": 8.0, "OUR": 0.5, "CER": 0.4},
        controls={"Fc": 100.0, "Fb": 5.0, "Fpaa": 4.0, "Fg": 60.0,
                  "RPM": 100.0, "pressure": 0.9},
    )
    cfg = UnsConfig(equipment="bioreactor-real")
    msgs = build_messages(sample, cfg)

    # Spot-check: temperature converted to Celsius
    temp_msg = next(p for t, p in msgs if t.endswith("/temperature"))
    payload = json.loads(temp_msg)
    assert payload["unit"] == "degC"
    assert abs(payload["value"] - (298.0 - 273.15)) < 1e-9
    assert payload["k"] == 42

    # Topic prefix matches learning-sim's UNS schema
    for topic, _ in msgs:
        assert topic.startswith("uns/plant1/fermentation/line1/bioreactor-real/")


def test_uns_state_message():
    from indpensim.streaming.sample import Sample
    from indpensim.streaming.uns import UnsConfig, build_state_message

    sample = Sample(k=1, sim_time_h=0.2, wall_time_s=None, state={}, controls={})
    cfg = UnsConfig(equipment="bioreactor-real")
    topic, payload = build_state_message(sample, cfg, phase="FERMENT", batch_id=99)
    p = json.loads(payload)
    assert topic.endswith("/_state")
    assert p["phase"] == "FERMENT"
    assert p["batch_id"] == 99
    assert p["k"] == 1
    assert p["running"] is True


# ---------------------------------------------------------------------------
# 5: StreamConfig gating
# ---------------------------------------------------------------------------

def test_stream_config_gates_raman(short_spec):
    """With raman_every=10, only every 10th sample (and k>10) carries raman."""
    from indpensim.simulation import simulate_iter
    from indpensim.streaming.sample import StreamConfig

    cfg = StreamConfig(raman_every=10)
    samples = list(simulate_iter(short_spec, stream_config=cfg))

    raman_at_k = [s.k for s in samples if s.raman is not None]
    # k must be > 10 AND k % 10 == 0 → for short_spec with 50 samples,
    # those are k=20, 30, 40, 50.
    assert raman_at_k == [20, 30, 40, 50]

    # And every Sample at any k > 10 with k % 1 == 0 (default) would emit.
    cfg_default = StreamConfig(raman_every=1)
    samples_def = list(simulate_iter(short_spec, stream_config=cfg_default))
    raman_at_k_def = [s.k for s in samples_def if s.raman is not None]
    assert raman_at_k_def == list(range(11, 51))


def test_stream_config_gates_offline(short_spec):
    """Offline measurements with delay; only emitted on the configured cadence."""
    from indpensim.simulation import simulate_iter
    from indpensim.streaming.sample import StreamConfig

    # offline_every=20 + delay=5 → emit at k=20, 40 (where k > delay AND k%20==0)
    cfg = StreamConfig(lab_every=20, lab_delay_samples=5)
    samples = list(simulate_iter(short_spec, stream_config=cfg))
    offline_at_k = [s.k for s in samples if s.offline is not None]
    assert offline_at_k == [20, 40]
    # Verify analyte set
    s = next(s for s in samples if s.k == 20)
    assert set(s.offline.keys()) == {"NH3", "P", "X", "PAA", "Viscosity"}
