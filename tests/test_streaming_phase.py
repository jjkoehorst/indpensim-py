"""Phase/phase_state propagation through the streaming layer.

Two things to pin down:

  1. Samples from ``simulate_iter`` carry ``phase`` / ``phase_state``
     when a Recipe is attached, and leave them ``None`` otherwise.
  2. Each executor transition produces exactly one ``_phase_start``
     message via ``build_phase_transitions`` — no duplicates across
     samples.
"""
from __future__ import annotations

import json

import numpy as np
import pytest

from indpensim.driver import (
    BatchConfig, CampaignConfig, batch_spec_from_python_rng,
)
from indpensim.recipe.legacy import legacy_sbc_recipe
from indpensim.recipe.types import Phase, Recipe, TransitionTrigger
from indpensim.simulation import simulate_iter
from indpensim.streaming.uns import (
    UnsConfig, build_batch_start_message, build_phase_transitions,
    build_state_message,
)


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------

def _short_spec_with_recipe(recipe):
    rng = np.random.default_rng(0)
    cfg = CampaignConfig(optimum_T=10)       # 50 samples at h=0.2
    bcfg = BatchConfig(recipe=recipe)
    return batch_spec_from_python_rng(rng, batch_no=1, campaign=cfg, batch=bcfg)


def _short_spec_no_recipe():
    rng = np.random.default_rng(0)
    cfg = CampaignConfig(optimum_T=10)
    bcfg = BatchConfig()
    return batch_spec_from_python_rng(rng, batch_no=1, campaign=cfg, batch=bcfg)


@pytest.fixture(scope="module")
def short_recipe():
    """4-phase recipe that fires all boundaries inside a 10h batch.

    Reuses the legacy SBC setpoint profiles (so Fg/pressure etc. have
    physically reasonable values the ODE can integrate) but compresses
    the per-phase time triggers so INOCULATE→GROWTH→PRODUCTION→HARVEST
    all occur within 50 samples.
    """
    base = legacy_sbc_recipe()
    # max_hours=2.0 for every phase → transitions at k=11, 21, 31.
    return Recipe(name="short_sbc", phases=tuple(
        Phase(
            name=p.name,
            setpoints=p.setpoints,
            transition=TransitionTrigger(max_hours=2.0),
        )
        for p in base.phases
    ))


# ---------------------------------------------------------------------------
# Sample carries phase fields
# ---------------------------------------------------------------------------

def test_sample_phase_none_without_recipe():
    spec = _short_spec_no_recipe()
    samples = list(simulate_iter(spec))
    assert all(s.phase is None for s in samples)
    assert all(s.phase_state is None for s in samples)
    assert all(not s.phase_transitions for s in samples)


def test_sample_phase_populated_with_recipe(short_recipe):
    spec = _short_spec_with_recipe(short_recipe)
    samples = list(simulate_iter(spec))
    # k=1..10 on INOCULATE; trigger fires at k=11.
    assert samples[0].phase == "INOCULATE"
    assert samples[0].phase_state == "RUNNING"
    assert samples[9].phase == "INOCULATE"       # k=10 is idx 9
    assert samples[10].phase == "GROWTH"         # k=11 is idx 10
    assert samples[20].phase == "PRODUCTION"     # k=21
    assert samples[30].phase == "HARVEST"        # k=31
    # HARVEST's trigger fires at k=41; phase name stays HARVEST but
    # phase_state becomes COMPLETE (terminal; no further phases).
    assert samples[-1].phase == "HARVEST"
    assert samples[-1].phase_state == "COMPLETE"


# ---------------------------------------------------------------------------
# Transitions fire exactly once
# ---------------------------------------------------------------------------

def test_transitions_emitted_exactly_once(short_recipe):
    spec = _short_spec_with_recipe(short_recipe)
    samples = list(simulate_iter(spec))
    cfg = UnsConfig()

    transition_msgs: list[dict] = []
    for s in samples:
        for _, payload in build_phase_transitions(s, cfg, batch_id=1):
            transition_msgs.append(json.loads(payload))

    # Four events: the three phase advances plus the COMPLETE of HARVEST.
    assert len(transition_msgs) == 4
    phases = [m["phase"] for m in transition_msgs]
    assert phases == ["GROWTH", "PRODUCTION", "HARVEST", "__COMPLETE__"]
    from_phases = [m["from_phase"] for m in transition_msgs]
    assert from_phases == ["INOCULATE", "GROWTH", "PRODUCTION", "HARVEST"]
    reasons = [m["reason"] for m in transition_msgs]
    assert reasons == ["trigger", "trigger", "trigger", "complete"]
    # Monotonic in sim_time_h.
    times = [m["sim_time_h"] for m in transition_msgs]
    assert times == sorted(times)


def test_legacy_recipe_emits_no_transitions_on_short_batch():
    # Legacy recipe boundaries are at 4h, 40h, 280h, 350h — a 10h batch
    # only crosses the first boundary, so exactly one transition fires.
    spec = _short_spec_with_recipe(legacy_sbc_recipe())
    samples = list(simulate_iter(spec))
    cfg = UnsConfig()
    n_transitions = sum(
        len(build_phase_transitions(s, cfg, batch_id=1)) for s in samples
    )
    assert n_transitions == 1


# ---------------------------------------------------------------------------
# State message carries phase_state
# ---------------------------------------------------------------------------

def test_state_message_uses_sample_phase(short_recipe):
    spec = _short_spec_with_recipe(short_recipe)
    samples = list(simulate_iter(spec))
    cfg = UnsConfig()
    _, payload = build_state_message(samples[15], cfg, batch_id=1)
    msg = json.loads(payload)
    assert msg["phase"] == "GROWTH"
    assert msg["phase_state"] == "RUNNING"


def test_state_message_falls_back_to_default_without_recipe():
    spec = _short_spec_no_recipe()
    samples = list(simulate_iter(spec))
    cfg = UnsConfig()
    _, payload = build_state_message(samples[0], cfg, batch_id=1,
                                      phase="FERMENT")
    msg = json.loads(payload)
    assert msg["phase"] == "FERMENT"
    assert msg["phase_state"] == "RUNNING"


# ---------------------------------------------------------------------------
# .control — live steering of an in-flight stream
# ---------------------------------------------------------------------------

def test_control_is_recipe_executor_when_recipe_attached(short_recipe):
    spec = _short_spec_with_recipe(short_recipe)
    stream = simulate_iter(spec)
    from indpensim.recipe.executor import RecipeExecutor
    assert isinstance(stream.control, RecipeExecutor)


def test_control_advance_phase_forces_transition_mid_stream(short_recipe):
    spec = _short_spec_with_recipe(short_recipe)
    stream = simulate_iter(spec)
    samples = []
    for sample in stream:
        samples.append(sample)
        if sample.k == 5:
            stream.control.advance_phase(reason="operator")
    # k=5 (idx 4) is still INOCULATE; the forced advance takes effect on the
    # very next sample, well before INOCULATE's own trigger would fire (k=11).
    assert samples[4].phase == "INOCULATE"
    assert samples[5].phase == "GROWTH"
    transition = samples[5].phase_transitions[0]
    assert transition["from_phase"] == "INOCULATE"
    assert transition["to_phase"] == "GROWTH"
    assert transition["reason"] == "operator"


def test_control_abort_ends_stream_early(short_recipe):
    spec = _short_spec_with_recipe(short_recipe)
    stream = simulate_iter(spec)
    samples = []
    for sample in stream:
        samples.append(sample)
        if sample.k == 7:
            stream.control.abort(reason="emergency stop")
    # One more sample (k=8) is yielded after the abort call, then the
    # stream stops — well short of the batch's full 50 samples.
    assert len(samples) == 8
    assert samples[-1].k == 8
    assert samples[-1].phase_state == "ABORTED"


# ---------------------------------------------------------------------------
# Batch-start message carries recipe metadata
# ---------------------------------------------------------------------------

def test_batch_start_with_recipe():
    recipe = legacy_sbc_recipe()
    _, payload = build_batch_start_message(recipe, UnsConfig(), batch_id=7)
    msg = json.loads(payload)
    assert msg["batch_id"] == 7
    assert msg["recipe_name"] == "legacy_sbc"
    assert msg["phases"] == ["INOCULATE", "GROWTH", "PRODUCTION", "HARVEST"]


def test_batch_start_without_recipe():
    _, payload = build_batch_start_message(None, UnsConfig(), batch_id=1)
    msg = json.loads(payload)
    assert msg["batch_id"] == 1
    assert "recipe_name" not in msg
    assert "phases" not in msg
