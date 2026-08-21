"""Manual step-by-step control — no second thread needed.

Companion to ``live_control_demo.py`` (which steers a batch from a
*second* thread). Here, analysis and control decisions happen inline, in
the same loop that's consuming samples: ``SampleStream`` is a normal
iterator, so you drive it with ``.next()`` instead of ``for``/``list()``.

Run it:

    python examples/manual_step_loop_demo.py
"""
from __future__ import annotations

import numpy as np

from indpensim.driver import BatchConfig, CampaignConfig, batch_spec_from_python_rng
from indpensim.recipe import legacy_sbc_recipe
from indpensim.simulation import simulate_iter


def main() -> None:
    rng = np.random.default_rng(42)
    spec = batch_spec_from_python_rng(
        rng, batch_no=1,
        campaign=CampaignConfig(optimum_T=10),   # short batch: 50 samples
        batch=BatchConfig(recipe=legacy_sbc_recipe()),
    )

    stream = simulate_iter(spec)
    print(f"control handle: {type(stream.control).__name__}")
    # `.control` is `RecipeExecutor | None` (None only for a batch with no
    # attached Recipe) — this spec always attaches one, so assert it here
    # once rather than under every call below.
    assert stream.control is not None

    while True:
        try:
            sample = stream.next()   # same as next(stream)
            print(f"sample: {sample}")
        except StopIteration:
            break                    # batch finished (or was aborted)

        # --- your analysis: every field is on `sample` ---
        t = sample.state["T"]
        p = sample.state["P"]

        # --- your control decisions, made inline, no other thread involved ---
        if sample.k == 5:
            print(f"k={sample.k:3d}  T={t:.2f}  P={p:.4f}  -> forcing early phase advance")
            stream.control.advance_phase(reason="operator override")

        if sample.k == 10:
            print(f"k={sample.k:3d}  T={t:.2f}  P={p:.4f}  -> overriding Fg to 99.0")
            stream.control.set_setpoint("Fg", 99.0)

        if sample.k == 20:
            print(f"k={sample.k:3d}  T={t:.2f}  P={p:.4f}  -> aborting the batch")
            stream.control.abort(reason="operator stop")

    print("done")


if __name__ == "__main__":
    main()
