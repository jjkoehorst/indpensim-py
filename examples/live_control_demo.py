"""Live control of a streaming batch — pause / resume / advance / abort.

Demonstrates the ``.control`` handle that ``simulate_iter()`` exposes on its
``SampleStream``: a second thread reaches into an in-flight batch and steers
it while the main thread is mid-iteration. See the "Live control" section of
the top-level README for the full write-up.

Run it:

    python examples/live_control_demo.py
"""
from __future__ import annotations

import threading
import time

import numpy as np

from indpensim.driver import BatchConfig, CampaignConfig, batch_spec_from_python_rng
from indpensim.recipe import legacy_sbc_recipe
from indpensim.simulation import simulate_iter
from indpensim.streaming.pacing import Pacing, paced


def operator_console(stream) -> None:
    """Runs on a background thread while the main thread streams samples.

    Timed to land well before INOCULATE's own 4h/20-sample trigger would
    naturally fire, so each effect below is unambiguously due to the call
    that produced it, not a coincidence with the recipe's own schedule.
    """
    time.sleep(0.3)
    print("[console] pausing the recipe")
    stream.control.pause()

    time.sleep(0.3)
    print("[console] resuming")
    stream.control.resume()

    time.sleep(0.3)
    print("[console] forcing an early phase advance")
    stream.control.advance_phase(reason="operator override")

    time.sleep(0.5)
    print("[console] aborting the batch")
    stream.control.abort(reason="operator stop")


def main() -> None:
    rng = np.random.default_rng(42)
    spec = batch_spec_from_python_rng(
        rng, batch_no=1,
        campaign=CampaignConfig(optimum_T=10),   # short batch: 50 samples
        batch=BatchConfig(recipe=legacy_sbc_recipe()),
    )

    stream = simulate_iter(spec)
    print(f"control handle: {type(stream.control).__name__}")

    threading.Thread(target=operator_console, args=(stream,), daemon=True).start()

    # fixed_interval paces one sample per 0.1 wall-seconds, giving the
    # console thread above real gaps to act in.
    last_phase = None
    n = 0
    for sample in paced(stream, Pacing.fixed_interval(0.1)):
        n += 1
        if sample.phase != last_phase:
            print(f"k={sample.k:3d}  phase={sample.phase:<10}  state={sample.phase_state}")
            last_phase = sample.phase

    print(f"stream ended after {n} samples (batch was authored for 50)")


if __name__ == "__main__":
    main()
