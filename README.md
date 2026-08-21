# indpensim-py

Python port of **IndPenSim V2.02** — an industrial-scale fed-batch penicillin
fermentation simulator. The original is MATLAB; this is a faithful end-to-end
Python port validated against MATLAB reference trajectories — mean
peak-normalized error under 1% on most states across 12 distinct configs
(see [VALIDATION.md](VALIDATION.md) for the full picture, including the
three channels where closed-loop solver amplification leaves wider bounds).

## What this is

- 33-state stiff ODE model of an industrial penicillin fermentation
- pH and temperature PID control, sequential-batch recipe driver,
  fault injection, PRBS noise
- Simulated Raman spectroscopy + PLS-based PAA concentration prediction
- Multi-batch campaign driver with CLI and CSV outputs
- Streaming layer — iterator API (`simulate_iter()`) plus an MQTT runner
  that publishes each timestep to a UNS-shaped topic tree; pacing is
  configurable from as-fast-as-possible down to true real-time
- Optional ISA-88-subset **Recipe layer** — author phase-structured
  batches (INOCULATE → GROWTH → PRODUCTION → HARVEST) with hybrid
  time/state transition triggers; phase context flows into the MQTT
  stream as `_phase_start` events. The legacy hardcoded SBC tables
  remain the default; attach a Recipe explicitly to opt in.

## Upstream / original

The original MATLAB simulator (which this port reproduces) is by Stephen Goldrick et al.

- Download:  http://www.industrialpenicillinsimulation.com/
- Paper:     Goldrick et al., "Modern day control challenges for industrial-scale
             fermentation processes," *Computers & Chemical Engineering*, 2019.
             https://doi.org/10.1016/j.compchemeng.2019.05.037
- Earlier:   Goldrick et al., "The Development of an Industrial Scale Fed-Batch
             Fermentation Simulation," *Journal of Biotechnology*, 2015.
             https://doi.org/10.1016/j.jbiotec.2014.10.029

## Install

```bash
uv venv && source .venv/bin/activate
uv pip install -e .
```

(Or use `pip install -e .` if you don't have uv.)

For the MQTT streaming runner, install the optional extra:

```bash
uv pip install -e '.[mqtt]'
```

For running tests or other contributor work, install the dev extra (pulls
in pytest and matplotlib):

```bash
uv pip install -e '.[dev]'
```

For the Streamlit Recipe authoring / visualization UI, install the `ui`
extra (streamlit + plotly):

```bash
uv pip install -e '.[ui]'
```

## Run

Generate a campaign of batches and dump per-batch CSVs:

```bash
python -m indpensim.driver --num-batches 5 --seed 42 --out runs/
```

Or replay a captured MATLAB initial condition (for validation):

```bash
python -m indpensim.driver --from-capture --capture-seeds 42 --capture-batches 1 --out runs/
```

From Python:

```python
from indpensim.driver import batch_spec_from_python_rng, CampaignConfig, BatchConfig
from indpensim.simulation import simulate
import numpy as np

rng = np.random.default_rng(42)
spec = batch_spec_from_python_rng(rng, batch_no=1,
                                   campaign=CampaignConfig(),
                                   batch=BatchConfig(raman_spec=1))
result = simulate(spec)
print(result.history.channels["P"][-1])     # final penicillin g/L
```

### Streaming to MQTT

Stream a live batch to an MQTT broker (default pacing: one sample every
2 wall-seconds, ~360× real-time, matching a 2s SCADA scan rate):

```bash
python -m indpensim.streaming.mqtt_runner --broker localhost
```

Pacing options via `--pace`:

| Spec | Behavior |
|---|---|
| `fast` | No sleep; full 230h batch streams in ~25s wall time. For backfilling ML training data. |
| `fixed:<seconds>` | One sample every N wall-seconds regardless of sim h. Default `fixed:2.0`. |
| `accelerated:<factor>` | Wall ≈ sim_h · 3600 / factor. `factor=1` is true real-time (12 min per sample); `factor=360` ≈ `fixed:2.0`. |

Other useful flags: `--raman-every N` / `--lab-every N` control cadence
of the two slow channels independently of the state stream; `--faults
0..8` and `--raman-spec 0..2` configure the batch; `--from-capture
--capture-seed 42` streams from a captured MATLAB init instead of a
fresh Python RNG draw. Topic tree follows the UNS convention
`uns/{site}/{area}/{line}/{equipment}/{tag}` — equipment defaults to
`bioreactor-real`.

From Python (for custom sinks, no MQTT dependency):

```python
from indpensim.simulation import simulate_iter
from indpensim.streaming.pacing import Pacing, paced

for sample in paced(simulate_iter(spec), Pacing.fixed_interval(2.0)):
    print(sample.sim_time_h, sample.state["P"], sample.state["T"])
```

#### Live control

`simulate_iter()` returns a `SampleStream`, not a plain generator. Hold onto
it and its `.control` attribute gives you the batch's live `RecipeExecutor`
(`None` for a batch with no attached Recipe) — safe to call from another
thread while the main thread is mid-iteration, e.g. from an operator console
that reacts to what it sees streaming past:

```python
import threading
from indpensim.simulation import simulate_iter
from indpensim.streaming.pacing import Pacing, paced

stream = simulate_iter(spec)   # spec.batch.recipe must be set

def operator_console():
    input("press enter to pause...")
    stream.control.pause()
    input("press enter to resume...")
    stream.control.resume()

threading.Thread(target=operator_console, daemon=True).start()

for sample in paced(stream, Pacing.accelerated(factor=1.0)):
    print(sample.sim_time_h, sample.phase, sample.state["P"])
```

`pause()` stops automatic phase transitions (the setpoint schedule keeps
resolving normally); `resume()` restarts them, and the time spent paused
never counts against a phase's `max_hours` trigger. `advance_phase(reason=...)`
forces an immediate transition to the next phase, bypassing its trigger —
valid from a running or paused phase. `abort(reason=...)` ends the batch
early: the stream yields one more sample (with `phase_state ==
"ABORTED"`) and then stops. Calling a hook from a state it doesn't accept
(e.g. `resume()` when not paused) raises `RuntimeError` naming the actual
state — never a silent no-op.

To change an actual value — a flow rate, `T_sp`, `pH_sp` — rather than which
phase is active, use `set_setpoint`/`set_setpoints`. These force a fixed
number regardless of what the active phase's `SetpointProfile` authors, and
(unlike the phase hooks above) persist across phase transitions until you
clear them.

Valid names are any `ResolvedSetpoints` field — an unknown name raises
`ValueError`:

| name | meaning |
|---|---|
| `Fs` | substrate (sugar) feed rate |
| `Foil` | oil feed rate |
| `Fg` | aeration / gas flow rate |
| `pressure` | vessel headspace pressure setpoint |
| `Fdischarge` | harvest/discharge flow rate |
| `Fwater` | dilution water flow rate |
| `Fpaa` | PAA precursor feed rate |
| `T_sp` | temperature setpoint (Kelvin) |
| `pH_sp` | pH setpoint |

These are the only things you can directly force to a value. The rest of
`sample.controls` (`RPM`, `Fa`, `Fb`, `Fc`, `Fh`, `viscosity`, `Fremoved`)
are PID/controller *outputs* computed each step from `T_sp`/`pH_sp` and
internal state — not independent setpoints. To change cooling duty (`Fc`)
or acid/base dosing (`Fa`/`Fb`), change `T_sp`/`pH_sp` and let the PID react
to it; you can't pin those directly.

**Set:**

```python
stream.control.set_setpoint("T_sp", 305.0)       # one value
stream.control.set_setpoints(Fg=60.0, pH_sp=6.8) # several at once
```

**Get:**

```python
stream.control.overrides           # {'T_sp': 305.0, 'Fg': 60.0, 'pH_sp': 6.8}
                                    # — currently active overrides only

sample.state["T"]                  # any of the 33 actual ODE states
sample.controls["Fg"]              # any of the 12 actual actuator outputs
                                    # (the resolved value the controller used,
                                    # override or not)
```

**Clear:**

```python
stream.control.clear_setpoint("T_sp")  # revert just T_sp to the phase's authored value
stream.control.clear_all_setpoints()   # revert everything
```

#### Single-thread step loop

You don't need a second thread at all if your analysis and control
decisions happen in the same place you're consuming samples.
`SampleStream` is a normal iterator, so drive it step by step with
`.next()` (or the builtin `next(stream)` — they're equivalent) instead of
`for`/`list()`:

```python
stream = simulate_iter(spec)

while True:
    try:
        sample = stream.next()
    except StopIteration:
        break               # batch finished (or was aborted)

    # --- your analysis ---
    if sample.state["T"] > 305.0:
        # --- your control decision ---
        stream.control.set_setpoint("Fc", 50.0)   # e.g. bump cooling flow
    if sample.k == 100:
        stream.control.pause()
```

### Recipe layer (optional)

Authoring a phase-structured batch via the ISA-88-subset Recipe API:

```python
from indpensim.driver import BatchConfig, CampaignConfig, batch_spec_from_python_rng
from indpensim.recipe import legacy_sbc_recipe
from indpensim.simulation import simulate
import numpy as np

rng = np.random.default_rng(42)
spec = batch_spec_from_python_rng(
    rng, batch_no=1,
    campaign=CampaignConfig(),
    batch=BatchConfig(recipe=legacy_sbc_recipe()),
)
result = simulate(spec)
```

`legacy_sbc_recipe()` is a 4-phase Recipe (INOCULATE / GROWTH /
PRODUCTION / HARVEST) that reconstitutes the hardcoded SBC tables
bit-for-bit — the regression anchor for any custom Recipe. Build your
own with `Phase`, `SetpointProfile`, and `TransitionTrigger`
(time-in-phase, state threshold, or both — whichever fires first
advances). Recipes round-trip through JSON via `recipe.to_dict` /
`recipe.from_dict`.

When a Recipe is attached, each streaming `Sample` carries `phase`,
`phase_state`, and `phase_transitions`; the MQTT runner publishes a
`_batch_start` message with recipe metadata once per batch and emits
`_phase_start` messages as transitions fire.

### Streamlit UI (optional)

A lightweight Streamlit studio for authoring and visualizing recipes
lives in `indpensim/ui/`. Install the `ui` extra and launch:

```bash
uv pip install -e '.[ui]'
streamlit run indpensim/ui/streamlit_app.py
```

Two pages in the sidebar:

- **Authoring** — add/remove/reorder phases, edit per-channel setpoint
  schedules in live tables (`st.data_editor`), configure hybrid time /
  state transition triggers, validate on every interaction, save/load
  as JSON.
- **Visualize** — load a Recipe JSON (or use the session recipe) and
  inspect its setpoint timeline as a stacked Plotly figure with phase
  boundaries drawn as colored vertical bands.

The in-session default is `legacy_sbc_recipe()`.

## Layout

```
indpensim/
  driver.py           - multi-batch campaign + CLI
  simulation.py       - main loop (port of indpensim.m); simulate_iter()
                        returns a SampleStream with a live .control handle
  ode/rhs.py          - 33-state ODE right-hand side (port of indpensim_ode.m)
  control/
    controller.py     - port of fctrl_indpensim.m (PID + SBC + faults + Raman PAA loop)
    pid.py            - port of PIDSimple3.m
    history.py        - per-channel batch trajectory container
  pat/
    raman.py          - simulated Raman spectrum (port of Raman_Sim.m)
    substrate.py      - PLS-based PAA prediction (port of Substrate_prediction.m)
    pls_model.py      - PAA_PLS_model.mat loader
  io/
    parameters.py     - 105-element parameter vector
    initial_conditions.py  - loader for MATLAB-captured initial conditions
  streaming/
    sample.py         - Sample dataclass + StreamConfig (raman/lab cadence)
    pacing.py         - fast | fixed_interval | accelerated pacers
    uns.py            - UNS topic builder + tag/unit conversion
    mqtt_runner.py    - paho-mqtt publisher + CLI
  recipe/
    types.py          - Phase/Recipe/SetpointProfile/TransitionTrigger
    executor.py       - stateful RecipeExecutor (transitions, phase log,
                        pause/resume/advance_phase/abort for live control)
    legacy.py         - legacy_sbc_recipe() — regression anchor
    io.py             - JSON round-trip
  ui/                 - optional Streamlit recipe studio (needs [ui] extra)
    streamlit_app.py  - landing page
    pages/01_authoring.py
    pages/02_visualize.py
    state.py          - session state + pure form-dict helpers
    widgets.py        - reusable phase / trigger / schedule editors
    rendering.py      - plotly timeline figure builder
  validation/
    playback.py       - replay a captured batch with MATLAB inputs
docs/
  state_vector.md     - 33-state glossary (Y(N) MATLAB <-> y[N-1] Python)
  parameters.md       - 105-parameter catalog
  pls_model.md        - PLS coefficient interpretation
  matlab_reference_capture.md  - how to capture MATLAB reference data
scripts/
  matlab_*.m          - MATLAB capture scripts
tests/                - 656 tests (states, controller, ODE, PLS, streaming,
                        multi-seed validation, end-to-end, recipe parity, UI)
```

## Validation

Tests compare against MATLAB-captured trajectories from the original simulator.
Three layers: ODE-only playback, single-seed end-to-end, and a 12-config
multi-seed suite covering each fault branch, the Raman closed-loop PAA
controller, and variable-length batches.

End-to-end batch trajectory matches MATLAB on the validated configs within
tight per-channel bounds (mean peak-normalized error <1% on most channels).
See [VALIDATION.md](VALIDATION.md) for methodology, the solver-tolerance
decision (production uses `rtol=1e-3` for speed; validation uses `rtol=1e-6`),
per-channel thresholds, and known limits.

Run the full suite:

```bash
pytest
```

## Faithful port notes

Three published-source quirks in the MATLAB are reproduced verbatim, not silently
fixed:

- `fctrl_indpensim.m:45` — `ph_err1` is missing the `pH_sp -` term (typo)
- `fctrl_indpensim.m:147` — heating-branch PID for Fh uses Fc history as `u_prev`
- `fctrl_indpensim.m:148-149` — dead write of `Fc=0` immediately overwritten

See the docstring in `indpensim/control/controller.py` for context.
