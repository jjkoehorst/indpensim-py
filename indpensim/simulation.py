"""End-to-end batch simulation — port of indpensim.m main loop.

Provides two entry points sharing a single source of truth (``_SimulationRun``):

  - ``simulate(spec, ...)``  — runs the whole batch and returns a
    ``SimulationResult`` (collect-all). Used by tests and the offline driver.
  - ``simulate_iter(spec, ...)`` — returns a ``SampleStream`` yielding one
    ``Sample`` per integration step (streaming). Used by MQTT/file/callback
    sinks. When the batch has an attached Recipe, ``SampleStream.control``
    exposes the live ``RecipeExecutor`` so another thread can
    pause/resume/advance_phase/abort an in-flight batch — see
    ``SampleStream``'s docstring.

Both run the same per-step physics. Per-sample pH and Q conversions are
applied in the ``Sample`` so streaming consumers see plant-readable units;
the post-loop bulk conversions on history/states preserve backward
compatibility with the original ``SimulationResult`` shape.

Mirrors `indpensim.m:107-379` exactly (per-step floors, DO2 stability rule,
inhibition mu_X tweak, OUR/CER derived calcs, post-loop pH and Q
conversions).

What's NOT yet implemented:
  - Off-line measurements: skipped (recording-only, no feedback in the
    physics — surfaced via ``Sample.offline`` in streaming mode only).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp

from indpensim.control.controller import controller_step
from indpensim.control.history import BatchHistory, IndustrialData
from indpensim.io.initial_conditions import CapturedBatch, ControlFlags
from indpensim.io.parameters import Parameters
from indpensim.ode.rhs import N_STATES, rhs
from indpensim.pat.pls_model import PAAPLSModel
from indpensim.pat.raman import build_reference, simulate_spectrum
from indpensim.pat.substrate import predict_and_store
from indpensim.recipe.executor import RecipeExecutor
from indpensim.recipe.types import PhaseState
from indpensim.streaming.sample import Sample, StreamConfig

# Lives under indpensim/data/ (inside the package), not a repo-root data/ —
# hatchling only ships files under `packages`, so a repo-root path here would
# silently 404 once installed as a pip dependency rather than run from a checkout.
_REFERENCE_SPECTRA_PATH = Path(__file__).resolve().parent / "data" / "reference_Specra.txt"


# Map state index → BatchHistory channel name (for post-step storage).
# Mirrors indpensim.m:248-320.
_STATE_INDEX_TO_CHANNEL: tuple[str, ...] = (
    "S", "DO2", "O2", "P", "V", "Wt", "pH", "T",            # 0-7
    "Q", "Viscosity", "Culture_age",                          # 8-10
    "a0", "a1", "a3", "a4",                                   # 11-14
    "n0", "n1", "n2", "n3", "n4", "n5", "n6", "n7", "n8", "n9",  # 15-24
    "nm", "phi0",                                             # 25-26
    "CO2outgas", "CO2_d",                                     # 27-28
    "PAA", "NH3",                                             # 29-30
    "mu_P_calc", "mu_X_calc",                                 # 31-32
)


@dataclass
class SimulationResult:
    t: np.ndarray                  # shape (N+1,)
    history: BatchHistory          # all channels, MATLAB 1-based indexing
    states: np.ndarray             # shape (N+1, 33), endpoint state per sample
    pH_trajectory: np.ndarray      # shape (N+1,), -log10(y[6])
    raman_spectra: np.ndarray | None = None
    """Raman intensities, shape (N+1, 2200), MATLAB 1-based; None if Raman_spec=0."""


def _build_initial_state(cap: CapturedBatch) -> np.ndarray:
    """Mirror of indpensim.m:131 (the k==1 x00 vector)."""
    ic = cap.initial_conditions
    y = np.zeros(N_STATES)
    y[0] = ic.S
    y[1] = ic.DO2
    y[2] = ic.O2
    y[3] = ic.P
    y[4] = ic.V
    y[5] = ic.Wt
    y[6] = 10.0 ** (-ic.pH)        # pH → [H+] (mirror of indpensim.m:105)
    y[7] = ic.T
    y[8] = 0.0                      # Q
    y[9] = 4.0                      # Viscosity init literal
    y[10] = ic.Culture_age
    y[11] = ic.a0
    y[12] = ic.a1
    y[13] = ic.a3
    y[14] = ic.a4
    # vacuoles n0..n9 + nm + phi0 = zeros(1,12)
    y[27] = ic.CO2outgas
    y[28] = 0.0                     # CO2_d
    y[29] = ic.PAA
    y[30] = ic.NH3
    y[31] = 0.0                     # mu_P_calc
    y[32] = 0.0                     # mu_X_calc
    return y


def _apply_matlab_floors(y: np.ndarray) -> None:
    """Mirror of indpensim.m:216-220 and :252-257."""
    mask = y[:31] <= 0.0
    y[:31][mask] = 0.001
    if y[1] < 2.0:
        y[1] = 1.0


def _build_inputs(u, cap: CapturedBatch, k: int, h_ode: float) -> np.ndarray:
    """Build the 26-element controller-input vector for ``rhs``.

    Mirror of indpensim.m:178. Pulls disturbances from the captured trajectory.
    """
    dist = cap.disturbances.at(k - 1)    # disturbances are 0-indexed by sample
    inp = np.zeros(26)
    inp[0]  = cap.control_flags.Inhib
    inp[1]  = u.Fs
    inp[2]  = u.Fg
    inp[3]  = u.RPM
    inp[4]  = u.Fc
    inp[5]  = u.Fh
    inp[6]  = u.Fb
    inp[7]  = u.Fa
    inp[8]  = h_ode
    inp[9]  = u.Fw
    inp[10] = u.pressure
    inp[11] = u.viscosity
    inp[12] = u.Fremoved
    inp[13] = u.Fpaa
    inp[14] = u.Foil
    inp[15] = u.NH3_shots
    inp[16] = cap.control_flags.Dis
    inp[17:25] = dist
    inp[25] = cap.control_flags.Vis
    return inp


def _calc_OUR_CER(history: BatchHistory, k: int) -> tuple[float, float]:
    """Mirror of indpensim.m:335-339."""
    O2_in = 0.204
    Fg = history.y("Fg", k)
    O2 = history.y("O2", k)
    CO2outgas = history.y("CO2outgas", k)
    OUR = (32 * Fg / 22.4) * (O2_in - O2 * (0.7902 / (1 - O2 - CO2outgas / 100)))
    CER = (44 * Fg / 22.4) * ((0.65 * CO2outgas / 100)
                              * (0.7902 / (1 - O2_in - CO2outgas / 100) - 0.0330))
    return float(OUR), float(CER)


# ============================================================================
# Shared run state (used by both simulate and simulate_iter)
# ============================================================================

class _SimulationRun:
    """Holds all per-batch state that ``simulate`` and ``simulate_iter`` share.

    The lifecycle is: ``__init__`` (set up arrays / RNG / Raman) → call
    ``step(k)`` for k=1..N (each call advances the integrator one ODE
    interval and returns a Sample) → call ``finalize()`` (apply post-loop
    conversions on history and states for backward-compat).
    """

    def __init__(
        self,
        cap: CapturedBatch,
        *,
        ctrl_flags: ControlFlags | None = None,
        rtol: float = 1e-3,
        atol: float = 1e-6,
        raman_rng: np.random.Generator | None = None,
        raman_noise_traj: np.ndarray | None = None,
        stream_config: StreamConfig | None = None,
    ) -> None:
        self.cap = cap
        self.ctrl_flags = cap.control_flags if ctrl_flags is None else ctrl_flags
        self.rtol = rtol
        self.atol = atol
        self.raman_noise_traj = raman_noise_traj
        self.stream_config = stream_config or StreamConfig()

        self.h = cap.h
        self.T = cap.T
        self.N = int(round(self.T / self.h))
        self.h_ode = self.h / 20.0
        self.t_grid = np.linspace(0.0, self.N * self.h, self.N + 1)

        # Parameter vector — mutable: inhibition logic rewrites par[1] in place.
        self.par = Parameters.default(
            mu_p=cap.initial_conditions.mup,
            mux_max=cap.initial_conditions.mux,
            alpha_kla=cap.initial_conditions.alpha_kla,
            N_conc_paa=cap.initial_conditions.N_conc_paa,
            PAA_c=cap.initial_conditions.PAA_c,
        ).to_legacy_par_vector().copy()

        self.history = BatchHistory.empty(self.N)
        self.states = np.zeros((self.N + 1, N_STATES))
        self.pH_trajectory = np.zeros(self.N + 1)

        # Initial state
        self.y_curr = _build_initial_state(cap)
        self.states[0] = self.y_curr.copy()

        # Pre-fill k=1 history slots that fctrl reads at first call.
        ic = cap.initial_conditions
        self.history.set("S", 1, ic.S)
        self.history.set("DO2", 1, ic.DO2)
        self.history.set("X", 1, ic.X)
        self.history.set("P", 1, ic.P)
        self.history.set("V", 1, ic.V)
        self.history.set("CO2outgas", 1, ic.CO2outgas)
        self.history.set("pH", 1, 10.0 ** (-ic.pH))
        self.history.set("T", 1, ic.T)

        self.Xd = IndustrialData()

        # ---- Raman pipeline setup
        self.raman_active = self.ctrl_flags.Raman_spec >= 1
        if self.raman_active:
            self.raman_ref = build_reference(_REFERENCE_SPECTRA_PATH)
            self.raman_spectra = np.zeros((self.N + 1, 2200))
            self.raman_rng = raman_rng or np.random.default_rng(0)
        else:
            self.raman_ref = None
            self.raman_spectra = None
            self.raman_rng = raman_rng
        self.pls = PAAPLSModel.load() if self.ctrl_flags.Raman_spec == 2 else None

        # Recipe path — optional. When cap.recipe is present, construct
        # an executor and thread it into every controller_step.
        self.recipe_exec: RecipeExecutor | None = (
            RecipeExecutor(recipe=cap.recipe, h=self.h) if cap.recipe is not None else None
        )

    # ------------------------------------------------------------------
    def step(self, k: int) -> Sample:
        """Advance integrator one ODE interval. Returns the user-facing Sample."""
        # ---- 1. Inhibition: degrade mu_X if 63+ of last 64 diffs are negative.
        if self.ctrl_flags.Inhib in (1, 2) and k > 65:
            mu_X_window = self.history.channels["mu_X_calc"][k - 65 : k]
            d = np.diff(mu_X_window)
            if int(np.sum(d < 0)) >= 63:
                self.par[1] = self.history.y("mu_X_calc", k - 1) * 5.0

        # ---- 2. Controller
        u = controller_step(
            self.history, self.Xd, k, self.h, self.T, self.ctrl_flags,
            recipe_exec=self.recipe_exec,
        )

        # Reset per-sample mu_P_calc / mu_X_calc accumulators
        self.y_curr[31] = 0.0
        self.y_curr[32] = 0.0

        # ---- 3. Build inputs and integrate
        inp = _build_inputs(u, self.cap, k, self.h_ode)
        sol = solve_ivp(
            rhs, [self.t_grid[k - 1], self.t_grid[k]], self.y_curr,
            args=(inp, self.par), method="BDF",
            rtol=self.rtol, atol=self.atol,
            t_eval=[self.t_grid[k]],
        )
        if not sol.success:
            raise RuntimeError(f"solve_ivp failed at sample {k}: {sol.message}")
        self.y_curr = sol.y[:, -1]
        _apply_matlab_floors(self.y_curr)
        self.states[k] = self.y_curr

        # ---- 4. Save controller outputs to history
        self.history.set("Fa", k, u.Fa)
        self.history.set("Fb", k, u.Fb)
        self.history.set("Fc", k, u.Fc)
        self.history.set("Fh", k, u.Fh)
        self.history.set("Fs", k, u.Fs)
        self.history.set("Fpaa", k, u.Fpaa)
        self.history.set("Fg", k, u.Fg)
        self.history.set("Foil", k, u.Foil)
        self.history.set("RPM", k, u.RPM)
        self.history.set("Fw", k, u.Fw)
        self.history.set("pressure", k, u.pressure)
        self.history.set("Fremoved", k, u.Fremoved)
        self.history.set("Fault_ref", k, u.Fault_ref)
        self.history.set("viscosity", k, u.viscosity)

        # ---- 5. Save all 33 state values to history
        for idx, channel in enumerate(_STATE_INDEX_TO_CHANNEL):
            self.history.set(channel, k, float(self.y_curr[idx]))

        # Total biomass X = a0 + a1 + a3 + a4
        self.history.set(
            "X", k,
            self.history.y("a0", k) + self.history.y("a1", k)
            + self.history.y("a3", k) + self.history.y("a4", k),
        )

        # ---- 6. OUR / CER
        OUR, CER = _calc_OUR_CER(self.history, k)
        self.history.set("OUR", k, OUR)
        self.history.set("CER", k, CER)

        # ---- 7. Raman
        raman_sample: list[float] | None = None
        if self.raman_active and k > 10:
            noise_k = self.raman_noise_traj[k] if self.raman_noise_traj is not None else None
            spec = simulate_spectrum(
                reference=self.raman_ref,
                P=self.history.y("P", k),
                X=self.history.y("X", k),
                viscosity=self.history.y("Viscosity", k),
                S=self.history.y("S", k),
                PAA=self.history.y("PAA", k),
                k=k, N_samples=self.N,
                rng=self.raman_rng, noise=noise_k,
            )
            self.raman_spectra[k] = spec

            # Raman_spec=2 closes the PAA control loop.
            if self.ctrl_flags.Raman_spec == 2 and k >= 2:
                j = k - 1
                if j > 10:
                    predict_and_store(
                        pls=self.pls, spectrum_at_j=self.raman_spectra[j],
                        paa_pred_history=self.history.channels["PAA_pred"], j=j,
                    )

            # Stream the spectrum on the configured cadence
            if k % self.stream_config.raman_every == 0:
                raman_sample = spec.tolist()

        # ---- 8. Build the Sample (per-step user-facing snapshot)
        return self._build_sample(k, u, raman_sample)

    # ------------------------------------------------------------------
    def _build_sample(self, k: int, u, raman_sample: list[float] | None) -> Sample:
        """Construct a Sample with plant-readable units (pH, Q converted)."""
        y = self.y_curr
        state = {
            "S": float(y[0]), "DO2": float(y[1]), "O2": float(y[2]),
            "P": float(y[3]), "V": float(y[4]), "Wt": float(y[5]),
            "pH": float(-np.log10(max(y[6], 1e-30))),     # converted
            "T": float(y[7]),
            "Q": float(y[8]) / 1000.0,                      # converted
            "Viscosity": float(y[9]), "Culture_age": float(y[10]),
            "a0": float(y[11]), "a1": float(y[12]),
            "a3": float(y[13]), "a4": float(y[14]),
            "X": float(y[11] + y[12] + y[13] + y[14]),
            "n0": float(y[15]), "n1": float(y[16]), "n2": float(y[17]),
            "n3": float(y[18]), "n4": float(y[19]), "n5": float(y[20]),
            "n6": float(y[21]), "n7": float(y[22]), "n8": float(y[23]),
            "n9": float(y[24]), "nm": float(y[25]), "phi0": float(y[26]),
            "CO2outgas": float(y[27]), "CO2_d": float(y[28]),
            "PAA": float(y[29]), "NH3": float(y[30]),
            "mu_P_calc": float(y[31]), "mu_X_calc": float(y[32]),
            "OUR": self.history.y("OUR", k),
            "CER": self.history.y("CER", k),
        }
        controls = {
            "Fg": u.Fg, "RPM": u.RPM, "Fs": u.Fs, "Fa": u.Fa, "Fb": u.Fb,
            "Fc": u.Fc, "Fh": u.Fh, "Fw": u.Fw, "pressure": u.pressure,
            "viscosity": u.viscosity, "Fremoved": u.Fremoved,
            "Fpaa": u.Fpaa, "Foil": u.Foil,
        }

        offline: dict[str, float] | None = None
        cfg = self.stream_config
        if k % cfg.lab_every == 0 and k > cfg.lab_delay_samples:
            j = k - cfg.lab_delay_samples
            offline = {
                "NH3": self.history.y("NH3", j),
                "P": self.history.y("P", j),
                "X": self.history.y("X", j),
                "PAA": self.history.y("PAA", j),
                "Viscosity": self.history.y("Viscosity", j),
            }

        phase: str | None = None
        phase_state: str | None = None
        phase_transitions: list[dict] | None = None
        if self.recipe_exec is not None:
            phase = self.recipe_exec.current_phase.name
            phase_state = self.recipe_exec.phase_state.value
            drained = self.recipe_exec.drain_new_transitions()
            if drained:
                phase_transitions = [
                    {
                        "from_phase": t.from_phase,
                        "to_phase": t.to_phase,
                        "at_k": t.at_k,
                        "at_time_h": t.at_time_h,
                        "reason": t.reason,
                    }
                    for t in drained
                ]

        return Sample(
            k=k, sim_time_h=float(self.t_grid[k]), wall_time_s=None,
            state=state, controls=controls,
            raman=raman_sample, offline=offline,
            phase=phase, phase_state=phase_state,
            phase_transitions=phase_transitions,
        )

    # ------------------------------------------------------------------
    def finalize(self) -> None:
        """Apply post-loop conversions to history/states for backward-compat
        with the original ``SimulationResult`` shape."""
        self.pH_trajectory = -np.log10(np.maximum(self.states[:, 6], 1e-30))
        self.states[:, 8] = self.states[:, 8] / 1000.0
        self.history.channels["pH"][:] = -np.log10(
            np.maximum(self.history.channels["pH"][:], 1e-30)
        )
        self.history.channels["Q"][:] = self.history.channels["Q"][:] / 1000.0


# ============================================================================
# Public API
# ============================================================================

def simulate(
    cap: CapturedBatch,
    *,
    ctrl_flags: ControlFlags | None = None,
    rtol: float = 1e-3,
    atol: float = 1e-6,
    raman_rng: np.random.Generator | None = None,
    raman_noise_traj: np.ndarray | None = None,
) -> SimulationResult:
    """Run a full batch through the Python controller + ODE.

    Args:
        cap: CapturedBatch from `load_captured_batch(...)` — supplies x0,
            randomized parameters, disturbance trajectories, and (by default)
            control flags.
        ctrl_flags: override `cap.control_flags` (e.g. to disable PRBS for
            reproducibility). Defaults to the captured flags.
        rtol/atol: solver tolerances (defaults match playback).
        raman_rng: rng for stochastic Raman noise.
        raman_noise_traj: optional (N+1, 2200) array of pre-captured noise
            vectors; overrides ``raman_rng`` when set.
    """
    run = _SimulationRun(
        cap, ctrl_flags=ctrl_flags, rtol=rtol, atol=atol,
        raman_rng=raman_rng, raman_noise_traj=raman_noise_traj,
    )
    for k in range(1, run.N + 1):
        run.step(k)
    run.finalize()
    return SimulationResult(
        t=run.t_grid, history=run.history, states=run.states,
        pH_trajectory=run.pH_trajectory, raman_spectra=run.raman_spectra,
    )


class SampleStream:
    """Iterator returned by ``simulate_iter``.

    Behaves exactly like the plain generator it replaces — drive it with
    ``for sample in stream:`` or ``list(stream)`` — but exposes ``.control``,
    the batch's live ``RecipeExecutor`` (``None`` when no Recipe is
    attached). Code on another thread can call
    ``stream.control.pause()/.resume()/.advance_phase()/.abort()`` on that
    handle while this thread is mid-iteration, to steer an in-flight batch.

    Once the executor's ``phase_state`` becomes ``ABORTED``, the stream
    yields the in-flight sample and then stops — it never reaches ``N``.
    """

    def __init__(self, run: "_SimulationRun") -> None:
        self._run = run
        self._k = 0
        self._done = False

    def __iter__(self) -> "SampleStream":
        return self

    def next(self) -> Sample:
        """Alias for ``next(stream)`` — advance one step and return its
        ``Sample``, for a manual ``while True: sample = stream.next()``
        loop (raises ``StopIteration`` at the end, same as the builtin).
        Prefer this over the ``for``/``list()`` form when you need to
        interleave your own analysis and ``.control`` calls between steps
        in a single thread, rather than reacting from a second thread.
        """
        return next(self)

    def __next__(self) -> Sample:
        if self._done:
            raise StopIteration
        self._k += 1
        if self._k > self._run.N:
            self._done = True
            self._run.finalize()
            raise StopIteration
        sample = self._run.step(self._k)
        exec_ = self._run.recipe_exec
        if exec_ is not None and exec_.phase_state is PhaseState.ABORTED:
            self._done = True
            self._run.finalize()
        return sample

    @property
    def control(self) -> RecipeExecutor | None:
        return self._run.recipe_exec


def simulate_iter(
    cap: CapturedBatch,
    *,
    ctrl_flags: ControlFlags | None = None,
    rtol: float = 1e-3,
    atol: float = 1e-6,
    raman_rng: np.random.Generator | None = None,
    raman_noise_traj: np.ndarray | None = None,
    stream_config: StreamConfig | None = None,
) -> SampleStream:
    """Yield one ``Sample`` per integration step.

    Same physics as ``simulate()``. Use this for streaming sinks
    (MQTT, jsonlines, callbacks). Per-sample pH and Q are converted to
    plant-readable units in the yielded Sample's state dict.

    Returns a ``SampleStream`` — iterate it like the generator it replaces,
    or hold onto it and use ``.control`` to steer a Recipe-driven batch
    while it's streaming (see ``SampleStream``).
    """
    run = _SimulationRun(
        cap, ctrl_flags=ctrl_flags, rtol=rtol, atol=atol,
        raman_rng=raman_rng, raman_noise_traj=raman_noise_traj,
        stream_config=stream_config,
    )
    return SampleStream(run)
