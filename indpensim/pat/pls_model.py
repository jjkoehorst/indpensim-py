"""PLS regression model for PAA concentration from Raman spectra.

Ports `Substrate_prediction.m` (the PLS prediction step). Loads coefficients
from `PAA_PLS_model.mat` once and exposes a `predict_raw(spectrum)` that
returns a single PAA estimate per spectrum.

The 3-point temporal smoothing (averaging the last 2 + current PLS outputs
when sample index > 20) lives in the calling loop, not here — see
`Substrate_prediction.m:13-15`.

See `docs/pls_model.md` for the full derivation of the 212-element feature
vector and the boundary-handling note on `savgol_filter`.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import scipy.io
from scipy.signal import savgol_filter


# Hard-coded in Substrate_prediction.m:11. The .mat file ships 10 candidate
# coefficient rows (one per LV count); production model uses 4 LVs.
DEFAULT_NO_LV: int = 4

# Wavenumber-bin windows used to slice the differenced, SG-smoothed spectrum.
# MATLAB indices [350:500 800:860] are 1-based on a post-`diff` array.
# Python equivalent: 0-based, exclusive upper bound — produces 151 + 61 = 212.
_FEATURE_WINDOWS = (slice(349, 500), slice(799, 860))
EXPECTED_FEATURE_LEN = 151 + 61   # = 212; must match coefficient row width

# Lives under indpensim/data/ (inside the package) so it ships with the
# installed wheel — see the matching note on _REFERENCE_SPECTRA_PATH in
# simulation.py.
_DEFAULT_MAT_PATH = Path(__file__).resolve().parents[1] / "data" / "PAA_PLS_model.mat"


@dataclass(frozen=True)
class PAAPLSModel:
    """PLS coefficients for PAA prediction from Raman spectrum.

    Attributes:
        coefficients: shape (n_lv_options, n_features) = (10, 212).
            Row i (0-based) is the regression vector for an (i+1)-LV model.
        no_lv: number of latent variables to use; row index = no_lv - 1.
    """
    coefficients: np.ndarray
    no_lv: int = DEFAULT_NO_LV

    def __post_init__(self) -> None:
        if self.coefficients.ndim != 2:
            raise ValueError(f"coefficients must be 2D, got shape {self.coefficients.shape}")
        if self.coefficients.shape[1] != EXPECTED_FEATURE_LEN:
            raise ValueError(
                f"coefficients axis 1 must be {EXPECTED_FEATURE_LEN}, "
                f"got {self.coefficients.shape[1]}"
            )
        if not (1 <= self.no_lv <= self.coefficients.shape[0]):
            raise ValueError(
                f"no_lv must be in [1, {self.coefficients.shape[0]}], got {self.no_lv}"
            )

    @classmethod
    def load(cls, mat_path: Path | str | None = None, no_lv: int = DEFAULT_NO_LV) -> "PAAPLSModel":
        """Load coefficients from PAA_PLS_model.mat (default: indpensim/data/PAA_PLS_model.mat)."""
        path = Path(mat_path) if mat_path is not None else _DEFAULT_MAT_PATH
        raw = scipy.io.loadmat(str(path), squeeze_me=False, struct_as_record=True)
        if "b" not in raw:
            raise KeyError(f"{path} has no 'b' variable; got {list(raw.keys())}")
        b = np.asarray(raw["b"], dtype=float)
        return cls(coefficients=b, no_lv=no_lv)

    @property
    def beta(self) -> np.ndarray:
        """The active regression vector — shape (212,)."""
        return self.coefficients[self.no_lv - 1]

    def features(self, spectrum: np.ndarray) -> np.ndarray:
        """Build the 212-element feature vector from a raw Raman spectrum.

        Mirrors Substrate_prediction.m lines 7-10:
            sg = sgolayfilt(spectrum, polyorder=2, window=5)
            sg_d = diff(sg)
            features = sg_d[[350:500 800:860]]   # MATLAB 1-based on post-diff array

        scipy's savgol_filter handles boundaries via polynomial extrapolation
        ('interp' mode). MATLAB's sgolayfilt uses a different scheme. With
        window=5 and a 2200-bin spectrum, only the first 2 and last 2 samples
        differ — neither falls inside the [350:500] or [800:860] slices, so
        the impact on the feature vector is zero.
        """
        spectrum = np.asarray(spectrum, dtype=float).ravel()
        if spectrum.size < 1000:
            raise ValueError(
                f"spectrum too short for windows up to bin 860, got len={spectrum.size}"
            )
        sg = savgol_filter(spectrum, window_length=5, polyorder=2)
        sg_d = np.diff(sg)
        feats = np.concatenate([sg_d[w] for w in _FEATURE_WINDOWS])
        if feats.size != EXPECTED_FEATURE_LEN:
            raise RuntimeError(
                f"feature vector length {feats.size} != expected {EXPECTED_FEATURE_LEN}"
            )
        return feats

    def predict_raw(self, spectrum: np.ndarray) -> float:
        """Predict PAA concentration (mg/L) from one raw Raman spectrum.

        Returns the *unsmoothed* PLS output. The 3-point temporal moving
        average (active for sample index > 20) is the caller's responsibility.
        """
        return float(self.features(spectrum) @ self.beta)
