"""
Spectral Decomposer
===================
Feature engineering node that decomposes ``velocity_in`` into
low-frequency (mean flow + large eddies) and high-frequency (turbulent
fluctuations) components along the temporal axis.

Two decomposition methods are supported:

fft
    1-D FFT along the T_in axis per spatial point and component.
    Frequencies below ``cutoff_freq`` (fraction of Nyquist, 0–1) are
    kept for the low-frequency component; the remainder forms the
    high-frequency component via inverse FFT.

wavelet
    Discrete Wavelet Transform (DWT) using PyWavelets.
    ``wavelet_levels`` levels of decomposition; approximation
    coefficients → low-freq reconstruction, detail coefficients →
    high-freq reconstruction.

Outputs
-------
vel_low_freq   : (T_in, N, 3)  – slow / mean-flow component
vel_high_freq  : (T_in, N, 3)  – turbulent fluctuations

Both tensors are added to the sample dict / output dict alongside the
original ``velocity_in`` (which is preserved).

Hyperparameters
---------------
method          : "fft" | "wavelet"  (default "fft")
cutoff_freq     : float – fraction of Nyquist (0–1) for FFT split  (default 0.2)
wavelet         : str   – PyWavelets wavelet name                   (default "db4")
wavelet_levels  : int   – DWT decomposition levels                   (default 2)
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


class SpectralDecomposer:
    """
    Decompose temporal velocity fields into low- and high-frequency bands.

    In **lazy multi-sample** mode the node registers itself into
    ``fe_pipeline`` so the GFF applies it per sample on demand.
    """

    def __init__(
        self,
        hyperparams: dict[str, Any] | None = None,
        seed: int | None = None,
    ) -> None:
        hp = hyperparams or {}
        # Accept both the unique frontend key and the short alias
        self._method: str = str(hp.get("spectral_method", hp.get("method", "fft")))
        self._cutoff_freq: float = float(hp.get("cutoff_freq", 0.2))
        self._wavelet: str = str(hp.get("wavelet", "db4"))
        self._wavelet_levels: int = int(hp.get("wavelet_levels", 2))

    # ── pipeline interface ───────────────────────────────────────────────

    def execute(self, inputs: dict[str, Any]) -> dict[str, Any]:
        samples = inputs.get("samples")
        if samples is not None and isinstance(samples, list) and len(samples) > 0:
            return self._execute_multi(inputs)
        return self._execute_single(inputs)

    def _execute_single(self, inputs: dict[str, Any]) -> dict[str, Any]:
        vel_in = inputs.get("velocity_in")
        if vel_in is None:
            logger.warning("SpectralDecomposer: no velocity_in – pass-through.")
            return {**inputs}
        vel_in = np.asarray(vel_in, dtype=np.float32)
        low, high = self._decompose(vel_in)
        result = {**inputs}
        result["vel_low_freq"] = low
        result["vel_high_freq"] = high
        logger.info(
            "SpectralDecomposer[single]: method=%s  shape=%s",
            self._method, vel_in.shape,
        )
        return result

    def _execute_multi(self, inputs: dict[str, Any]) -> dict[str, Any]:
        samples = inputs["samples"]
        lazy = "filepath" in samples[0] and "velocity_in" not in samples[0]

        if lazy:
            if "fe_pipeline" not in inputs:
                inputs["fe_pipeline"] = []
            inputs["fe_pipeline"].append(self)
            logger.info(
                "SpectralDecomposer[lazy]: registered for %d samples.", len(samples),
            )
            return {**inputs}

        for s in samples:
            self.process_sample(s)
        return {**inputs}

    # ── lazy per-sample processing (called by GFF) ───────────────────────

    def process_sample(self, sample_data: dict[str, Any]) -> None:
        """Compute vel_low_freq and vel_high_freq in-place."""
        vel_in = sample_data.get("velocity_in")
        if vel_in is None:
            return
        vel_in = np.asarray(vel_in, dtype=np.float32)
        low, high = self._decompose(vel_in)
        sample_data["vel_low_freq"] = low
        sample_data["vel_high_freq"] = high

    # ── core decomposition ───────────────────────────────────────────────

    def _decompose(
        self, vel_in: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Parameters
        ----------
        vel_in : (T_in, N, 3)

        Returns
        -------
        (low_freq, high_freq) – both (T_in, N, 3) float32
        """
        if self._method == "fft":
            return self._decompose_fft(vel_in)
        if self._method == "wavelet":
            return self._decompose_wavelet(vel_in)
        raise ValueError(f"SpectralDecomposer: unknown method '{self._method}'.")

    def _decompose_fft(
        self, vel_in: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """FFT along T_in axis, zero-out high/low frequency bands."""
        T = vel_in.shape[0]
        cutoff_idx = max(1, int(self._cutoff_freq * (T // 2 + 1)))

        # Transform along axis 0 (time)
        spec = np.fft.rfft(vel_in, axis=0)  # (T//2+1, N, 3)

        # Low-freq: keep [0, cutoff_idx)
        spec_low = np.zeros_like(spec)
        spec_low[:cutoff_idx] = spec[:cutoff_idx]

        # High-freq: keep [cutoff_idx, ...]
        spec_high = spec.copy()
        spec_high[:cutoff_idx] = 0.0

        low = np.fft.irfft(spec_low, n=T, axis=0).astype(np.float32)
        high = np.fft.irfft(spec_high, n=T, axis=0).astype(np.float32)
        return low, high

    def _decompose_wavelet(
        self, vel_in: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Multi-level DWT along T_in; approx → low, details → high."""
        try:
            import pywt
        except ImportError:
            logger.warning(
                "PyWavelets not installed – falling back to FFT decomposition."
            )
            return self._decompose_fft(vel_in)

        T, N, C = vel_in.shape
        low = np.zeros_like(vel_in)
        high = np.zeros_like(vel_in)

        for n in range(N):
            for c in range(C):
                sig = vel_in[:, n, c].astype(np.float64)
                coeffs = pywt.wavedec(sig, self._wavelet, level=self._wavelet_levels)
                # Reconstruct low-freq from approximation only
                coeffs_low = [coeffs[0]] + [np.zeros_like(d) for d in coeffs[1:]]
                # Reconstruct high-freq from detail coefficients only
                coeffs_high = [np.zeros_like(coeffs[0])] + list(coeffs[1:])
                low_sig = pywt.waverec(coeffs_low, self._wavelet)[:T]
                high_sig = pywt.waverec(coeffs_high, self._wavelet)[:T]
                low[:, n, c] = low_sig.astype(np.float32)
                high[:, n, c] = high_sig.astype(np.float32)

        return low.astype(np.float32), high.astype(np.float32)
