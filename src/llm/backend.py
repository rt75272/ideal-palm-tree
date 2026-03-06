"""Array backend selection for CPU (NumPy) or GPU (CuPy).

This module centralises array-library selection so the rest of the codebase can
write backend-agnostic tensor code.
"""

from __future__ import annotations

import os
from typing import Any

import numpy as np

try:
    import cupy as cp  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover - depends on local environment
    cp = None


_AVAILABLE_DEVICE_VALUES = {"auto", "cpu", "gpu"}
_DEVICE: str = "cpu"
_xp: Any = np
_BACKEND_WARNING: str | None = None
_CUPY_RUNTIME_PROBED: bool = False
_CUPY_RUNTIME_OK: bool = False
_CUPY_RUNTIME_ERROR: str | None = None


def _probe_cupy_runtime() -> tuple[bool, str | None]:
    """Check whether CuPy can execute kernels in this environment."""
    global _CUPY_RUNTIME_PROBED, _CUPY_RUNTIME_OK, _CUPY_RUNTIME_ERROR

    if _CUPY_RUNTIME_PROBED:
        return _CUPY_RUNTIME_OK, _CUPY_RUNTIME_ERROR

    _CUPY_RUNTIME_PROBED = True
    if cp is None:
        _CUPY_RUNTIME_OK = False
        _CUPY_RUNTIME_ERROR = "CuPy is not installed."
        return _CUPY_RUNTIME_OK, _CUPY_RUNTIME_ERROR

    try:
        # Trigger an actual kernel path to catch missing CUDA runtime pieces
        # (for example libnvrtc.so) early.
        _ = cp.arange(1, dtype=cp.int32)
        _CUPY_RUNTIME_OK = True
        _CUPY_RUNTIME_ERROR = None
    except Exception as exc:  # pragma: no cover - environment-specific
        _CUPY_RUNTIME_OK = False
        _CUPY_RUNTIME_ERROR = str(exc)

    return _CUPY_RUNTIME_OK, _CUPY_RUNTIME_ERROR


def set_backend(device: str = "auto") -> str:
    """Set the active array backend.

    Args:
        device: One of ``auto``, ``cpu``, ``gpu``.

    Returns:
        The selected backend name (``cpu`` or ``gpu``).
    """
    global _BACKEND_WARNING, _DEVICE, _xp
    _BACKEND_WARNING = None

    selected = device.lower()
    if selected not in _AVAILABLE_DEVICE_VALUES:
        raise ValueError(
            f"Invalid device '{device}'. Expected one of: "
            f"{sorted(_AVAILABLE_DEVICE_VALUES)}"
        )

    if selected == "cpu":
        _DEVICE = "cpu"
        _xp = np
        return _DEVICE

    if selected == "gpu":
        if cp is None:
            raise RuntimeError(
                "GPU requested but CuPy is not installed. Install a CuPy package "
                "matching your CUDA version, e.g. 'cupy-cuda12x'."
            )
        ok, error = _probe_cupy_runtime()
        if not ok:
            raise RuntimeError(
                "GPU requested but CuPy runtime is not usable in this "
                "environment. Ensure CUDA runtime libraries are installed "
                f"(original error: {error})."
            )
        _DEVICE = "gpu"
        _xp = cp
        return _DEVICE

    # auto mode
    if cp is not None:
        ok, error = _probe_cupy_runtime()
        if ok:
            _DEVICE = "gpu"
            _xp = cp
        else:
            _DEVICE = "cpu"
            _xp = np
            _BACKEND_WARNING = (
                "CuPy was found but GPU runtime is unavailable; falling back "
                f"to CPU. Details: {error}"
            )
    else:
        _DEVICE = "cpu"
        _xp = np
    return _DEVICE


def get_device() -> str:
    """Return the current active device (``cpu`` or ``gpu``)."""
    return _DEVICE


def using_gpu() -> bool:
    """Whether the active backend is GPU/CuPy."""
    return _DEVICE == "gpu"


def is_gpu_available() -> bool:
    """Whether CuPy is importable in this environment."""
    return cp is not None


def is_gpu_usable() -> bool:
    """Whether CuPy is importable and runtime-usable."""
    ok, _ = _probe_cupy_runtime()
    return ok


def get_backend_warning() -> str | None:
    """Return warning generated during backend selection, if any."""
    return _BACKEND_WARNING


def array_module() -> Any:
    """Return the active array module (``numpy`` or ``cupy``)."""
    return _xp


def to_numpy(array: Any) -> np.ndarray:
    """Convert a backend array/scalar to a NumPy array."""
    if cp is not None and isinstance(array, cp.ndarray):
        return cp.asnumpy(array)
    return np.asarray(array)


def to_scalar(value: Any) -> float:
    """Convert a backend scalar to a Python float."""
    if cp is not None and isinstance(value, cp.ndarray):
        return float(cp.asnumpy(value).item())
    return float(value)


def seed(seed_value: int) -> None:
    """Seed the active backend RNG (and NumPy for deterministic CPU utilities)."""
    np.random.seed(seed_value)
    if _DEVICE == "gpu" and cp is not None:
        cp.random.seed(seed_value)


# Resolve backend at import time. Set LLM_DEVICE=cpu|gpu|auto to override.
set_backend(os.getenv("LLM_DEVICE", "auto"))
