"""
CUDA-STOI: GPU-accelerated Short-Time Objective Intelligibility

Fully batched GPU-accelerated implementation of the STOI metric.

Main API:
    >>> from cuda_stoi import stoi
    >>> import numpy as np
    >>>
    >>> # Single signal
    >>> clean = np.random.randn(16000)
    >>> degraded = clean + np.random.randn(16000) * 0.1
    >>> score = stoi(clean, degraded, 16000)
    >>> print(f"STOI: {score:.4f}")
    >>>
    >>> # Batch processing
    >>> import torch
    >>> clean = torch.randn(4, 16000)
    >>> degraded = clean + torch.randn(4, 16000) * 0.1
    >>> scores = stoi(clean, degraded, 16000)
    >>> print(scores.shape)  # (4,)

Features:
    - Fully batched GPU acceleration
    - Automatic device detection
    - Single signal and batch processing
    - Multiple sampling rates (8kHz - 48kHz)
    - Numerically equivalent to pystoi (MAE < 1e-6)
    - 2-7x speedup over sequential processing
"""

__version__ = "0.3.0"  # Refactored batched-only version
__status__ = "Batched processing only - Refactored and streamlined"

# Main API - single entry point
from .stoi import stoi

# Core modules (for advanced users)
from .core import (
    resample_to_10k,
    compute_stft,
    remove_silent_frames_batched,
    apply_octave_bands,
    get_octave_band_matrix,
    compute_stoi_batched,
    create_segments_batched,
    normalize_and_clip,
)

# Utility functions (for advanced users)
from .utils import (
    validate_and_convert_tensors,
    ensure_batch_dimension,
    create_hanning_window_matlab_compatible,
    get_device_and_dtype,
    convert_output_format,
)

__all__ = [
    # Main API
    'stoi',
    # Core modules (batched-only)
    'resample_to_10k',
    'compute_stft',
    'remove_silent_frames_batched',
    'apply_octave_bands',
    'get_octave_band_matrix',
    'compute_stoi_batched',
    'create_segments_batched',
    'normalize_and_clip',
    # Utility functions
    'validate_and_convert_tensors',
    'ensure_batch_dimension',
    'create_hanning_window_matlab_compatible',
    'get_device_and_dtype',
    'convert_output_format',
]
