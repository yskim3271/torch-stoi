"""
CUDA-STOI Core Modules

All core processing modules for STOI computation using fully batched operations.

Modules:
  ✅ Resampling      - CPU-only (pystoi compatibility)
  ✅ STFT            - GPU-compatible batched processing
  ✅ VAD             - GPU-compatible batched processing with padding
  ✅ Octave Bands    - GPU-compatible batched processing
  ✅ Correlation     - GPU-compatible batched processing
"""

# Core modules - all batched
from .resampling import resample_to_10k
from .stft import compute_stft
from .vad import remove_silent_frames_batched
from .octave_bands import apply_octave_bands, get_octave_band_matrix
from .correlation import (
    compute_stoi_batched,
    create_segments_batched,
    normalize_and_clip
)

__all__ = [
    # Primary processing functions
    'resample_to_10k',                # Resampling (torchaudio-based)
    'compute_stft',                   # STFT computation
    'remove_silent_frames_batched',   # VAD processing
    'apply_octave_bands',             # Octave band filtering
    'compute_stoi_batched',           # STOI computation (batched-only)
    # Utility functions
    'get_octave_band_matrix',         # Get OBM matrix
    'create_segments_batched',        # Create segments for correlation
    'normalize_and_clip',             # Normalization for testing
]
