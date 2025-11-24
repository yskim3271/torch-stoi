"""
CUDA-STOI Main Module

Complete Short-Time Objective Intelligibility (STOI) implementation
with GPU acceleration and batched processing.
"""

import torch

from .core.resampling import resample_to_10k
from .core.vad import remove_silent_frames_batched
from .core.stft import compute_stft
from .core.octave_bands import apply_octave_bands
from .core.correlation import compute_stoi_batched
from .constants import FS, DYN_RANGE, N_FRAME, N
from .utils import validate_and_convert_tensors, ensure_batch_dimension


def stoi(clean, degraded, fs_sig, extended=False):
    """
    Compute Short-Time Objective Intelligibility (STOI)

    A measure of speech intelligibility for time-frequency weighted noisy speech.
    Higher scores indicate better intelligibility.

    This function uses fully batched GPU-accelerated processing for optimal performance.
    Both single signals and batches are supported.

    Args:
        clean (np.ndarray or torch.Tensor): Clean reference speech signal(s)
            - Shape: (samples,) for single signal
            - Shape: (batch, samples) for batch processing
        degraded (np.ndarray or torch.Tensor): Degraded/processed speech signal(s)
            - Must have same shape as clean
        fs_sig (int): Sampling frequency of input signals in Hz
            - Typical values: 8000, 16000, 44100, 48000
        extended (bool): Use extended STOI (default: False)
            - Not implemented yet, raises NotImplementedError

    Returns:
        torch.Tensor: STOI score(s) in range [0, 1]
            - Shape (): Scalar tensor for single signal input (1D)
            - Shape (batch,): For batch input (2D)

    Raises:
        TypeError: If inputs are not np.ndarray or torch.Tensor
        ValueError: If clean and degraded have different shapes
        NotImplementedError: If extended=True (not yet supported)

    Examples:
        >>> import numpy as np
        >>> import torch
        >>>
        >>> # Single signal (returns scalar tensor)
        >>> clean = np.random.randn(16000)
        >>> degraded = clean + np.random.randn(16000) * 0.1
        >>> score = stoi(clean, degraded, 16000)
        >>> print(f"STOI: {score.item():.4f}")  # Use .item() to get Python float
        >>> print(score.shape)  # torch.Size([])
        >>>
        >>> # Batch processing
        >>> clean = torch.randn(4, 16000)
        >>> degraded = clean + torch.randn(4, 16000) * 0.1
        >>> scores = stoi(clean, degraded, 16000)
        >>> print(scores.shape)  # torch.Size([4])

    Performance:
        - Fully batched GPU processing
        - 2-7x speedup compared to sequential processing
        - Supports both CPU and CUDA devices
        - Numerical equivalence with pystoi (MAE < 1e-6)

    Notes:
        - All processing happens on the same device as input
        - GPU is recommended for best performance with large batches
        - VAD padding may add zeros to match longest signal in batch
        - For 1D input, output is converted to Python float

    References:
        [1] C.H.Taal, R.C.Hendriks, R.Heusdens, J.Jensen
            'A Short-Time Objective Intelligibility Measure for Time-Frequency
            Weighted Noisy Speech', ICASSP 2010, Texas, Dallas.
        [2] C.H.Taal, R.C.Hendriks, R.Heusdens, J.Jensen
            'An Algorithm for Intelligibility Prediction of Time-Frequency
            Weighted Noisy Speech', IEEE Transactions on Audio, Speech,
            and Language Processing, 2011.
    """
    # Extended STOI not implemented yet
    if extended:
        raise NotImplementedError(
            "Extended STOI is not implemented yet. Use extended=False."
        )

    # Input validation and conversion using utility functions
    clean, degraded, is_numpy, device = validate_and_convert_tensors(clean, degraded)

    # Handle single signal vs batch
    clean, squeeze_output = ensure_batch_dimension(clean)
    degraded, _ = ensure_batch_dimension(degraded)

    # Step 1: Resample to 10 kHz (if needed)
    if fs_sig != FS:
        clean = resample_to_10k(clean, original_fs=fs_sig)
        degraded = resample_to_10k(degraded, original_fs=fs_sig)

    # Move to GPU if device was specified
    if device is not None and device.type == 'cuda':
        clean = clean.to(device)
        degraded = degraded.to(device)

    # Step 2: Remove silent frames (VAD) - always use padding mode
    clean_vad, degraded_vad, vad_lengths = remove_silent_frames_batched(
        clean, degraded,
        dyn_range=DYN_RANGE,
        framelen=N_FRAME,
        hop=N_FRAME // 2
    )

    # Step 3: Compute STFT with variable lengths
    clean_stft, stft_frame_lengths = compute_stft(
        clean_vad,
        lengths=vad_lengths
    )
    degraded_stft, _ = compute_stft(
        degraded_vad,
        lengths=vad_lengths
    )

    # Step 4: Apply octave band filtering
    clean_bands = apply_octave_bands(clean_stft)
    degraded_bands = apply_octave_bands(degraded_stft)

    # Step 5: Compute STOI from octave bands (batched)
    scores = compute_stoi_batched(
        clean_bands, degraded_bands,
        segment_length=N,
        frame_lengths=stft_frame_lengths
    )

    # Return format (always return tensor for gradient support)
    if squeeze_output:
        return scores.squeeze(0)
    return scores
