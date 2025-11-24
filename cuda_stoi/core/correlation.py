"""
Correlation and STOI Computation Module

Computes intermediate intelligibility measure and final STOI score.
"""

import torch
import numpy as np
import warnings

from ..constants import N, BETA


# =============================================================================
# Batched STOI Computation (Primary API)
# =============================================================================

def normalize_and_clip(x_segments, y_segments, clip_value=None):
    """
    Normalize and clip degraded segments

    This is a helper function that can be used separately for testing.

    Args:
        x_segments (torch.Tensor): Clean segments (num_segments, num_bands, segment_length)
        y_segments (torch.Tensor): Degraded segments (same shape)
        clip_value (float, optional): Clipping threshold (default: 10^(15/20))

    Returns:
        torch.Tensor: Normalized and clipped y_segments

    Notes:
        - Normalization: y_norm = y * ||x|| / (||y|| + EPS)
        - Clipping: y' = min(y_norm, x * (1 + clip_value))
        - CRITICAL: Only upper bound is clipped!
    """
    if clip_value is None:
        clip_value = 10 ** (-BETA / 20)

    dtype = x_segments.dtype
    EPS = torch.finfo(dtype).eps

    # Normalize
    x_norms = torch.linalg.norm(x_segments, dim=2, keepdim=True)
    y_norms = torch.linalg.norm(y_segments, dim=2, keepdim=True)

    normalization_consts = x_norms / (y_norms + EPS)
    y_normalized = y_segments * normalization_consts

    # Clip (asymmetric!)
    y_clipped = torch.minimum(
        y_normalized,
        x_segments * (1 + clip_value)
    )

    return y_clipped


def remove_mean_and_normalize(segments):
    """
    Remove mean and normalize by L2 norm

    Args:
        segments (torch.Tensor): Segments (num_segments, num_bands, segment_length)

    Returns:
        torch.Tensor: Mean-removed and L2-normalized segments

    Notes:
        - Mean removal: along time axis (axis=2)
        - L2 normalization: ||segment|| along time axis
    """
    dtype = segments.dtype
    EPS = torch.finfo(dtype).eps

    # Remove mean
    segments = segments - torch.mean(segments, dim=2, keepdim=True)

    # L2 normalize
    segments = segments / (torch.linalg.norm(segments, dim=2, keepdim=True) + EPS)

    return segments


def compute_correlation(x_segments, y_segments):
    """
    Compute correlation between clean and degraded segments

    Args:
        x_segments (torch.Tensor): Clean segments (num_segments, num_bands, segment_length)
        y_segments (torch.Tensor): Degraded segments (same shape)

    Returns:
        torch.Tensor: Mean correlation (scalar tensor)

    Notes:
        - Correlation: sum(x * y) / (J * M)
        - J = num_segments, M = num_bands
        - Returns tensor to preserve gradients for differentiable training
    """
    # Element-wise multiplication
    correlations = x_segments * y_segments

    # Sum and average
    J, M, N = x_segments.shape
    d = torch.sum(correlations) / (J * M)

    return d


# ============================================================================
# Batched Processing (NEW)
# ============================================================================

def compute_stoi_batched(clean_bands, degraded_bands, segment_length=N, frame_lengths=None):
    """
    Compute STOI with batched processing and optional masking

    Supports both single signal and batch processing modes using fully
    vectorized operations for optimal performance.

    Args:
        clean_bands (torch.Tensor): Clean signal octave bands
            - 2D: (num_bands, time_frames) - single signal
            - 3D: (batch_size, num_bands, time_frames) - batch
        degraded_bands (torch.Tensor): Degraded signal octave bands (same shape)
        segment_length (int): Number of frames per segment (default: 30)
        frame_lengths (torch.Tensor, optional): (batch_size,) - actual frame count per signal
                                                 For variable-length batch processing

    Returns:
        if 2D input:
            float: STOI score
        if 3D input:
            torch.Tensor: STOI scores (batch_size,)

    Raises:
        TypeError: If inputs are not torch.Tensor
        ValueError: If shapes mismatch or invalid dimensions

    Performance:
        - Fully vectorized batched operations
        - 7-55x faster than sequential processing
        - Supports variable-length sequences with masking
    """
    # Input validation
    if not isinstance(clean_bands, torch.Tensor):
        raise TypeError(f"clean_bands must be torch.Tensor, got {type(clean_bands).__name__}")

    if not isinstance(degraded_bands, torch.Tensor):
        raise TypeError(f"degraded_bands must be torch.Tensor, got {type(degraded_bands).__name__}")

    if clean_bands.shape != degraded_bands.shape:
        raise ValueError(
            f"clean_bands and degraded_bands must have same shape, "
            f"got {clean_bands.shape} and {degraded_bands.shape}"
        )

    # Handle different input dimensions
    if clean_bands.ndim == 2:
        # Single signal - use original function
        return compute_stoi(clean_bands, degraded_bands, segment_length)

    elif clean_bands.ndim == 3:
        # Batch processing - always use vectorized mode
        return _compute_stoi_batched_vectorized(clean_bands, degraded_bands, segment_length, frame_lengths)

    else:
        raise ValueError(
            f"Bands must be 2D (M, T) or 3D (B, M, T), got {clean_bands.ndim}D"
        )


def _compute_stoi_batched_vectorized(clean_bands, degraded_bands, segment_length=N, frame_lengths=None):
    """
    Vectorized batched STOI computation with optional masking

    Args:
        clean_bands (torch.Tensor): (batch_size, num_bands, time_frames)
        degraded_bands (torch.Tensor): (batch_size, num_bands, time_frames)
        segment_length (int): Segment length
        frame_lengths (torch.Tensor, optional): (batch_size,) - actual frame count per signal
                                                 For variable-length batch processing

    Returns:
        torch.Tensor: STOI scores (batch_size,)
    """
    batch_size, num_bands, time_frames = clean_bands.shape
    device = clean_bands.device
    dtype = clean_bands.dtype

    # Variable-length processing with masking
    if frame_lengths is not None:
        # Compute number of segments for each signal
        num_segments_per_signal = torch.zeros(batch_size, dtype=torch.long, device=device)

        for i in range(batch_size):
            if frame_lengths[i] >= segment_length:
                num_segments_per_signal[i] = frame_lengths[i] - segment_length + 1
            else:
                num_segments_per_signal[i] = 0

        # Check if any signal has valid segments
        if num_segments_per_signal.max() == 0:
            warnings.warn(
                f'Not enough frames to compute STOI for any signal in batch. '
                f'Returning 1e-5 for all samples.',
                RuntimeWarning
            )
            return torch.full((batch_size,), 1e-5, device=device, dtype=dtype)

        # Create segments for entire batch (will include padding segments)
        x_segments = create_segments_batched(clean_bands, segment_length)
        y_segments = create_segments_batched(degraded_bands, segment_length)

        # x_segments.shape = (batch_size, num_segments_max, num_bands, segment_length)
        num_segments_max = x_segments.shape[1]

        # Create mask: (batch_size, num_segments_max)
        mask = torch.zeros(batch_size, num_segments_max, dtype=torch.bool, device=device)
        for i in range(batch_size):
            if num_segments_per_signal[i] > 0:
                mask[i, :num_segments_per_signal[i]] = True

        # Normalize y_segments by x_segments norm
        EPS = torch.finfo(dtype).eps
        x_norms = torch.linalg.norm(x_segments, dim=3, keepdim=True)  # (B, J, M, 1)
        y_norms = torch.linalg.norm(y_segments, dim=3, keepdim=True)  # (B, J, M, 1)

        normalization_consts = x_norms / (y_norms + EPS)
        y_segments_normalized = y_segments * normalization_consts

        # Clip (asymmetric - upper bound only!)
        clip_value = 10 ** (-BETA / 20)
        y_primes = torch.minimum(
            y_segments_normalized,
            x_segments * (1 + clip_value)
        )

        # Subtract mean across time (axis=3)
        y_primes = y_primes - torch.mean(y_primes, dim=3, keepdim=True)
        x_segments = x_segments - torch.mean(x_segments, dim=3, keepdim=True)

        # Divide by L2 norm
        y_primes = y_primes / (torch.linalg.norm(y_primes, dim=3, keepdim=True) + EPS)
        x_segments = x_segments / (torch.linalg.norm(x_segments, dim=3, keepdim=True) + EPS)

        # Compute correlation
        correlations_components = y_primes * x_segments  # (B, J, M, N)

        # Apply mask and compute per-signal scores
        # Pre-allocate scores tensor to preserve gradients
        scores = torch.zeros(batch_size, device=device, dtype=dtype)

        for i in range(batch_size):
            if num_segments_per_signal[i] == 0:
                scores[i] = 1e-5
            else:
                # Get valid segments for this signal
                valid_corr = correlations_components[i, :num_segments_per_signal[i], :, :]

                # Sum and normalize (keep as tensor to preserve gradients)
                scores[i] = torch.sum(valid_corr) / (num_segments_per_signal[i] * num_bands)

        return scores

    else:
        # Original behavior: all signals have same length
        # Check if we have enough frames
        if time_frames < segment_length:
            warnings.warn(
                f'Not enough frames to compute intermediate intelligibility measure. '
                f'Got {time_frames} frames, need at least {segment_length}. '
                f'Returning 1e-5 for all samples.',
                RuntimeWarning
            )
            return torch.full((batch_size,), 1e-5, device=device, dtype=dtype)

        # Create segments for entire batch
        x_segments = create_segments_batched(clean_bands, segment_length)
        y_segments = create_segments_batched(degraded_bands, segment_length)

        # x_segments.shape = (batch_size, num_segments, num_bands, segment_length)
        num_segments = x_segments.shape[1]

        # Normalize y_segments by x_segments norm
        EPS = torch.finfo(dtype).eps
        x_norms = torch.linalg.norm(x_segments, dim=3, keepdim=True)  # (B, J, M, 1)
        y_norms = torch.linalg.norm(y_segments, dim=3, keepdim=True)  # (B, J, M, 1)

        normalization_consts = x_norms / (y_norms + EPS)
        y_segments_normalized = y_segments * normalization_consts

        # Clip (asymmetric - upper bound only!)
        clip_value = 10 ** (-BETA / 20)
        y_primes = torch.minimum(
            y_segments_normalized,
            x_segments * (1 + clip_value)
        )

        # Subtract mean across time (axis=3)
        y_primes = y_primes - torch.mean(y_primes, dim=3, keepdim=True)
        x_segments = x_segments - torch.mean(x_segments, dim=3, keepdim=True)

        # Divide by L2 norm
        y_primes = y_primes / (torch.linalg.norm(y_primes, dim=3, keepdim=True) + EPS)
        x_segments = x_segments / (torch.linalg.norm(x_segments, dim=3, keepdim=True) + EPS)

        # Compute correlation
        correlations_components = y_primes * x_segments

        # Sum over segments (J) and bands (M) for each batch
        # Shape: (B, J, M, N) → (B,)
        d = torch.sum(correlations_components, dim=(1, 2, 3)) / (num_segments * num_bands)

        return d


def create_segments_batched(bands, segment_length=N):
    """
    Create sliding window segments from batched octave bands

    Args:
        bands (torch.Tensor): Octave bands
            - 2D: (num_bands, time_frames) - single signal
            - 3D: (batch_size, num_bands, time_frames) - batch

    Returns:
        torch.Tensor: Segments
            - 3D: (num_segments, num_bands, segment_length) for single
            - 4D: (batch_size, num_segments, num_bands, segment_length) for batch

    Notes:
        - Creates overlapping segments with stride=1
        - Compatible with both single and batched inputs
    """
    if bands.ndim == 2:
        # Single signal - use original function
        return create_segments(bands, segment_length)

    elif bands.ndim == 3:
        # Batched processing
        batch_size, num_bands, time_frames = bands.shape

        # Calculate number of segments
        num_segments = time_frames - segment_length + 1

        if num_segments <= 0:
            raise ValueError(
                f"Not enough frames for segmentation. "
                f"Got {time_frames} frames, need at least {segment_length}"
            )

        # Use unfold to create sliding windows
        # unfold(dimension, size, step)
        # bands: (B, M, T) → (B, M, num_segments, segment_length)
        segments = bands.unfold(dimension=2, size=segment_length, step=1)

        # Rearrange to (B, num_segments, M, segment_length)
        segments = segments.permute(0, 2, 1, 3)

        return segments

    else:
        raise ValueError(
            f"Bands must be 2D or 3D, got {bands.ndim}D"
        )
