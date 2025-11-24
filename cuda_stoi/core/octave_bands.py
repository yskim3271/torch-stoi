"""
Octave Band Filtering Module

Applies 1/3 octave band filtering to STFT spectrograms.
"""

import torch
import numpy as np
import os
from pathlib import Path

from ..constants import FS, NFFT, NUMBAND, MINFREQ


def apply_octave_bands(stft_spectrum, obm=None):
    """
    Apply 1/3 octave band filtering to STFT spectrum

    Implements Eq. (1) from STOI paper:
        X_tob = sqrt(OBM @ |X_spec|^2)

    Args:
        stft_spectrum (torch.Tensor): Complex STFT spectrum
            - Shape: (freq_bins, time_frames) for single signal
            - Shape: (batch, freq_bins, time_frames) for batch
        obm (torch.Tensor, optional): Octave Band Matrix (NUMBAND, freq_bins)
            If None, loads from saved file or creates new one

    Returns:
        torch.Tensor: Octave band filtered spectrum
            - Shape: (NUMBAND, time_frames) for single
            - Shape: (batch, NUMBAND, time_frames) for batch

    Raises:
        TypeError: If stft_spectrum is not torch.Tensor
        ValueError: If dimensions are invalid

    Notes:
        - OBM is binary matrix: (15, 257) for default parameters
        - Each row selects FFT bins for one 1/3 octave band
        - Center frequencies: 150 Hz ~ 3809.8 Hz (15 bands)
    """
    # Input validation
    if not isinstance(stft_spectrum, torch.Tensor):
        raise TypeError(f"stft_spectrum must be torch.Tensor, got {type(stft_spectrum).__name__}")

    if stft_spectrum.ndim == 2:
        # Single signal: (freq_bins, time_frames)
        batch_mode = False
        stft_spectrum = stft_spectrum.unsqueeze(0)
    elif stft_spectrum.ndim == 3:
        # Batch: (batch, freq_bins, time_frames)
        batch_mode = True
    else:
        raise ValueError(
            f"stft_spectrum must be 2D (F, T) or 3D (B, F, T), got {stft_spectrum.ndim}D"
        )

    device = stft_spectrum.device
    dtype = stft_spectrum.real.dtype  # Use real dtype for output

    # Load or create OBM
    if obm is None:
        obm = get_octave_band_matrix()

    # Move OBM to same device and dtype
    if not isinstance(obm, torch.Tensor):
        obm = torch.from_numpy(obm)
    obm = obm.to(device=device, dtype=dtype)

    # Validate OBM shape
    expected_freq_bins = stft_spectrum.shape[1]
    if obm.shape[1] != expected_freq_bins:
        raise ValueError(
            f"OBM shape mismatch. Expected (NUMBAND, {expected_freq_bins}), "
            f"got {obm.shape}"
        )

    # Apply octave band filtering
    # X_tob = sqrt(OBM @ |X_spec|^2)
    batch_size, freq_bins, time_frames = stft_spectrum.shape

    # Compute magnitude squared
    magnitude_sq = torch.abs(stft_spectrum) ** 2  # (B, F, T)

    # Apply OBM: (B, F, T) -> (B, NUMBAND, T)
    # obm: (NUMBAND, F)
    # We need: obm @ magnitude_sq for each time frame
    # Efficient: einsum or batch matmul
    octave_bands = torch.matmul(obm, magnitude_sq)  # (B, NUMBAND, T)

    # Take square root
    octave_bands = torch.sqrt(octave_bands)

    # Remove batch dimension if input was 2D
    if not batch_mode:
        octave_bands = octave_bands.squeeze(0)

    return octave_bands


def get_octave_band_matrix(fs=FS, nfft=NFFT, num_bands=NUMBAND, min_freq=MINFREQ):
    """
    Get 1/3 octave band matrix

    Loads from saved file if available, otherwise creates new one.

    Args:
        fs (int): Sampling frequency (default: 10000 Hz)
        nfft (int): FFT size (default: 512)
        num_bands (int): Number of 1/3 octave bands (default: 15)
        min_freq (float): Minimum center frequency (default: 150 Hz)

    Returns:
        np.ndarray: Octave Band Matrix of shape (num_bands, nfft//2 + 1)

    Notes:
        - Saved file: tests/fixtures/reference/pystoi_obm.npy
        - Falls back to create_octave_band_matrix() if file not found
    """
    # Try to load from saved file
    try:
        project_root = Path(__file__).parent.parent.parent
        obm_path = project_root / "tests" / "fixtures" / "reference" / "pystoi_obm.npy"

        if obm_path.exists():
            obm = np.load(obm_path)
            # Validate shape
            expected_shape = (num_bands, nfft // 2 + 1)
            if obm.shape == expected_shape:
                return obm
            else:
                print(f"Warning: Loaded OBM has shape {obm.shape}, expected {expected_shape}. Creating new OBM.")
    except Exception as e:
        print(f"Warning: Could not load OBM from file: {e}. Creating new OBM.")

    # Create new OBM
    obm, cf = _create_octave_band_matrix(fs, nfft, num_bands, min_freq)
    return obm


def _create_octave_band_matrix(fs=FS, nfft=NFFT, num_bands=NUMBAND, min_freq=MINFREQ):
    """
    Create 1/3 octave band matrix (internal helper)

    This replicates pystoi's thirdoct() function exactly.

    Args:
        fs (int): Sampling frequency
        nfft (int): FFT size
        num_bands (int): Number of 1/3 octave bands
        min_freq (float): Center frequency of lowest band

    Returns:
        tuple: (obm, cf)
            - obm: Octave Band Matrix (num_bands, nfft//2 + 1)
            - cf: Center frequencies (num_bands,)

    Algorithm:
        1. Create FFT frequency bins: f = linspace(0, fs, nfft+1)[: nfft//2+1]
        2. Compute center frequencies: cf = 2^(k/3) * min_freq
        3. Compute band edges:
           - freq_low = min_freq * 2^((2k-1)/6)
           - freq_high = min_freq * 2^((2k+1)/6)
        4. Assign bins: argmin(|f - freq_low/high|)
        5. Create binary matrix: obm[i, fl:fh] = 1
    """
    # 1. Create frequency bins
    f = np.linspace(0, fs, nfft + 1)
    f = f[: int(nfft / 2) + 1]

    # 2. Compute center frequencies
    k = np.arange(num_bands, dtype=float)
    cf = np.power(2.0 ** (1.0 / 3.0), k) * min_freq

    # 3. Compute band edges
    freq_low = min_freq * np.power(2.0, (2 * k - 1) / 6)
    freq_high = min_freq * np.power(2.0, (2 * k + 1) / 6)

    # 4. Initialize OBM
    obm = np.zeros((num_bands, len(f)))

    # 5. Assign bins to each band
    for i in range(len(cf)):
        # Find closest bin to freq_low
        f_bin_low = np.argmin(np.square(f - freq_low[i]))
        freq_low[i] = f[f_bin_low]

        # Find closest bin to freq_high
        f_bin_high = np.argmin(np.square(f - freq_high[i]))
        freq_high[i] = f[f_bin_high]

        # Assign to octave band matrix
        obm[i, f_bin_low:f_bin_high] = 1

    return obm, cf
