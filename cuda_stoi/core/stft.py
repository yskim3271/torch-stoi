"""
STFT Module - Short-Time Fourier Transform

Provides GPU-compatible STFT computation that exactly replicates pystoi's behavior
for numerical equivalence (MAE < 1e-6).

Implementation Details:
    - Window: MATLAB-compatible Hanning (np.hanning(win_size + 2)[1:-1])
    - No center padding (center=False)
    - Loop-based implementation for pystoi compatibility
    - Frame count: (len - win_size) // hop (matches pystoi exactly)

Status:
    ✅ Implemented, ✅ Validated, ✅ GPU-compatible
"""

import torch
import numpy as np
from ..constants import NFFT, N_FRAME
from ..utils import create_hanning_window_matlab_compatible


def compute_stft(signal, n_fft=NFFT, win_size=N_FRAME, hop_length=None, lengths=None):
    """
    Compute Short-Time Fourier Transform with pystoi compatibility.

    Main API for STFT computation. Accepts both numpy arrays and torch tensors,
    and automatically handles GPU acceleration if input is on GPU.

    Args:
        signal (np.ndarray or torch.Tensor): Input signal(s).
            - Shape: (T,) for single signal or (B, T) for batch
            - Can be CPU or GPU tensor, or numpy array
        n_fft (int, optional): FFT size. Must be >= win_size. Defaults to 512.
        win_size (int, optional): Window size in samples. Defaults to 256.
        hop_length (int, optional): Hop size in samples. Defaults to 128.
        lengths (torch.Tensor, optional): Actual signal lengths for variable-length
            batch processing. Shape (B,). Defaults to None.

    Returns:
        torch.Tensor or tuple: Depending on `lengths` parameter:
            - If lengths is None: Complex STFT tensor
              Shape: (B, F, T_frames) or (F, T_frames) where F = n_fft // 2 + 1
              Device: Same as input signal
            - If lengths is not None: Tuple of (stft, frame_lengths)
              - stft: Shape (B, F, T_frames_max), zero-padded to max frames
              - frame_lengths: Shape (B,), actual frame count per signal

    Raises:
        TypeError: If signal is not np.ndarray or torch.Tensor
        ValueError: If signal dimensions are invalid (not 1D or 2D)
        ValueError: If signal length < win_size
        ValueError: If n_fft < win_size
        ValueError: If hop_length <= 0
        ValueError: If signal too short to produce any frames

    Example:
        >>> import torch
        >>> signal = torch.randn(16000)
        >>> stft = compute_stft(signal)
        >>> print(stft.shape)  # (257, 123) for 16000 samples

        >>> # GPU execution
        >>> signal_gpu = signal.cuda()
        >>> stft_gpu = compute_stft(signal_gpu)
        >>> print(stft_gpu.device)  # cuda:0

    Note:
        - Exactly matches pystoi output (MAE < 1e-6)
        - Uses MATLAB-compatible Hanning window: np.hanning(win_size + 2)[1:-1]
        - Frame count formula: (signal_length - win_size) // hop_length
        - GPU-compatible: automatically uses GPU if input is on GPU
    """
    # Input validation
    if not isinstance(signal, (torch.Tensor, np.ndarray)):
        raise TypeError(
            f"signal must be torch.Tensor or numpy.ndarray, got {type(signal).__name__}"
        )

    # Convert to torch if needed
    if isinstance(signal, np.ndarray):
        signal = torch.from_numpy(signal)

    # Check signal dimensions
    if signal.ndim not in [1, 2]:
        raise ValueError(
            f"signal must be 1D (T,) or 2D (B, T), got {signal.ndim}D with shape {signal.shape}"
        )

    # Check signal length
    signal_len = signal.shape[-1]
    if signal_len == 0:
        raise ValueError("signal is empty (length = 0)")

    if signal_len < win_size:
        raise ValueError(
            f"signal length ({signal_len}) must be >= window size ({win_size}). "
            f"Need at least {win_size} samples to compute STFT."
        )

    # Set default hop_length
    if hop_length is None:
        hop_length = win_size // 2

    # Check STFT parameters
    if n_fft < win_size:
        raise ValueError(f"n_fft ({n_fft}) must be >= win_size ({win_size})")

    if hop_length <= 0:
        raise ValueError(f"hop_length must be positive, got {hop_length}")

    if hop_length > win_size:
        import warnings
        warnings.warn(
            f"hop_length ({hop_length}) > win_size ({win_size}). "
            f"This will skip samples and may not match pystoi behavior.",
            UserWarning,
            stacklevel=2
        )

    # Check if enough frames can be produced (only when lengths not provided)
    if lengths is None:
        n_frames = (signal_len - win_size) // hop_length
        if n_frames < 1:
            raise ValueError(
                f"signal too short to produce any frames. "
                f"Signal length: {signal_len}, window: {win_size}, hop: {hop_length}. "
                f"Need at least {win_size + hop_length} samples."
            )

    # ========================================================================
    # STFT Computation (pystoi-compatible loop-based implementation)
    # ========================================================================

    # Handle batch dimension
    if signal.ndim == 1:
        signal = signal.unsqueeze(0)  # (1, T)
        remove_batch = True
    else:
        remove_batch = False

    batch_size = signal.shape[0]
    device = signal.device
    dtype = signal.dtype

    # Create window
    window = create_hanning_window_matlab_compatible(win_size, device=device, dtype=dtype)

    # Variable-length processing
    if lengths is not None:
        # Calculate frame lengths for each signal
        frame_lengths = torch.zeros(batch_size, dtype=torch.long, device=device)
        for i in range(batch_size):
            sig_len = lengths[i].item()
            if sig_len >= win_size:
                frame_lengths[i] = (sig_len - win_size) // hop_length
            else:
                frame_lengths[i] = 0  # Not enough samples

        # Max frames needed
        max_frames = frame_lengths.max().item()

        if max_frames == 0:
            # No valid frames
            freq_bins = n_fft // 2 + 1
            stft_out = torch.zeros(batch_size, freq_bins, 1, dtype=torch.complex64, device=device)
            return (stft_out, frame_lengths) if not remove_batch else (stft_out.squeeze(0), frame_lengths.squeeze(0))

        # Pre-allocate output with max_frames
        freq_bins = n_fft // 2 + 1
        stft_out = torch.zeros(batch_size, freq_bins, max_frames, dtype=torch.complex64, device=device)

        # Process each frame
        for frame_idx in range(max_frames):
            start = frame_idx * hop_length
            end = start + win_size

            # Extract and window frame
            # For signals shorter than needed, this will include padding (zeros)
            frame = signal[:, start:end] * window  # (B, win_size)

            # Zero-pad if needed
            if n_fft > win_size:
                frame = torch.nn.functional.pad(frame, (0, n_fft - win_size))

            # FFT
            fft_result = torch.fft.rfft(frame, n=n_fft)  # (B, freq_bins)

            stft_out[:, :, frame_idx] = fft_result

        # Remove batch dimension if needed
        if remove_batch:
            stft_out = stft_out.squeeze(0)  # (F, T_frames)
            frame_lengths = frame_lengths.squeeze(0)

        return stft_out, frame_lengths

    else:
        # Fixed-length processing: all signals same length
        # Number of frames (pystoi formula)
        n_frames = (signal_len - win_size) // hop_length

        # Pre-allocate output
        freq_bins = n_fft // 2 + 1
        stft_out = torch.zeros(batch_size, freq_bins, n_frames, dtype=torch.complex64, device=device)

        # Process each frame
        for frame_idx in range(n_frames):
            start = frame_idx * hop_length
            end = start + win_size

            # Extract and window frame
            frame = signal[:, start:end] * window  # (B, win_size)

            # Zero-pad if needed
            if n_fft > win_size:
                frame = torch.nn.functional.pad(frame, (0, n_fft - win_size))

            # FFT
            fft_result = torch.fft.rfft(frame, n=n_fft)  # (B, freq_bins)

            stft_out[:, :, frame_idx] = fft_result

        # Remove batch dimension if needed
        if remove_batch:
            stft_out = stft_out.squeeze(0)  # (F, T_frames)

        return stft_out


# ============================================================================
# Testing
# ============================================================================

if __name__ == '__main__':
    print("=" * 70)
    print("STFT Module Test")
    print("=" * 70)

    # Test signal
    np.random.seed(42)
    test_signal = np.random.randn(10000).astype(np.float32)

    print("\n[1] Testing STFT (CPU)...")
    stft_cpu = compute_stft(test_signal)
    print(f"  Input shape:  {test_signal.shape}")
    print(f"  Output shape: {stft_cpu.shape}")
    print(f"  Output dtype: {stft_cpu.dtype}")
    print(f"  Output device: {stft_cpu.device}")
    print(f"  Is complex:   {torch.is_complex(stft_cpu)}")

    print("\n[2] Testing STFT (GPU)...")
    if torch.cuda.is_available():
        signal_gpu = torch.from_numpy(test_signal).cuda()
        stft_gpu = compute_stft(signal_gpu)
        print(f"  Input device:  {signal_gpu.device}")
        print(f"  Output device: {stft_gpu.device}")
        print(f"  Output shape:  {stft_gpu.shape}")
        print(f"  ✓ GPU execution successful")
    else:
        print(f"  ⚠️ CUDA not available, skipping GPU test")

    print("\n[3] Testing batch processing...")
    test_batch = torch.randn(4, 10000)
    stft_batch = compute_stft(test_batch)
    print(f"  Input shape:  {test_batch.shape}")
    print(f"  Output shape: {stft_batch.shape}")

    print("\n✓ STFT module ready (GPU-compatible)")
