"""
Benchmark Utilities

Helper functions for STOI benchmarking:
- Noise generation and SNR application
- Audio loading and preprocessing
- Timing and statistical analysis
"""

import time
import torch
import numpy as np
from typing import Tuple, Optional, Callable, Any, List, Dict
from functools import wraps
import json
from pathlib import Path
import random
import torchaudio

# Import HuggingFace data loaders
from tests.data_loader import load_librispeech_batch, load_demand_noise_batch

# Constants
DEFAULT_FS = 16000
DEFAULT_SEED = 42


def add_noise_at_snr(clean: torch.Tensor, noise: torch.Tensor, snr_db: float) -> torch.Tensor:
    """
    Add noise to clean signal at specified SNR level.

    Args:
        clean: Clean signal tensor, shape (samples,) or (batch, samples)
        noise: Noise signal tensor, same shape as clean
        snr_db: Desired Signal-to-Noise Ratio in dB

    Returns:
        Noisy signal with specified SNR
    """
    # Calculate signal power
    signal_power = torch.mean(clean ** 2, dim=-1, keepdim=True)

    # Calculate noise power
    noise_power = torch.mean(noise ** 2, dim=-1, keepdim=True)

    # Calculate required noise scaling factor
    snr_linear = 10 ** (snr_db / 10)
    noise_scale = torch.sqrt(signal_power / (noise_power * snr_linear))

    # Scale noise and add to signal
    noisy = clean + noise_scale * noise

    return noisy


def generate_white_noise(shape: Tuple[int, ...], device: str = 'cpu') -> torch.Tensor:
    """
    Generate white Gaussian noise.

    Args:
        shape: Shape of the noise tensor
        device: Device to create tensor on ('cpu' or 'cuda')

    Returns:
        White noise tensor
    """
    return torch.randn(shape, device=device)


def adjust_signal_length(signal: torch.Tensor, target_length: int) -> torch.Tensor:
    """
    Adjust signal length by truncating or padding.

    Args:
        signal: Input signal, shape (samples,) or (batch, samples)
        target_length: Desired length in samples

    Returns:
        Signal adjusted to target_length
    """
    current_length = signal.shape[-1]

    if current_length == target_length:
        return signal
    elif current_length > target_length:
        return signal[..., :target_length]
    else:
        padding = (0, target_length - current_length)
        return torch.nn.functional.pad(signal, padding, mode='constant', value=0)


def load_audio_dataset(
    data_dir: Optional[Path] = None,
    num_samples: int = 1,
    target_length: int = 16000,
    target_fs: int = DEFAULT_FS,
    seed: Optional[int] = None
) -> torch.Tensor:
    """
    Load audio samples from HuggingFace dataset with uniform length for batching.

    Note: data_dir parameter is kept for backward compatibility but is ignored.
          Data is now loaded from HuggingFace (LibriSpeech test-clean).

    Args:
        data_dir: Deprecated, kept for backward compatibility
        num_samples: Number of samples to load
        target_length: Target length in samples (for uniform batching)
        target_fs: Target sampling rate
        seed: Random seed for reproducibility

    Returns:
        Tensor of audio samples, shape (num_samples, target_length)
    """
    return load_librispeech_batch(
        num_samples=num_samples,
        target_length=target_length,
        target_fs=target_fs,
        seed=seed
    )


def load_noise_samples(
    noise_dir: Optional[Path] = None,
    num_samples: int = 1,
    target_length: int = 16000,
    target_fs: int = DEFAULT_FS,
    seed: Optional[int] = None
) -> torch.Tensor:
    """
    Load environmental noise samples from HuggingFace dataset (ESC-50).

    Note: noise_dir parameter is kept for backward compatibility but is ignored.
          Data is now loaded from HuggingFace (ESC-50 dataset).

    Args:
        noise_dir: Deprecated, kept for backward compatibility
        num_samples: Number of noise samples to load
        target_length: Target length in samples
        target_fs: Target sampling rate
        seed: Random seed for reproducibility

    Returns:
        Tensor of noise samples, shape (num_samples, target_length)
    """
    return load_demand_noise_batch(
        num_samples=num_samples,
        target_length=target_length,
        target_fs=target_fs,
        seed=seed
    )


def create_realistic_noisy_speech(
    clean_batch: torch.Tensor,
    noise_dir: Optional[Path] = None,
    snr_db: float = 10.0,
    target_fs: int = DEFAULT_FS,
    seed: Optional[int] = None
) -> torch.Tensor:
    """
    Create realistic noisy speech by mixing clean speech with environmental noise.

    Note: noise_dir parameter is kept for backward compatibility but is ignored.
          Noise is now loaded from HuggingFace (ESC-50 dataset).

    Args:
        clean_batch: Clean speech (batch, samples)
        noise_dir: Deprecated, kept for backward compatibility
        snr_db: Desired Signal-to-Noise Ratio in dB
        target_fs: Target sampling rate
        seed: Random seed for reproducibility

    Returns:
        Noisy speech (batch, samples)
    """
    batch_size, signal_length = clean_batch.shape

    # Load noise samples from HuggingFace
    noise_batch = load_noise_samples(
        noise_dir=None,  # Ignored, uses HuggingFace
        num_samples=batch_size,
        target_length=signal_length,
        target_fs=target_fs,
        seed=seed
    )

    # Mix at target SNR
    noisy_batch = add_noise_at_snr(clean_batch, noise_batch, snr_db)

    return noisy_batch


class Timer:
    """
    Context manager for timing code blocks.

    Usage:
        with Timer() as t:
            # code to time
            pass
        print(f"Elapsed: {t.elapsed:.3f} seconds")
    """

    def __init__(self):
        self.start_time = None
        self.end_time = None
        self.elapsed = None

    def __enter__(self):
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, *args):
        self.end_time = time.perf_counter()
        self.elapsed = self.end_time - self.start_time


def time_function(func: Callable, *args, num_runs: int = 1, warmup: int = 0,
                 device_sync: bool = False, **kwargs) -> Tuple[Any, float]:
    """
    Time a function execution.

    Args:
        func: Function to time
        *args: Positional arguments to pass to func
        num_runs: Number of runs to average over
        warmup: Number of warmup runs (not counted)
        device_sync: If True, synchronize CUDA device before timing
        **kwargs: Keyword arguments to pass to func

    Returns:
        Tuple of (function result, average elapsed time in seconds)
    """
    # Warmup runs
    for _ in range(warmup):
        _ = func(*args, **kwargs)

    if device_sync and torch.cuda.is_available():
        torch.cuda.synchronize()

    # Timed runs
    start_time = time.perf_counter()
    for _ in range(num_runs):
        result = func(*args, **kwargs)

    if device_sync and torch.cuda.is_available():
        torch.cuda.synchronize()

    end_time = time.perf_counter()
    avg_time = (end_time - start_time) / num_runs

    return result, avg_time


def compute_statistics(values: torch.Tensor) -> Dict[str, float]:
    """
    Compute basic statistics for a tensor of values.

    Args:
        values: Tensor of values

    Returns:
        Dictionary with statistics (mean, std, min, max, median)
    """
    return {
        'mean': float(torch.mean(values)),
        'std': float(torch.std(values)),
        'min': float(torch.min(values)),
        'max': float(torch.max(values)),
        'median': float(torch.median(values))
    }


def calculate_mae_mse(predictions: torch.Tensor, targets: torch.Tensor) -> Tuple[float, float]:
    """
    Calculate Mean Absolute Error and Mean Squared Error.

    Args:
        predictions: Predicted values
        targets: Target values

    Returns:
        Tuple of (MAE, MSE)
    """
    mae = torch.mean(torch.abs(predictions - targets)).item()
    mse = torch.mean((predictions - targets) ** 2).item()
    return mae, mse


def save_results(results: Dict, output_file: Path) -> None:
    """
    Save benchmark results to JSON file.

    Args:
        results: Dictionary of results
        output_file: Path to output JSON file
    """
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"Results saved to: {output_file}")


def load_results(input_file: Path) -> Dict:
    """
    Load benchmark results from JSON file.

    Args:
        input_file: Path to input JSON file

    Returns:
        Dictionary of results
    """
    with open(input_file, 'r') as f:
        return json.load(f)
