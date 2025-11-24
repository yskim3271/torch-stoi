"""
HuggingFace Dataset Loader

Loads speech and noise datasets from HuggingFace Hub for STOI benchmarking.
Replaces local file-based loading with streaming/cached HuggingFace datasets.
"""

import torch
import torchaudio
from typing import Optional, Tuple
from datasets import load_dataset
import random

# Dataset configurations
SPEECH_DATASET = "openslr/librispeech_asr"
SPEECH_CONFIG = "clean"
SPEECH_SPLIT = "test"

NOISE_DATASET = "ashraq/esc50"

DEFAULT_FS = 16000


def _resample_if_needed(waveform: torch.Tensor, orig_sr: int, target_sr: int) -> torch.Tensor:
    """
    Resample waveform if sampling rates don't match.

    Args:
        waveform: Input waveform (channels, samples)
        orig_sr: Original sampling rate
        target_sr: Target sampling rate

    Returns:
        Resampled waveform
    """
    if orig_sr != target_sr:
        resampler = torchaudio.transforms.Resample(orig_sr, target_sr)
        waveform = resampler(waveform)
    return waveform


def _to_mono(waveform: torch.Tensor) -> torch.Tensor:
    """
    Convert stereo to mono by averaging channels.

    Args:
        waveform: Input waveform (channels, samples)

    Returns:
        Mono waveform (1, samples)
    """
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
    return waveform


def _adjust_length(signal: torch.Tensor, target_length: int) -> torch.Tensor:
    """
    Adjust signal length by truncating or padding.

    Args:
        signal: Input signal (samples,)
        target_length: Desired length in samples

    Returns:
        Signal adjusted to target_length
    """
    current_length = signal.shape[-1]

    if current_length == target_length:
        return signal
    elif current_length > target_length:
        return signal[:target_length]
    else:
        padding = (0, target_length - current_length)
        return torch.nn.functional.pad(signal, padding, mode='constant', value=0)


def load_librispeech_batch(
    num_samples: int,
    target_length: int,
    target_fs: int = DEFAULT_FS,
    seed: Optional[int] = None
) -> torch.Tensor:
    """
    Load a batch of clean speech from LibriSpeech test-clean using streaming.

    Uses streaming mode to avoid downloading the entire dataset.
    Only downloads the samples that are actually needed.

    Args:
        num_samples: Number of samples to load
        target_length: Target length in samples (for uniform batching)
        target_fs: Target sampling rate (default: 16000)
        seed: Random seed for reproducibility

    Returns:
        Tensor of audio samples, shape (num_samples, target_length)
    """
    if seed is not None:
        random.seed(seed)

    # Load LibriSpeech test-clean dataset in streaming mode
    # This avoids downloading the entire dataset
    dataset = load_dataset(
        SPEECH_DATASET,
        SPEECH_CONFIG,
        split=SPEECH_SPLIT,
        streaming=True
    )

    # Convert to iterable and take num_samples
    # For streaming datasets, we need to iterate
    audio_list = []

    # Shuffle if seed is provided
    if seed is not None:
        dataset = dataset.shuffle(seed=seed, buffer_size=1000)

    # Take the required number of samples
    for i, sample in enumerate(dataset):
        if i >= num_samples:
            break

        # LibriSpeech format: {'audio': {'array': np.array, 'sampling_rate': int}, ...}
        audio_array = torch.from_numpy(sample['audio']['array']).float()
        sampling_rate = sample['audio']['sampling_rate']

        # Ensure mono (LibriSpeech is already mono, but just in case)
        if audio_array.ndim == 1:
            audio_array = audio_array.unsqueeze(0)  # (1, samples)
        else:
            audio_array = _to_mono(audio_array)

        # Resample if needed
        audio_array = _resample_if_needed(audio_array, sampling_rate, target_fs)

        # Squeeze to 1D and adjust length
        audio_array = audio_array.squeeze(0)
        audio_array = _adjust_length(audio_array, target_length)

        audio_list.append(audio_array)

    return torch.stack(audio_list, dim=0)


def load_demand_noise_batch(
    num_samples: int,
    target_length: int,
    target_fs: int = DEFAULT_FS,
    seed: Optional[int] = None
) -> torch.Tensor:
    """
    Load a batch of noise from ESC-50 dataset using streaming.

    Uses streaming mode to avoid downloading the entire dataset.
    Only downloads the samples that are actually needed.

    Args:
        num_samples: Number of noise samples to load
        target_length: Target length in samples
        target_fs: Target sampling rate (default: 16000)
        seed: Random seed for reproducibility

    Returns:
        Tensor of noise samples, shape (num_samples, target_length)
    """
    if seed is not None:
        random.seed(seed)

    # Load ESC-50 dataset in streaming mode
    dataset = load_dataset(
        NOISE_DATASET,
        split="train",  # ESC-50 uses 'train' split
        streaming=True
    )

    noise_list = []

    # Shuffle if seed is provided
    if seed is not None:
        dataset = dataset.shuffle(seed=seed, buffer_size=100)

    # Take the required number of samples
    for i, sample in enumerate(dataset):
        if i >= num_samples:
            break

        # DEMAND format: {'audio': {'array': np.array, 'sampling_rate': int}, ...}
        audio_array = torch.from_numpy(sample['audio']['array']).float()
        sampling_rate = sample['audio']['sampling_rate']

        # Convert to mono if stereo
        if audio_array.ndim == 1:
            audio_array = audio_array.unsqueeze(0)
        else:
            audio_array = _to_mono(audio_array)

        # Resample if needed
        audio_array = _resample_if_needed(audio_array, sampling_rate, target_fs)

        # Squeeze to 1D and adjust length
        audio_array = audio_array.squeeze(0)

        # If noise is shorter than target, repeat it
        if audio_array.shape[0] < target_length:
            num_repeats = (target_length // audio_array.shape[0]) + 1
            audio_array = audio_array.repeat(num_repeats)

        audio_array = _adjust_length(audio_array, target_length)

        noise_list.append(audio_array)

    return torch.stack(noise_list, dim=0)


def get_dataset_info() -> dict:
    """
    Get information about configured datasets.

    Returns:
        Dictionary with dataset configuration
    """
    return {
        'speech': {
            'dataset': SPEECH_DATASET,
            'config': SPEECH_CONFIG,
            'split': SPEECH_SPLIT,
            'description': 'LibriSpeech ASR test-clean (English audiobook)'
        },
        'noise': {
            'dataset': NOISE_DATASET,
            'description': 'ESC-50 - 50 environmental sound classes (2000 samples)'
        },
        'target_fs': DEFAULT_FS
    }
