"""
Common Utility Functions for CUDA-STOI

This module provides reusable utility functions that are used across multiple
modules in the CUDA-STOI package, reducing code duplication and improving
maintainability.

Utilities:
    - Input validation and type conversion
    - Batch dimension handling
    - Hanning window creation (MATLAB-compatible)
    - Device and dtype management
"""

import torch
import numpy as np
from typing import Tuple, Union, Optional


def validate_and_convert_tensors(
    clean: Union[np.ndarray, torch.Tensor],
    degraded: Union[np.ndarray, torch.Tensor],
    param_name_clean: str = "clean",
    param_name_degraded: str = "degraded"
) -> Tuple[torch.Tensor, torch.Tensor, bool, Optional[torch.device]]:
    """
    Validate input types and convert to torch tensors.

    This function standardizes input validation and conversion logic that is
    repeated across multiple modules (stoi.py, vad.py, etc.).

    Args:
        clean: Clean reference signal(s)
        degraded: Degraded/processed signal(s)
        param_name_clean: Name of clean parameter for error messages
        param_name_degraded: Name of degraded parameter for error messages

    Returns:
        Tuple containing:
            - clean_tensor: Converted clean signal as float32 tensor
            - degraded_tensor: Converted degraded signal as float32 tensor
            - is_numpy: True if original inputs were numpy arrays
            - device: Original device if inputs were tensors, None otherwise

    Raises:
        TypeError: If inputs are not np.ndarray or torch.Tensor
        ValueError: If clean and degraded have different shapes

    Example:
        >>> clean = np.random.randn(16000)
        >>> degraded = np.random.randn(16000)
        >>> clean_t, degraded_t, is_numpy, device = validate_and_convert_tensors(clean, degraded)
        >>> print(clean_t.dtype, is_numpy)
        torch.float32 True
    """
    # Type validation
    if not isinstance(clean, (np.ndarray, torch.Tensor)):
        raise TypeError(
            f"{param_name_clean} must be np.ndarray or torch.Tensor, "
            f"got {type(clean).__name__}"
        )

    if not isinstance(degraded, (np.ndarray, torch.Tensor)):
        raise TypeError(
            f"{param_name_degraded} must be np.ndarray or torch.Tensor, "
            f"got {type(degraded).__name__}"
        )

    # Shape validation
    if clean.shape != degraded.shape:
        raise ValueError(
            f"{param_name_clean} and {param_name_degraded} must have the same shape, "
            f"got {clean.shape} and {degraded.shape}"
        )

    # Convert to torch tensors
    is_numpy = isinstance(clean, np.ndarray)
    device = None

    if isinstance(clean, np.ndarray):
        clean = torch.from_numpy(clean).float()
        degraded = torch.from_numpy(degraded).float()
    else:
        device = clean.device
        clean = clean.float()
        degraded = degraded.float()

        # Ensure both tensors are on the same device
        if degraded.device != device:
            degraded = degraded.to(device)

    return clean, degraded, is_numpy, device


def ensure_batch_dimension(
    tensor: torch.Tensor,
    expected_ndim: int = 2
) -> Tuple[torch.Tensor, bool]:
    """
    Ensure tensor has batch dimension, adding it if necessary.

    Args:
        tensor: Input tensor (1D or 2D)
        expected_ndim: Expected number of dimensions (default: 2)

    Returns:
        Tuple containing:
            - tensor: Tensor with batch dimension (2D)
            - squeeze_output: Whether to remove batch dim in output

    Raises:
        ValueError: If tensor has invalid number of dimensions

    Example:
        >>> # Single signal
        >>> x = torch.randn(16000)
        >>> x_batch, squeeze = ensure_batch_dimension(x)
        >>> print(x_batch.shape, squeeze)
        torch.Size([1, 16000]) True

        >>> # Already batched
        >>> x = torch.randn(4, 16000)
        >>> x_batch, squeeze = ensure_batch_dimension(x)
        >>> print(x_batch.shape, squeeze)
        torch.Size([4, 16000]) False
    """
    if tensor.ndim == 1:
        # Single signal - add batch dimension
        tensor = tensor.unsqueeze(0)
        squeeze_output = True
    elif tensor.ndim == expected_ndim:
        # Already batched
        squeeze_output = False
    else:
        raise ValueError(
            f"Input must be 1D (samples,) or {expected_ndim}D (batch, samples, ...), "
            f"got {tensor.ndim}D"
        )

    return tensor, squeeze_output


def create_hanning_window_matlab_compatible(
    win_size: int,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None
) -> torch.Tensor:
    """
    Create MATLAB-compatible Hanning window.

    CRITICAL: pystoi uses np.hanning(win_size + 2)[1:-1]
    This excludes first and last points to match MATLAB's hanning() function.

    This window creation is used in multiple modules (stft.py, vad.py) and
    is extracted here to eliminate duplication.

    Args:
        win_size: Window size (number of samples)
        device: Target torch device (default: CPU)
        dtype: Target torch dtype (default: float32)

    Returns:
        window: Hanning window tensor of shape (win_size,)

    Example:
        >>> window = create_hanning_window_matlab_compatible(256)
        >>> print(window.shape, window.dtype)
        torch.Size([256]) torch.float32

        >>> # GPU window
        >>> window_gpu = create_hanning_window_matlab_compatible(
        ...     256, device=torch.device('cuda'), dtype=torch.float64
        ... )
        >>> print(window_gpu.device, window_gpu.dtype)
        cuda:0 torch.float64
    """
    if device is None:
        device = torch.device('cpu')
    if dtype is None:
        dtype = torch.float32

    # MATLAB-compatible: hanning(N+2)[1:-1]
    # This matches pystoi's exact window for numerical equivalence
    window_np = np.hanning(win_size + 2)[1:-1].astype(np.float32)
    window = torch.from_numpy(window_np).to(device=device, dtype=dtype)

    return window


def get_device_and_dtype(
    tensor: Optional[torch.Tensor] = None,
    default_device: Optional[torch.device] = None,
    default_dtype: Optional[torch.dtype] = None
) -> Tuple[torch.device, torch.dtype]:
    """
    Get device and dtype from tensor or defaults.

    Args:
        tensor: Optional tensor to extract device/dtype from
        default_device: Default device if tensor is None (default: CPU)
        default_dtype: Default dtype if tensor is None (default: float32)

    Returns:
        Tuple containing:
            - device: torch.device
            - dtype: torch.dtype

    Example:
        >>> tensor = torch.randn(100, device='cuda', dtype=torch.float64)
        >>> device, dtype = get_device_and_dtype(tensor)
        >>> print(device, dtype)
        cuda:0 torch.float64

        >>> # With defaults
        >>> device, dtype = get_device_and_dtype()
        >>> print(device, dtype)
        cpu torch.float32
    """
    if tensor is not None:
        device = tensor.device
        dtype = tensor.dtype
    else:
        device = default_device if default_device is not None else torch.device('cpu')
        dtype = default_dtype if default_dtype is not None else torch.float32

    return device, dtype


def convert_output_format(
    tensor: torch.Tensor,
    is_numpy: bool,
    squeeze_output: bool
) -> Union[float, np.ndarray, torch.Tensor]:
    """
    Convert output tensor to appropriate format based on input type.

    Handles conversion back to numpy if input was numpy, and squeezing
    batch dimension if input was single signal.

    Args:
        tensor: Output tensor to convert
        is_numpy: Whether original input was numpy array
        squeeze_output: Whether to remove batch dimension

    Returns:
        Converted output in appropriate format:
            - float: If 1D input and single value result
            - np.ndarray: If input was numpy
            - torch.Tensor: If input was torch tensor

    Example:
        >>> # Single signal, numpy input -> float
        >>> tensor = torch.tensor([[0.85]])
        >>> result = convert_output_format(tensor, is_numpy=True, squeeze_output=True)
        >>> print(type(result), result)
        <class 'float'> 0.85

        >>> # Batch, torch input -> tensor
        >>> tensor = torch.tensor([0.85, 0.90, 0.88])
        >>> result = convert_output_format(tensor, is_numpy=False, squeeze_output=False)
        >>> print(type(result), result.shape)
        <class 'torch.Tensor'> torch.Size([3])
    """
    # Squeeze batch dimension if needed
    if squeeze_output:
        tensor = tensor.squeeze(0)

    # Convert to scalar if single value
    if tensor.numel() == 1 and squeeze_output:
        return tensor.item()

    # Convert to numpy if original input was numpy
    if is_numpy:
        return tensor.cpu().numpy()

    return tensor
