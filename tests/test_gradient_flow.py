"""
Gradient Flow Diagnostic for STOI

This script traces where gradients are being broken in the STOI implementation
by testing each pipeline step individually.
"""

import torch
import numpy as np
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from cuda_stoi.core.resampling import resample_to_10k
from cuda_stoi.core.vad import remove_silent_frames_batched
from cuda_stoi.core.stft import compute_stft
from cuda_stoi.core.octave_bands import apply_octave_bands
from cuda_stoi.core.correlation import compute_stoi_batched
from cuda_stoi.utils import validate_and_convert_tensors, ensure_batch_dimension
from cuda_stoi.constants import FS, DYN_RANGE, N_FRAME, N


def print_gradient_status(tensor, name):
    """Print detailed gradient status of a tensor"""
    print(f"\n  {name}:")
    print(f"    Type: {type(tensor)}")
    if isinstance(tensor, torch.Tensor):
        print(f"    Shape: {tensor.shape}")
        print(f"    Device: {tensor.device}")
        print(f"    Dtype: {tensor.dtype}")
        print(f"    requires_grad: {tensor.requires_grad}")
        print(f"    is_leaf: {tensor.is_leaf}")
        print(f"    grad_fn: {tensor.grad_fn}")
        if hasattr(tensor, '_version') and tensor._version > 0:
            print(f"    ⚠ Modified in-place! (_version={tensor._version})")
    else:
        print(f"    ✗ Not a tensor! Type: {type(tensor)}")


def test_step_1_validation():
    """Test Step 1: Input validation and conversion"""
    print("=" * 80)
    print("STEP 1: validate_and_convert_tensors()")
    print("=" * 80)

    # Create input with gradients
    clean = torch.randn(16000, requires_grad=True)
    degraded = torch.randn(16000, requires_grad=True)

    print("\n[INPUT]")
    print_gradient_status(clean, "clean (input)")
    print_gradient_status(degraded, "degraded (input)")

    # Call validation
    clean_out, degraded_out, is_numpy, device = validate_and_convert_tensors(clean, degraded)

    print("\n[OUTPUT]")
    print_gradient_status(clean_out, "clean (output)")
    print_gradient_status(degraded_out, "degraded (output)")
    print(f"  is_numpy: {is_numpy}")
    print(f"  device: {device}")

    # Test backward
    if isinstance(clean_out, torch.Tensor) and clean_out.requires_grad:
        try:
            loss = clean_out.sum()
            loss.backward()
            print(f"\n  ✓ Backward pass successful!")
            print(f"  ✓ Input gradient exists: {clean.grad is not None}")
        except Exception as e:
            print(f"\n  ✗ Backward failed: {e}")
    else:
        print(f"\n  ✗ Output doesn't require grad - gradient broken!")

    return clean_out.requires_grad if isinstance(clean_out, torch.Tensor) else False


def test_step_2_ensure_batch():
    """Test Step 2: Ensure batch dimension"""
    print("\n" + "=" * 80)
    print("STEP 2: ensure_batch_dimension()")
    print("=" * 80)

    clean = torch.randn(16000, requires_grad=True)

    print("\n[INPUT]")
    print_gradient_status(clean, "clean (1D)")

    clean_batch, squeeze_output = ensure_batch_dimension(clean)

    print("\n[OUTPUT]")
    print_gradient_status(clean_batch, "clean_batch (2D)")
    print(f"  squeeze_output: {squeeze_output}")

    # Test backward
    if clean_batch.requires_grad:
        try:
            loss = clean_batch.sum()
            loss.backward()
            print(f"\n  ✓ Backward pass successful!")
            print(f"  ✓ Input gradient exists: {clean.grad is not None}")
        except Exception as e:
            print(f"\n  ✗ Backward failed: {e}")
    else:
        print(f"\n  ✗ Output doesn't require grad - gradient broken!")

    return clean_batch.requires_grad


def test_step_3_resampling():
    """Test Step 3: Resampling to 10kHz"""
    print("\n" + "=" * 80)
    print("STEP 3: resample_to_10k()")
    print("=" * 80)

    fs = 16000
    clean = torch.randn(1, fs * 3, requires_grad=True)

    print("\n[INPUT]")
    print_gradient_status(clean, "clean (16kHz)")

    clean_10k = resample_to_10k(clean, original_fs=fs)

    print("\n[OUTPUT]")
    print_gradient_status(clean_10k, "clean_10k (10kHz)")

    # Test backward
    if clean_10k.requires_grad:
        try:
            loss = clean_10k.sum()
            loss.backward()
            print(f"\n  ✓ Backward pass successful!")
            print(f"  ✓ Input gradient exists: {clean.grad is not None}")
            if clean.grad is not None:
                print(f"  ✓ Gradient magnitude: {clean.grad.abs().mean():.6e}")
        except Exception as e:
            print(f"\n  ✗ Backward failed: {e}")
    else:
        print(f"\n  ✗ Output doesn't require grad - gradient broken!")

    return clean_10k.requires_grad


def test_step_4_vad():
    """Test Step 4: Voice Activity Detection (VAD)"""
    print("\n" + "=" * 80)
    print("STEP 4: remove_silent_frames_batched()")
    print("=" * 80)

    clean = torch.randn(2, 10000 * 3, requires_grad=True)
    degraded = torch.randn(2, 10000 * 3, requires_grad=True)

    print("\n[INPUT]")
    print_gradient_status(clean, "clean")
    print_gradient_status(degraded, "degraded")

    clean_vad, degraded_vad, vad_lengths = remove_silent_frames_batched(
        clean, degraded,
        dyn_range=DYN_RANGE,
        framelen=N_FRAME,
        hop=N_FRAME // 2
    )

    print("\n[OUTPUT]")
    print_gradient_status(clean_vad, "clean_vad")
    print_gradient_status(degraded_vad, "degraded_vad")
    print(f"  vad_lengths: {vad_lengths}")

    # Test backward
    if isinstance(clean_vad, torch.Tensor) and clean_vad.requires_grad:
        try:
            loss = clean_vad.sum()
            loss.backward()
            print(f"\n  ✓ Backward pass successful!")
            print(f"  ✓ Input gradient exists: {clean.grad is not None}")
            if clean.grad is not None:
                print(f"  ✓ Gradient magnitude: {clean.grad.abs().mean():.6e}")
        except Exception as e:
            print(f"\n  ✗ Backward failed: {e}")
    else:
        print(f"\n  ✗ Output doesn't require grad - gradient broken!")

    return isinstance(clean_vad, torch.Tensor) and clean_vad.requires_grad


def test_step_5_stft():
    """Test Step 5: STFT computation"""
    print("\n" + "=" * 80)
    print("STEP 5: compute_stft()")
    print("=" * 80)

    signal = torch.randn(2, 30000, requires_grad=True)
    lengths = torch.tensor([30000, 30000])

    print("\n[INPUT]")
    print_gradient_status(signal, "signal")
    print(f"  lengths: {lengths}")

    stft_out, frame_lengths = compute_stft(signal, lengths=lengths)

    print("\n[OUTPUT]")
    print_gradient_status(stft_out, "stft_out")
    print(f"  frame_lengths: {frame_lengths}")

    # Test backward
    if stft_out.requires_grad:
        try:
            loss = stft_out.sum()
            loss.backward()
            print(f"\n  ✓ Backward pass successful!")
            print(f"  ✓ Input gradient exists: {signal.grad is not None}")
            if signal.grad is not None:
                print(f"  ✓ Gradient magnitude: {signal.grad.abs().mean():.6e}")
        except Exception as e:
            print(f"\n  ✗ Backward failed: {e}")
    else:
        print(f"\n  ✗ Output doesn't require grad - gradient broken!")

    return stft_out.requires_grad


def test_step_6_octave_bands():
    """Test Step 6: Octave band filtering"""
    print("\n" + "=" * 80)
    print("STEP 6: apply_octave_bands()")
    print("=" * 80)

    # Create fake STFT output
    stft_mag = torch.randn(2, 100, 257, requires_grad=True)

    print("\n[INPUT]")
    print_gradient_status(stft_mag, "stft_mag")

    bands = apply_octave_bands(stft_mag)

    print("\n[OUTPUT]")
    print_gradient_status(bands, "bands")

    # Test backward
    if bands.requires_grad:
        try:
            loss = bands.sum()
            loss.backward()
            print(f"\n  ✓ Backward pass successful!")
            print(f"  ✓ Input gradient exists: {stft_mag.grad is not None}")
            if stft_mag.grad is not None:
                print(f"  ✓ Gradient magnitude: {stft_mag.grad.abs().mean():.6e}")
        except Exception as e:
            print(f"\n  ✗ Backward failed: {e}")
    else:
        print(f"\n  ✗ Output doesn't require grad - gradient broken!")

    return bands.requires_grad


def test_step_7_correlation():
    """Test Step 7: STOI correlation computation"""
    print("\n" + "=" * 80)
    print("STEP 7: compute_stoi_batched()")
    print("=" * 80)

    # Create fake octave band outputs
    clean_bands = torch.randn(2, 100, 15, requires_grad=True)
    degraded_bands = torch.randn(2, 100, 15, requires_grad=True)
    frame_lengths = torch.tensor([100, 100])

    print("\n[INPUT]")
    print_gradient_status(clean_bands, "clean_bands")
    print_gradient_status(degraded_bands, "degraded_bands")
    print(f"  frame_lengths: {frame_lengths}")

    scores = compute_stoi_batched(
        clean_bands, degraded_bands,
        segment_length=N,
        frame_lengths=frame_lengths
    )

    print("\n[OUTPUT]")
    print_gradient_status(scores, "scores")

    # Test backward
    if isinstance(scores, torch.Tensor) and scores.requires_grad:
        try:
            loss = scores.sum()
            loss.backward()
            print(f"\n  ✓ Backward pass successful!")
            print(f"  ✓ Input gradient exists: {clean_bands.grad is not None}")
            if clean_bands.grad is not None:
                print(f"  ✓ Gradient magnitude: {clean_bands.grad.abs().mean():.6e}")
        except Exception as e:
            print(f"\n  ✗ Backward failed: {e}")
    else:
        print(f"\n  ✗ Output doesn't require grad - gradient broken!")

    return isinstance(scores, torch.Tensor) and scores.requires_grad


def test_full_pipeline():
    """Test full STOI pipeline end-to-end"""
    print("\n" + "=" * 80)
    print("FULL PIPELINE TEST")
    print("=" * 80)

    from cuda_stoi import stoi

    clean = torch.randn(16000 * 3, requires_grad=True)
    degraded = clean + torch.randn(16000 * 3) * 0.1

    print("\n[INPUT]")
    print_gradient_status(clean, "clean")

    score = stoi(clean, degraded, fs_sig=16000)

    print("\n[OUTPUT]")
    print_gradient_status(score if isinstance(score, torch.Tensor) else torch.tensor(score), "score")

    # Test backward
    if isinstance(score, torch.Tensor) and score.requires_grad:
        try:
            score.backward()
            print(f"\n  ✓ Backward pass successful!")
            print(f"  ✓ Input gradient exists: {clean.grad is not None}")
            if clean.grad is not None:
                print(f"  ✓ Gradient magnitude: {clean.grad.abs().mean():.6e}")
        except Exception as e:
            print(f"\n  ✗ Backward failed: {e}")
    else:
        print(f"\n  ✗ Output doesn't require grad or is not a tensor - gradient broken!")

    return isinstance(score, torch.Tensor) and score.requires_grad


def main():
    print("\n" + "=" * 80)
    print("GRADIENT FLOW DIAGNOSTIC FOR CUDA-STOI")
    print("=" * 80)
    print("\nThis script tests each pipeline step to identify where gradients break.\n")

    results = {}

    # Test each step
    results['validation'] = test_step_1_validation()
    results['batch_dim'] = test_step_2_ensure_batch()
    results['resampling'] = test_step_3_resampling()
    results['vad'] = test_step_4_vad()
    results['stft'] = test_step_5_stft()
    results['octave_bands'] = test_step_6_octave_bands()
    results['correlation'] = test_step_7_correlation()
    results['full_pipeline'] = test_full_pipeline()

    # Summary
    print("\n" + "=" * 80)
    print("GRADIENT FLOW SUMMARY")
    print("=" * 80)

    print("\nStep-by-Step Results:")
    for step, passes in results.items():
        status = "✓ PASS" if passes else "✗ FAIL"
        print(f"  {step:20s}: {status}")

    # Find first failure
    print("\nGradient Flow Analysis:")
    for i, (step, passes) in enumerate(results.items()):
        if not passes:
            print(f"  ⚠ FIRST FAILURE at step {i+1}: {step}")
            print(f"  ⚠ This is where the gradient graph is broken!")
            break
    else:
        print(f"  ✓ All steps preserve gradients!")

    print("\n" + "=" * 80)


if __name__ == '__main__':
    main()
