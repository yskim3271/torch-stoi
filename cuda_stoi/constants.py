"""
STOI Constants

All parameters extracted from pystoi in Phase 0
These values MUST match pystoi exactly for numerical equivalence
"""

import torch
import numpy as np

# ============================================================================
# Basic Parameters (from pystoi.stoi)
# ============================================================================

FS = 10000              # Internal sampling frequency (Hz)
N_FRAME = 256           # Window size (samples)
NFFT = 512              # FFT size
NUMBAND = 15            # Number of 1/3 octave bands
MINFREQ = 150           # Center frequency of 1st octave band (Hz)
N = 30                  # Number of frames per segment
BETA = -15.0            # Lower SDR bound (dB)
DYN_RANGE = 40          # Speech dynamic range for VAD (dB)

# Computed parameters
HOP_LENGTH = N_FRAME // 2  # 128 samples
SEGMENT_LENGTH = N * HOP_LENGTH  # 3840 samples ≈ 384ms @ 10kHz
CLIP_VALUE = 10 ** (-BETA / 20)  # ≈ 5.623413

# Numerical stability constant
# CRITICAL: Must match pystoi exactly!
EPS = np.finfo(np.float32).eps  # For GPU (float32)
EPS_DOUBLE = np.finfo(np.float64).eps  # For CPU validation (float64)

# ============================================================================
# 1/3 Octave Band Center Frequencies
# ============================================================================

CENTER_FREQS = np.array([
    150.0, 189.0, 238.1, 300.0, 378.0, 476.2, 600.0, 756.0,
    952.4, 1200.0, 1511.9, 1904.9, 2400.0, 3023.8, 3809.8
], dtype=np.float32)

# ============================================================================
# Device Management
# ============================================================================

def get_default_device():
    """Get default device (CUDA if available, else CPU)"""
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def get_default_dtype():
    """Get default dtype (float32 for GPU compatibility)"""
    return torch.float32


# ============================================================================
# Load pre-computed Octave Band Matrix
# ============================================================================

def load_obm(device=None, dtype=None):
    """
    Load pre-computed 1/3 octave band matrix from Phase 0

    Args:
        device: torch device
        dtype: torch dtype

    Returns:
        obm: (15, 257) tensor - Octave Band Matrix
        cf: (15,) tensor - Center frequencies
    """
    if device is None:
        device = get_default_device()
    if dtype is None:
        dtype = get_default_dtype()

    # Load from Phase 0 generated files
    obm_path = 'tests/fixtures/reference/pystoi_obm.npy'
    cf_path = 'tests/fixtures/reference/pystoi_cf.npy'

    try:
        obm = np.load(obm_path)
        cf = np.load(cf_path)

        # Convert to torch tensors
        obm = torch.from_numpy(obm).to(device=device, dtype=dtype)
        cf = torch.from_numpy(cf).to(device=device, dtype=dtype)

        return obm, cf

    except FileNotFoundError:
        raise FileNotFoundError(
            f"OBM files not found. Please run Phase 0 tools first:\n"
            f"  python tools/extract_pystoi_params.py"
        )


# ============================================================================
# Testing
# ============================================================================

if __name__ == '__main__':
    # Test loading OBM
    print("Testing OBM loading...")
    try:
        obm, cf = load_obm()
        print(f"✓ OBM shape: {obm.shape}")
        print(f"✓ CF shape: {cf.shape}")
        print(f"✓ Device: {obm.device}")
        print(f"✓ Dtype: {obm.dtype}")
    except Exception as e:
        print(f"✗ Error: {e}")
