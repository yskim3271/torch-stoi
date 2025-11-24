# CUDA-STOI Project

GPU-accelerated STOI (Short-Time Objective Intelligibility) implementation using PyTorch.

## Project Overview

This project aims to create a GPU-accelerated version of the STOI speech intelligibility metric, providing:
- Numerical equivalence with the original pystoi implementation (MAE < 1e-6)
- Significant performance improvements through GPU batch processing
- PyTorch integration for differentiable loss functions

## Directory Structure

```
cuda-pesq/
├── .claude/          # Claude Code configuration (agents, commands, skills)
├── cuda_stoi/        # Main PyTorch implementation of STOI
├── data/             # Dataset files (ESC50, Zeroth Korean, etc.)
├── docs/             # Additional documentation and analysis
├── pystoi/           # Reference implementation (original pystoi library)
├── README.md         # Main project documentation
├── setup.py          # Package installation script
└── requirements.txt  # Python dependencies
```

### Directory Details

#### `.claude/`
Claude Code configuration directory containing custom agents, slash commands, and skills for this project.

#### `cuda_stoi/`
Main implementation directory. Contains the PyTorch-based STOI modules:
- Core signal processing functions (STFT, resampling, etc.)
- STOI computation logic
- GPU-accelerated batch processing

#### `data/`
Audio datasets used for testing and validation. See [data/README.md](data/README.md) for details.

#### `docs/`
Additional documentation including:
- [STOI_ALGORITHM_BLOCK_DIAGRAM.md](docs/STOI_ALGORITHM_BLOCK_DIAGRAM.md) - Algorithm flowcharts
- [BATCHED_PROCESSING_ARCHITECTURE.md](docs/BATCHED_PROCESSING_ARCHITECTURE.md) - Batched processing implementation details
- Other analysis and planning documents

#### `pystoi/`
Original pystoi library used as reference implementation for numerical validation. See [pystoi/README.md](pystoi/README.md) for usage.

## Documentation Map

### For New Users
Start here to understand the project:
1. [README.md](README.md) - Project overview, goals, and current status
2. [docs/STOI_ALGORITHM_BLOCK_DIAGRAM.md](docs/STOI_ALGORITHM_BLOCK_DIAGRAM.md) - Understand the STOI algorithm

### For Developers
When working on this project:
1. Check [README.md](README.md) for project goals and architecture
2. Use [docs/](docs/) for technical algorithm details
3. Review core modules in [cuda_stoi/core/](cuda_stoi/core/)
4. See [docs/BATCHED_PROCESSING_ARCHITECTURE.md](docs/BATCHED_PROCESSING_ARCHITECTURE.md) for batch processing implementation

### Quick Reference
- **What is STOI?** See [README.md](README.md#why-gpu-acceleration)
- **How to use?** See [README.md](README.md#usage)
- **Algorithm details?** See [docs/STOI_ALGORITHM_BLOCK_DIAGRAM.md](docs/STOI_ALGORITHM_BLOCK_DIAGRAM.md)
- **Batch processing?** See [docs/BATCHED_PROCESSING_ARCHITECTURE.md](docs/BATCHED_PROCESSING_ARCHITECTURE.md)
- **Reference implementation?** See [pystoi/README.md](pystoi/README.md)

## Key Technologies

- **PyTorch**: GPU acceleration and differentiable operations
- **Python 3.x**: Primary implementation language
- **Reference**: pystoi (original CPU implementation)

## Current Status

**Version 0.3.0 - Refactoring Complete** ✅

### Implementation Status
Phase 1 (PyTorch implementation) is complete with all 6 core modules implemented:
- ✅ Resampling (torchaudio-based, GPU-accelerated)
- ✅ STFT (pystoi-compatible, fully batched)
- ✅ Voice Activity Detection (VAD with padding support)
- ✅ Octave Band Filtering (15 bands, GPU-accelerated)
- ✅ Correlation Computation (fully vectorized)
- ✅ Main STOI API (batched-only architecture)

The implementation is GPU-accelerated and numerically validated against pystoi (MAE < 1e-6).

### Recent Refactoring (v0.3.0)
**Completed**: Codebase streamlined and simplified
- **26% code reduction**: Removed ~784 lines
- **Batched-only architecture**: Single unified API for all operations
- **Code cleanup**:
  - Deleted `gpu_resampling.py` (unused polyphase implementation)
  - Removed redundant single-signal functions
  - Simplified module exports and structure
- **Improved maintainability**: Cleaner code, better organization
- **Backward compatible**: User-facing API unchanged

### File Structure Updates
```
cuda_stoi/
├── core/
│   ├── resampling.py         # Resampling (torchaudio)
│   ├── stft.py               # STFT (refactored, -17%)
│   ├── vad.py                # VAD with padding
│   ├── octave_bands.py       # Octave filtering (refactored, -13%)
│   └── correlation.py        # Correlation (refactored, -28%)
├── stoi.py                   # Main API
├── constants.py              # Algorithm parameters
└── utils.py                  # Utility functions
```

**Removed**: `cuda_stoi/core/gpu_resampling.py` (528 lines)
