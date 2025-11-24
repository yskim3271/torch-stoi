"""
CUDA-STOI Setup
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="cuda-stoi",
    version="0.1.0",
    author="CUDA-STOI Development Team",
    description="GPU-accelerated STOI (Short-Time Objective Intelligibility) implementation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/cuda-stoi",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "torchaudio>=2.0.0",
        "numpy>=1.20.0",
        "scipy>=1.7.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=3.0.0",
            "pystoi>=0.3.3",
        ],
        "viz": [
            "matplotlib>=3.5.0",
            "pandas>=1.3.0",
            "seaborn>=0.11.0",
        ],
    },
)
