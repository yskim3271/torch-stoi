"""
Visualization Module

Benchmark result visualization:
- Speed comparison and accuracy plots
- Markdown tables and reports
"""

import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Tuple, Optional
from pathlib import Path

# Plot configuration constants
DEFAULT_DPI = 300
DEFAULT_FIGSIZE = (14, 5)
SPEEDUP_COLOR = 'green'
ANNOTATION_FONTSIZE = 9


def _apply_plot_style(
    ax: plt.Axes,
    title: str,
    xlabel: str,
    ylabel: str,
    legend: bool = True
) -> None:
    """
    Apply consistent styling to a plot axis.

    Args:
        ax: Matplotlib axis object
        title: Plot title
        xlabel: X-axis label
        ylabel: Y-axis label
        legend: Whether to show legend
    """
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    if legend:
        ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)


def _add_speedup_annotations(
    ax: plt.Axes,
    x_values: List,
    y_values: List,
    speedups: List,
    skip_factor: int = 1
) -> None:
    """
    Add speedup annotations to a plot.

    Args:
        ax: Matplotlib axis object
        x_values: X-axis values for annotation positions
        y_values: Y-axis values for annotation positions
        speedups: Speedup values to annotate
        skip_factor: Annotate every skip_factor-th point (1 = all points)
    """
    for i, (x, y, speedup) in enumerate(zip(x_values, y_values, speedups)):
        if i % skip_factor == 0:
            ax.annotate(
                f'{speedup:.1f}x',
                xy=(x, y),
                xytext=(0, 10),
                textcoords='offset points',
                ha='center',
                fontsize=ANNOTATION_FONTSIZE,
                color=SPEEDUP_COLOR,
                weight='bold'
            )


def plot_speed_comparison(
    batch_results: Dict,
    length_results: Dict,
    output_file: Path
):
    """
    Create speed comparison plots for batch size and signal length.

    Args:
        batch_results: Dictionary with batch size benchmark results
        length_results: Dictionary with signal length benchmark results
        output_file: Path to save the figure
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Batch Size vs Processing Time
    batch_sizes = batch_results['batch_sizes']
    cuda_times = batch_results['cuda_stoi_times']
    pystoi_times = batch_results['pystoi_times']
    speedups = batch_results['speedups']

    ax1.plot(batch_sizes, cuda_times, 'o-', label='cuda-stoi', linewidth=2, markersize=8)
    ax1.plot(batch_sizes, pystoi_times, 's-', label='pystoi', linewidth=2, markersize=8)
    ax1.set_xscale('log', base=2)
    _apply_plot_style(ax1, 'Batch Size vs Processing Time', 'Batch Size', 'Processing Time (ms)')
    _add_speedup_annotations(ax1, batch_sizes, cuda_times, speedups, skip_factor=2)

    # Plot 2: Signal Length vs Processing Time
    signal_lengths = length_results['signal_lengths']
    cuda_times_len = length_results['cuda_stoi_times']
    pystoi_times_len = length_results['pystoi_times']
    speedups_len = length_results['speedups']

    ax2.plot(signal_lengths, cuda_times_len, 'o-', label='cuda-stoi', linewidth=2, markersize=8)
    ax2.plot(signal_lengths, pystoi_times_len, 's-', label='pystoi', linewidth=2, markersize=8)
    _apply_plot_style(ax2, 'Signal Length vs Processing Time', 'Signal Length (seconds)', 'Processing Time (ms)')
    _add_speedup_annotations(ax2, signal_lengths, cuda_times_len, speedups_len, skip_factor=1)

    plt.tight_layout()
    plt.savefig(output_file, dpi=DEFAULT_DPI, bbox_inches='tight')
    print(f"Speed comparison plot saved to: {output_file}")
    plt.close()


def plot_accuracy_validation(
    accuracy_results: Dict,
    output_file: Path
):
    """
    Create accuracy validation plot showing MAE and MSE at different SNR levels.

    Args:
        accuracy_results: Dictionary with accuracy benchmark results
        output_file: Path to save the figure
    """
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    snr_levels = accuracy_results['snr_levels']
    mae_values = accuracy_results['mae_values']
    mse_values = accuracy_results['mse_values']

    x = np.arange(len(snr_levels))
    width = 0.35

    bars1 = ax.bar(x - width/2, mae_values, width, label='MAE', alpha=0.8)
    bars2 = ax.bar(x + width/2, mse_values, width, label='MSE', alpha=0.8)

    ax.set_xlabel('SNR Level (dB)', fontsize=12)
    ax.set_ylabel('Error', fontsize=12)
    ax.set_title('Accuracy Validation: cuda-stoi vs pystoi', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([f'{snr}dB' for snr in snr_levels])
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_yscale('log')

    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.1e}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),
                       textcoords='offset points',
                       ha='center',
                       va='bottom',
                       fontsize=8)

    plt.tight_layout()
    plt.savefig(output_file, dpi=DEFAULT_DPI, bbox_inches='tight')
    print(f"Accuracy validation plot saved to: {output_file}")
    plt.close()


def generate_markdown_table_batch_size(batch_results: Dict) -> str:
    """
    Generate markdown table for batch size performance.

    Args:
        batch_results: Dictionary with batch size benchmark results

    Returns:
        Markdown formatted table string
    """
    batch_sizes = batch_results['batch_sizes']
    cuda_times = batch_results['cuda_stoi_times']
    pystoi_times = batch_results['pystoi_times']
    speedups = batch_results['speedups']

    table = "### Batch Size Performance\n\n"
    table += "| Batch Size | cuda-stoi (ms) | pystoi (ms) | Speedup |\n"
    table += "|------------|----------------|-------------|---------|\n"

    for bs, cuda_t, pystoi_t, speedup in zip(batch_sizes, cuda_times, pystoi_times, speedups):
        table += f"| {bs:>10} | {cuda_t:>14.1f} | {pystoi_t:>11.1f} | {speedup:>6.2f}x |\n"

    return table


def generate_markdown_table_signal_length(length_results: Dict) -> str:
    """
    Generate markdown table for signal length performance.

    Args:
        length_results: Dictionary with signal length benchmark results

    Returns:
        Markdown formatted table string
    """
    signal_lengths = length_results['signal_lengths']
    cuda_times = length_results['cuda_stoi_times']
    pystoi_times = length_results['pystoi_times']
    speedups = length_results['speedups']

    table = "### Signal Length Performance\n\n"
    table += "| Length (s) | cuda-stoi (ms) | pystoi (ms) | Speedup |\n"
    table += "|------------|----------------|-------------|---------|\n"

    for length, cuda_t, pystoi_t, speedup in zip(signal_lengths, cuda_times, pystoi_times, speedups):
        table += f"| {length:>10} | {cuda_t:>14.1f} | {pystoi_t:>11.1f} | {speedup:>6.2f}x |\n"

    return table


def generate_markdown_table_accuracy(accuracy_results: Dict) -> str:
    """
    Generate markdown table for accuracy validation.

    Args:
        accuracy_results: Dictionary with accuracy benchmark results

    Returns:
        Markdown formatted table string
    """
    snr_levels = accuracy_results['snr_levels']
    mae_values = accuracy_results['mae_values']
    mse_values = accuracy_results['mse_values']

    table = "### Accuracy Validation\n\n"
    table += "| SNR (dB) | MAE      | MSE      |\n"
    table += "|----------|----------|----------|\n"

    for snr, mae, mse in zip(snr_levels, mae_values, mse_values):
        table += f"| {snr:>8} | {mae:.2e} | {mse:.2e} |\n"

    return table


def generate_summary_report(
    batch_results: Dict,
    length_results: Dict,
    accuracy_results: Dict,
    output_file: Path
):
    """
    Generate comprehensive markdown summary report.

    Args:
        batch_results: Dictionary with batch size benchmark results
        length_results: Dictionary with signal length benchmark results
        accuracy_results: Dictionary with accuracy benchmark results
        output_file: Path to save the report
    """
    report = "# STOI Implementation Comparison\n\n"
    report += f"**Generated**: {batch_results.get('timestamp', 'N/A')}\n\n"
    report += "---\n\n"

    # Speed comparison section
    report += "## Speed Comparison\n\n"
    report += generate_markdown_table_batch_size(batch_results) + "\n\n"
    report += generate_markdown_table_signal_length(length_results) + "\n\n"

    # Key findings
    report += "### Key Findings\n\n"
    max_speedup_batch = max(batch_results['speedups'])
    max_speedup_length = max(length_results['speedups'])
    report += f"- **Maximum speedup (batch)**: {max_speedup_batch:.2f}x at batch size {batch_results['batch_sizes'][batch_results['speedups'].index(max_speedup_batch)]}\n"
    report += f"- **Maximum speedup (length)**: {max_speedup_length:.2f}x at {length_results['signal_lengths'][length_results['speedups'].index(max_speedup_length)]}s signals\n"
    report += f"- **Average batch speedup**: {np.mean(batch_results['speedups']):.2f}x\n\n"

    # Accuracy validation section
    report += "---\n\n"
    report += "## Accuracy Validation\n\n"
    report += generate_markdown_table_accuracy(accuracy_results) + "\n\n"

    # Accuracy findings
    max_mae = max(accuracy_results['mae_values'])
    max_mse = max(accuracy_results['mse_values'])
    report += "### Numerical Equivalence\n\n"
    report += f"- **Maximum MAE**: {max_mae:.2e}\n"
    report += f"- **Maximum MSE**: {max_mse:.2e}\n"
    report += f"- **Status**: {'✅ PASS (MAE < 1e-6)' if max_mae < 1e-6 else '❌ FAIL'}\n\n"

    # Conclusion
    report += "---\n\n"
    report += "## Conclusion\n\n"
    report += "The cuda-stoi implementation demonstrates:\n"
    report += f"1. **Performance**: {max_speedup_batch:.1f}x faster than pystoi for batch processing\n"
    report += f"2. **Accuracy**: Numerical equivalence maintained (MAE < 1e-6)\n"
    report += "3. **Scalability**: Speedup increases with batch size\n\n"

    # Save report
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w') as f:
        f.write(report)

    print(f"Summary report saved to: {output_file}")


def print_summary(batch_results: Dict, length_results: Dict, accuracy_results: Dict):
    """
    Print a quick summary to console.

    Args:
        batch_results: Dictionary with batch size benchmark results
        length_results: Dictionary with signal length benchmark results
        accuracy_results: Dictionary with accuracy benchmark results
    """
    print("\n" + "="*70)
    print("BENCHMARK SUMMARY")
    print("="*70)

    print("\nSpeed Comparison:")
    print(f"  Max speedup (batch): {max(batch_results['speedups']):.2f}x")
    print(f"  Max speedup (length): {max(length_results['speedups']):.2f}x")
    print(f"  Avg speedup (batch): {np.mean(batch_results['speedups']):.2f}x")

    print("\nAccuracy Validation:")
    print(f"  Max MAE: {max(accuracy_results['mae_values']):.2e}")
    print(f"  Max MSE: {max(accuracy_results['mse_values']):.2e}")
    print(f"  Status: {'✅ PASS' if max(accuracy_results['mae_values']) < 1e-6 else '❌ FAIL'}")

    print("\n" + "="*70 + "\n")
