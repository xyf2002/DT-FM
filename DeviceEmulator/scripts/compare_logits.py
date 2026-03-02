#!/usr/bin/env python3
"""
Compare logits from different runs (with/without hooks).

This script compares two logits files and provides detailed analysis:
- Overall statistics (mean, std, min, max)
- Element-wise differences (absolute and relative)
- Statistical significance tests
- Visualization of differences

Usage:
  python3 scripts/compare_logits.py --logits1 logits_comparison/logits_without_hooks.pt \
                                     --logits2 logits_comparison/logits_with_hooks.pt
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch


def load_logits(path: str) -> torch.Tensor:
    """Load logits from a .pt file"""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Logits file not found: {path}")
    logits = torch.load(path, map_location="cpu")
    return logits


def compute_statistics(tensor: torch.Tensor) -> dict:
    """Compute basic statistics for a tensor"""
    return {
        "shape": tuple(tensor.shape),
        "dtype": str(tensor.dtype),
        "mean": float(tensor.mean()),
        "std": float(tensor.std()),
        "min": float(tensor.min()),
        "max": float(tensor.max()),
        "median": float(torch.median(tensor)),
    }


def compare_logits(logits1: torch.Tensor, logits2: torch.Tensor) -> dict:
    """Compare two logits tensors and return detailed metrics"""

    # Check shapes match
    if logits1.shape != logits2.shape:
        raise ValueError(f"Shape mismatch: {logits1.shape} vs {logits2.shape}")

    # Compute element-wise differences
    diff = logits1 - logits2
    abs_diff = torch.abs(diff)

    # Compute relative differences (avoid division by zero)
    with torch.no_grad():
        safe_logits2 = torch.where(
            torch.abs(logits2) < 1e-8, torch.ones_like(logits2) * 1e-8, logits2
        )
        rel_diff = torch.abs(diff / safe_logits2)
        rel_diff = torch.where(
            torch.isnan(rel_diff) | torch.isinf(rel_diff),
            torch.zeros_like(rel_diff),
            rel_diff,
        )

    results = {
        "logits1_stats": compute_statistics(logits1),
        "logits2_stats": compute_statistics(logits2),
        "difference_stats": {
            "abs_mean": float(abs_diff.mean()),
            "abs_std": float(abs_diff.std()),
            "abs_min": float(abs_diff.min()),
            "abs_max": float(abs_diff.max()),
            "abs_median": float(torch.median(abs_diff)),
        },
        "relative_difference_stats": {
            "rel_mean": float(rel_diff.mean()),
            "rel_std": float(rel_diff.std()),
            "rel_min": float(rel_diff.min()),
            "rel_max": float(rel_diff.max()),
            "rel_median": float(torch.median(rel_diff)),
        },
    }

    # Count how many elements are "close" within different thresholds
    close_1e3 = torch.sum(abs_diff < 1e-3).item()
    close_1e4 = torch.sum(abs_diff < 1e-4).item()
    close_1e5 = torch.sum(abs_diff < 1e-5).item()
    total_elements = abs_diff.numel()

    results["close_tolerance"] = {
        "1e-3": {
            "count": close_1e3,
            "percentage": 100.0 * close_1e3 / total_elements,
        },
        "1e-4": {
            "count": close_1e4,
            "percentage": 100.0 * close_1e4 / total_elements,
        },
        "1e-5": {
            "count": close_1e5,
            "percentage": 100.0 * close_1e5 / total_elements,
        },
    }

    # Check if logits are "effectively" identical using torch.allclose
    results["allclose"] = {
        "rtol_1e3_atol_1e3": torch.allclose(
            logits1, logits2, rtol=1e-3, atol=1e-3
        ),
        "rtol_1e4_atol_1e4": torch.allclose(
            logits1, logits2, rtol=1e-4, atol=1e-4
        ),
        "rtol_1e5_atol_1e5": torch.allclose(
            logits1, logits2, rtol=1e-5, atol=1e-5
        ),
        "rtol_0_atol_1e3": torch.allclose(logits1, logits2, rtol=0, atol=1e-3),
        "rtol_0_atol_1e4": torch.allclose(logits1, logits2, rtol=0, atol=1e-4),
    }

    # Top-k accuracy comparison (useful for language models)
    for k in [1, 5, 10]:
        _, topk_indices_1 = torch.topk(logits1, k, dim=-1)
        _, topk_indices_2 = torch.topk(logits2, k, dim=-1)

        # Count how many top-k sets match
        match_count = torch.sum(topk_indices_1 == topk_indices_2).item()
        total_topk = topk_indices_1.numel()

        results[f"topk_{k}"] = {
            "match_count": match_count,
            "match_percentage": 100.0 * match_count / total_topk,
        }

    # Per-token difference (if applicable)
    if len(logits1.shape) >= 2:
        # Assume shape is (batch, seq_len, vocab_size) or similar
        mean_abs_diff_per_token = torch.mean(
            torch.abs(logits1 - logits2), dim=-1
        )
        results["per_token_diff"] = {
            "mean": float(mean_abs_diff_per_token.mean()),
            "std": float(mean_abs_diff_per_token.std()),
            "max": float(mean_abs_diff_per_token.max()),
            "min": float(mean_abs_diff_per_token.min()),
        }

    return results


def print_results(results: dict, logits1_path: str, logits2_path: str):
    """Pretty print comparison results"""
    print("\n" + "=" * 80)
    print("LOGITS COMPARISON REPORT")
    print("=" * 80)

    print(f"\nFile 1 (without hooks): {logits1_path}")
    print(f"File 2 (with hooks):    {logits2_path}")

    # Logits 1 statistics
    print("\n" + "-" * 80)
    print("LOGITS 1 STATISTICS (without hooks)")
    print("-" * 80)
    stats1 = results["logits1_stats"]
    print(f"  Shape:    {stats1['shape']}")
    print(f"  Dtype:    {stats1['dtype']}")
    print(f"  Mean:     {stats1['mean']:.6f}")
    print(f"  Std:      {stats1['std']:.6f}")
    print(f"  Min:      {stats1['min']:.6f}")
    print(f"  Max:      {stats1['max']:.6f}")
    print(f"  Median:   {stats1['median']:.6f}")

    # Logits 2 statistics
    print("\n" + "-" * 80)
    print("LOGITS 2 STATISTICS (with hooks)")
    print("-" * 80)
    stats2 = results["logits2_stats"]
    print(f"  Shape:    {stats2['shape']}")
    print(f"  Dtype:    {stats2['dtype']}")
    print(f"  Mean:     {stats2['mean']:.6f}")
    print(f"  Std:      {stats2['std']:.6f}")
    print(f"  Min:      {stats2['min']:.6f}")
    print(f"  Max:      {stats2['max']:.6f}")
    print(f"  Median:   {stats2['median']:.6f}")

    # Difference statistics
    print("\n" + "-" * 80)
    print("ABSOLUTE DIFFERENCE STATISTICS")
    print("-" * 80)
    diff_stats = results["difference_stats"]
    print(f"  Mean:     {diff_stats['abs_mean']:.10f}")
    print(f"  Std:      {diff_stats['abs_std']:.10f}")
    print(f"  Min:      {diff_stats['abs_min']:.10f}")
    print(f"  Max:      {diff_stats['abs_max']:.10f}")
    print(f"  Median:   {diff_stats['abs_median']:.10f}")

    # Relative difference statistics
    print("\n" + "-" * 80)
    print("RELATIVE DIFFERENCE STATISTICS")
    print("-" * 80)
    rel_stats = results["relative_difference_stats"]
    print(
        f"  Mean:     {rel_stats['rel_mean']:.10f} ({rel_stats['rel_mean'] * 100:.6f}%)"
    )
    print(f"  Std:      {rel_stats['rel_std']:.10f}")
    print(f"  Min:      {rel_stats['rel_min']:.10f}")
    print(f"  Max:      {rel_stats['rel_max']:.10f}")
    print(
        f"  Median:   {rel_stats['rel_median']:.10f} ({rel_stats['rel_median'] * 100:.6f}%)"
    )

    # Elements within tolerance
    print("\n" + "-" * 80)
    print("ELEMENTS WITHIN TOLERANCE")
    print("-" * 80)
    close_tol = results["close_tolerance"]
    for tolerance, data in close_tol.items():
        print(
            f"  Tolerance {tolerance}: {data['count']:>10} / {stats1['shape'][0] * stats1['shape'][1] * stats1['shape'][2] if len(stats1['shape']) == 3 else stats1['shape'][0] * stats1['shape'][1]:>10} ({data['percentage']:>6.2f}%)"
        )

    # Allclose results
    print("\n" + "-" * 80)
    print("TORCH.ALLCLOSE TEST RESULTS")
    print("-" * 80)
    allclose = results["allclose"]
    for test_name, passed in allclose.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {test_name}: {status}")

    # Top-k accuracy
    print("\n" + "-" * 80)
    print("TOP-K ACCURACY COMPARISON")
    print("-" * 80)
    for k in [1, 5, 10]:
        if f"topk_{k}" in results:
            topk_data = results[f"topk_{k}"]
            print(
                f"  Top-{k:>2}: {topk_data['match_percentage']:>6.2f}% of positions have matching top-k tokens"
            )

    # Per-token difference
    if "per_token_diff" in results:
        print("\n" + "-" * 80)
        print("PER-TOKEN DIFFERENCE (averaged over vocabulary)")
        print("-" * 80)
        per_token = results["per_token_diff"]
        print(f"  Mean:     {per_token['mean']:.10f}")
        print(f"  Std:      {per_token['std']:.10f}")
        print(f"  Max:      {per_token['max']:.10f}")
        print(f"  Min:      {per_token['min']:.10f}")

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    if allclose["rtol_1e3_atol_1e3"]:
        print("✓ Logits are EFFECTIVELY IDENTICAL (rtol=1e-3, atol=1e-3)")
    elif allclose["rtol_1e4_atol_1e4"]:
        print(
            "⚠ Logits are VERY CLOSE (rtol=1e-4, atol=1e-4) but not identical"
        )
    elif allclose["rtol_1e5_atol_1e5"]:
        print(
            "⚠ Logits are CLOSE (rtol=1e-5, atol=1e-5) but have some differences"
        )
    else:
        print("✗ Logits have NOTABLE DIFFERENCES")

    top1_match = results.get("topk_1", {}).get("match_percentage", 0)
    print(f"  Top-1 token match: {top1_match:.2f}%")
    print(f"  Max absolute difference: {diff_stats['abs_max']:.10f}")
    print(f"  Mean absolute difference: {diff_stats['abs_mean']:.10f}")
    print("=" * 80 + "\n")


def save_results_to_file(results: dict, output_path: str):
    """Save comparison results to a JSON file"""
    import json

    # Convert tensors to native Python types for JSON serialization
    def convert_to_serializable(obj):
        if isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [convert_to_serializable(item) for item in obj]
        elif isinstance(obj, (np.floating, np.integer)):
            return float(obj)
        elif isinstance(obj, bool):
            return bool(obj)
        return obj

    with open(output_path, "w") as f:
        json.dump(convert_to_serializable(results), f, indent=2)
    print(f"✓ Results saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Compare logits from different runs"
    )
    parser.add_argument(
        "--logits1",
        type=str,
        default="logits_comparison/logits_without_hooks.pt",
        help="Path to first logits file (without hooks)",
    )
    parser.add_argument(
        "--logits2",
        type=str,
        default="logits_comparison/logits_with_hooks.pt",
        help="Path to second logits file (with hooks)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="logits_comparison/comparison_report.json",
        help="Path to save comparison report as JSON",
    )
    parser.add_argument(
        "--no-save", action="store_true", help="Do not save results to file"
    )

    args = parser.parse_args()

    print("Loading logits files...")
    try:
        logits1 = load_logits(args.logits1)
        logits2 = load_logits(args.logits2)
    except FileNotFoundError as e:
        print(f"✗ Error: {e}")
        sys.exit(1)

    print(f"  Logits 1: {logits1.shape}")
    print(f"  Logits 2: {logits2.shape}")

    print("Comparing logits...")
    results = compare_logits(logits1, logits2)

    print_results(results, args.logits1, args.logits2)

    if not args.no_save:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        save_results_to_file(results, args.output)


if __name__ == "__main__":
    main()
