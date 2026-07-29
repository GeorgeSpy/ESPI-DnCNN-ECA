#!/usr/bin/env python3
"""Validate internal consistency of the lightweight 2026 revision tables."""

from __future__ import annotations

import csv
import itertools
import math
import statistics
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results/revision_2026_corrected_robustness"
T_CRIT = {3: 4.3026527297, 5: 2.7764451052, 6: 2.5705818356}
TOLERANCE = 2e-9


def rows(name: str) -> list[dict[str, str]]:
    path = RESULTS / name
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def close(actual: float, expected: float, label: str) -> None:
    if not math.isclose(actual, expected, rel_tol=0.0, abs_tol=TOLERANCE):
        raise AssertionError(f"{label}: actual={actual} expected={expected}")


def mean_sd_ci(values: list[float]) -> tuple[float, float, float, float]:
    mean = statistics.mean(values)
    sd = statistics.stdev(values)
    half = T_CRIT[len(values)] * sd / math.sqrt(len(values))
    return mean, sd, mean - half, mean + half


def exact_sign_flip_p(values: list[float]) -> float:
    observed = abs(statistics.mean(values))
    outcomes = (
        abs(statistics.mean([sign * value for sign, value in zip(signs, values)]))
        for signs in itertools.product((-1.0, 1.0), repeat=len(values))
    )
    outcomes = list(outcomes)
    return sum(value >= observed - 1e-15 for value in outcomes) / len(outcomes)


def validate_seed_summary() -> None:
    manifest = rows("corrected_seed5_public_manifest.csv")
    summary = rows("corrected_seed5_summary.csv")
    if len(manifest) != 15 or len(summary) != 6:
        raise AssertionError("Unexpected seed-table row count")
    for result in summary:
        values = [
            float(row[result["metric"]])
            for row in manifest
            if row["model"] == result["model"]
        ]
        mean, sd, low, high = mean_sd_ci(values)
        close(mean, float(result["mean"]), f"seed mean {result}")
        close(sd, float(result["std_sample"]), f"seed sd {result}")
        close(low, float(result["ci95_low"]), f"seed ci low {result}")
        close(high, float(result["ci95_high"]), f"seed ci high {result}")


def validate_grouped_summary() -> None:
    board = rows("grouped_six_board_results.csv")
    summary = rows("grouped_six_board_summary.csv")
    if len(board) != 18 or len(summary) != 3:
        raise AssertionError("Unexpected grouped-table row count")
    for result in summary:
        selected = [row for row in board if row["model"] == result["model"]]
        for metric in ("accuracy", "macro_f1"):
            values = [float(row[metric]) for row in selected]
            close(statistics.mean(values), float(result[f"{metric}_mean"]), f"grouped mean {result['model']} {metric}")
            close(statistics.stdev(values), float(result[f"{metric}_sd"]), f"grouped sd {result['model']} {metric}")


def validate_unet_pairs() -> None:
    board = rows("unet_matched_epoch15_paired_board_effects.csv")
    summary = rows("unet_matched_epoch15_paired_summary.csv")
    if len(board) != 6 or len(summary) != 6:
        raise AssertionError("Unexpected U-Net table row count")
    scopes = {
        "all_6_boards": board,
        "carbon_3_boards": [row for row in board if row["material"] == "carbon"],
        "wood_3_boards": [row for row in board if row["material"] == "wood"],
    }
    for result in summary:
        metric = result["metric"]
        values = [float(row[f"delta_{metric}"]) for row in scopes[result["scope"]]]
        mean, sd, low, high = mean_sd_ci(values)
        close(mean, float(result["mean_delta"]), f"U-Net mean {result}")
        close(sd, float(result["sd_delta"]), f"U-Net sd {result}")
        close(low, float(result["ci95_low"]), f"U-Net ci low {result}")
        close(high, float(result["ci95_high"]), f"U-Net ci high {result}")
        close(exact_sign_flip_p(values), float(result["exact_sign_flip_p"]), f"U-Net sign flip {result}")
        if sum(value > 0 for value in values) != int(result["wins"]):
            raise AssertionError(f"U-Net win count {result}")
        if sum(value < 0 for value in values) != int(result["losses"]):
            raise AssertionError(f"U-Net loss count {result}")


def validate_unet_bn_pairs() -> None:
    board = rows("unet_bn_matched_epoch15_paired_board_effects.csv")
    summary = rows("unet_bn_matched_epoch15_paired_summary.csv")
    if len(board) != 6 or len(summary) != 2:
        raise AssertionError("Unexpected U-Net BN table row count")
    for result in summary:
        metric = result["metric"]
        values = [float(row[f"delta_{metric}"]) for row in board]
        mean, sd, low, high = mean_sd_ci(values)
        close(mean, float(result["mean_delta"]), f"U-Net BN mean {result}")
        close(sd, float(result["sd_delta"]), f"U-Net BN sd {result}")
        close(low, float(result["ci95_low"]), f"U-Net BN ci low {result}")
        close(high, float(result["ci95_high"]), f"U-Net BN ci high {result}")
        close(exact_sign_flip_p(values), float(result["exact_sign_flip_p"]), f"U-Net BN sign flip {result}")
        if sum(value > 0 for value in values) != int(result["wins"]):
            raise AssertionError(f"U-Net BN win count {result}")
        if sum(value < 0 for value in values) != int(result["losses"]):
            raise AssertionError(f"U-Net BN loss count {result}")


def validate_unet_normalization_context() -> None:
    bn = rows("unet_bn_matched_epoch15_paired_board_effects.csv")
    gn = rows("unet_matched_epoch15_paired_board_effects.csv")
    summary = rows("unet_eca_normalization_matched_epoch15_summary.csv")
    if len(bn) != 6 or len(gn) != 6 or len(summary) != 6:
        raise AssertionError("Unexpected U-Net normalization-context row count")
    if [row["board"] for row in bn] != [row["board"] for row in gn]:
        raise AssertionError("U-Net BN/GN board order mismatch")
    expected: dict[tuple[str, str], list[float]] = {}
    for metric in ("accuracy", "macro_f1"):
        bn_values = [float(row[f"delta_{metric}"]) for row in bn]
        gn_values = [float(row[f"delta_{metric}"]) for row in gn]
        expected[("BN_ECA3_minus_NoECA", metric)] = bn_values
        expected[("GN_ECA3_minus_NoECA", metric)] = gn_values
        expected[("GN_minus_BN_interaction", metric)] = [
            gn_value - bn_value
            for gn_value, bn_value in zip(gn_values, bn_values)
        ]
    for result in summary:
        values = expected[(result["effect"], result["metric"])]
        mean, sd, low, high = mean_sd_ci(values)
        close(mean, float(result["mean_delta"]), f"normalization mean {result}")
        close(sd, float(result["sd_delta"]), f"normalization sd {result}")
        close(low, float(result["ci95_low"]), f"normalization ci low {result}")
        close(high, float(result["ci95_high"]), f"normalization ci high {result}")
        close(exact_sign_flip_p(values), float(result["exact_sign_flip_p"]), f"normalization sign flip {result}")
        if sum(value > 0 for value in values) != int(result["wins"]):
            raise AssertionError(f"normalization win count {result}")
        if sum(value < 0 for value in values) != int(result["losses"]):
            raise AssertionError(f"normalization loss count {result}")


def validate_public_paths() -> None:
    for path in RESULTS.glob("*.csv"):
        text = path.read_text(encoding="utf-8")
        if ":\\" in text or "Mespi" in text or "MESPI" in text:
            raise AssertionError(f"Local/private path leaked into {path}")


def main() -> None:
    validate_seed_summary()
    validate_grouped_summary()
    validate_unet_pairs()
    validate_unet_bn_pairs()
    validate_unet_normalization_context()
    validate_public_paths()
    print("Revision result validation passed: seed5, grouped-six-board, matched U-Net, and normalization-interaction tables are consistent.")


if __name__ == "__main__":
    main()
