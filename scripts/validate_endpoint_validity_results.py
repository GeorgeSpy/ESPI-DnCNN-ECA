#!/usr/bin/env python3
"""Validate the public endpoint-validity result package.

This validator uses only Python's standard library. It checks the frozen public
counts, schemas, headline numerical results, leakage rows, paired contrasts,
and SHA-256 manifest. It does not access the private corpus or execute models.
"""

from __future__ import annotations

import csv
import hashlib
import math
from collections import defaultdict
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS = REPO_ROOT / "results" / "endpoint_validity_2026"
HASH_MANIFEST = RESULTS / "public_artifact_sha256.csv"


def rows(name: str) -> list[dict[str, str]]:
    with (RESULTS / name).open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def close(actual: float, expected: float, tolerance: float = 1e-12) -> None:
    if not math.isclose(actual, expected, rel_tol=0.0, abs_tol=tolerance):
        raise AssertionError(f"expected {expected:.15g}, got {actual:.15g}")


def keyed(records: list[dict[str, str]], field: str) -> dict[str, dict[str, str]]:
    result = {record[field]: record for record in records}
    if len(result) != len(records):
        raise AssertionError(f"duplicate key in {field}")
    return result


def validate_endpoint() -> None:
    endpoint = keyed(rows("endpoint_freeze_summary.csv"), "field")
    assert int(endpoint["candidate_physical_repeats"]["value"]) == 11706
    assert int(endpoint["primary_physical_repeats"]["value"]) == 11706
    assert int(endpoint["cross_project_label_differences"]["value"]) == 71
    assert int(endpoint["dncnn_label_corrections"]["value"]) == 0
    assert int(endpoint["unresolved_labels"]["value"]) == 0

    support = rows("class_support_by_board.csv")
    assert len(support) == 30
    assert {r["board"] for r in support} == {"C01", "C02", "C03", "W01", "W02", "W03"}
    assert {r["label_name"] for r in support} == {"1_1H", "1_1T", "1_2", "2_1", "higher"}
    assert sum(int(r["sample_count"]) for r in support) == 11706
    assert all(r["support_status"] == "PASS" for r in support)

    views = rows("view_lock_summary.csv")
    assert {r["view"] for r in views} == {"Raw", "V4R", "V5R"}
    assert all(int(r["materialized_count"]) == 11706 for r in views)
    assert all(r["determinism"] == "BITWISE_DETERMINISTIC" for r in views)
    assert all(float(r["historical_hard_prediction_agreement"]) == 1.0 for r in views)


def validate_incremental_results() -> None:
    summary = keyed(rows("board_balanced_summary.csv"), "arm")
    close(float(summary["F-only"]["board_balanced_macro_f1"]), 0.816656309694376)
    close(float(summary["Raw-F+I"]["board_balanced_macro_f1"]), 0.850355517750725)
    close(float(summary["V4R-F+I"]["board_balanced_macro_f1"]), 0.8006188363895909)
    close(float(summary["V5R-F+I"]["board_balanced_macro_f1"]), 0.7747161445980238)
    close(float(summary["nearest-frequency"]["board_balanced_macro_f1"]), 0.861236846454451)

    deltas = rows("incremental_deltas.csv")
    assert len(deltas) == 90
    assert all(r["level"] == "board_seed" for r in deltas)
    keys = {(r["board"], r["seed"], r["regime"]) for r in deltas}
    assert len(keys) == 90

    by_regime: dict[str, list[float]] = defaultdict(list)
    by_board_regime: dict[tuple[str, str], list[float]] = defaultdict(list)
    for record in deltas:
        value = float(record["delta_macro_f1"])
        by_regime[record["regime"]].append(value)
        by_board_regime[(record["board"], record["regime"])].append(value)

    expected = {"Raw": 0.033699208056349, "V4R": -0.016037473304785, "V5R": -0.041940165096352}
    expected_positive = {"Raw": 4, "V4R": 2, "V5R": 1}
    for regime, target in expected.items():
        close(sum(by_regime[regime]) / len(by_regime[regime]), target, 2e-12)
        board_means = [
            sum(by_board_regime[(board, regime)]) / len(by_board_regime[(board, regime)])
            for board in ("C01", "C02", "C03", "W01", "W02", "W03")
        ]
        assert sum(value > 0 for value in board_means) == expected_positive[regime]

    leakage = rows("leakage_audit.csv")
    assert len(leakage) == 90
    for record in leakage:
        assert record["status"] == "PASS"
        assert int(record["duplicate_predictions"]) == 0
        assert int(record["missing_predictions"]) == 0
        assert int(record["extra_predictions"]) == 0
        assert int(record["excluded_board_training_violations"]) == 0
        assert int(record["expected_development_samples"]) == int(record["actual_oof_samples"])


def validate_paired_results() -> None:
    paired = keyed(rows("paired_contrast_summary.csv"), "contrast")
    targets = {
        "Raw_minus_V4R": (0.049736681361, 6),
        "Raw_minus_V5R": (0.075639373153, 5),
        "V4R_minus_V5R": (0.025902691792, 4),
    }
    for contrast, (mean, positives) in targets.items():
        close(float(paired[contrast]["mean_macro_f1"]), mean)
        assert int(paired[contrast]["positive_boards"]) == positives

    boards = rows("paired_board_contrasts.csv")
    assert len(boards) == 6
    assert sum(r["raw_gt_v4r"].lower() == "true" for r in boards) == 6
    assert sum(r["raw_gt_v5r"].lower() == "true" for r in boards) == 5

    balanced = rows("condition_balanced_paired_contrasts.csv")
    assert len(balanced) == 6
    assert sum(r["raw_gt_v4r_cb"].lower() == "true" for r in balanced) == 6
    assert sum(r["raw_gt_v5r_cb"].lower() == "true" for r in balanced) == 5


def validate_hashes() -> None:
    manifest = rows(HASH_MANIFEST.name)
    assert manifest
    for record in manifest:
        path = REPO_ROOT / Path(record["path"])
        if not path.is_file():
            raise AssertionError(f"missing hashed artifact: {path}")
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != record["sha256"]:
            raise AssertionError(f"hash mismatch: {record['path']}")


def main() -> None:
    validate_endpoint()
    validate_incremental_results()
    validate_paired_results()
    validate_hashes()
    print("ENDPOINT_VALIDITY_PUBLIC_PACKAGE: PASS")
    print("PHYSICAL_REPEATS: 11706")
    print("LEAKAGE_ROWS: 90/90 PASS")
    print("ABSOLUTE_DECISION: DNCNN_INCREMENTAL_VALUE_INCONCLUSIVE")
    print("PAIRED_DECISION: RAW_CONDITIONAL_ADVANTAGE_CONSISTENT")


if __name__ == "__main__":
    main()

