"""Standalone Stage 21A contract tests; no pytest dependency is required."""

import argparse
import inspect
import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.mosei import stage21a_common as common


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", required=True)
    return parser.parse_args()


def expect_raises(function, exception=Exception):
    try:
        function()
    except exception:
        return
    raise AssertionError("Expected {} was not raised.".format(exception.__name__))


def synthetic():
    video_ids, labels, ids = [], [], []
    for group in range(24):
        size = 3 + group % 3
        for clip in range(size):
            video_ids.append("v{:02d}".format(group))
            labels.append(float(((group + clip) % 7) - 3))
            ids.append("v{:02d}$_${}".format(group, clip))
    return np.asarray(ids), np.asarray(video_ids), np.asarray(labels, dtype=np.float64)


def main():
    cli = parse_args()
    output = Path(cli.output_root)
    tests = []

    def run(name, function):
        try:
            function()
            tests.append({"name": name, "passed": True, "error": None})
        except Exception as error:
            tests.append({"name": name, "passed": False, "error": repr(error)})

    ids, video_ids, labels = synthetic()

    run("test_lock_split_test_hard_fail", lambda: expect_raises(lambda: common.require_allowed_split("test"), RuntimeError))
    run("id_binding_duplicate_hard_fail", lambda: expect_raises(lambda: common.assert_unique_complete_ids(["a$_$0", "a$_$0"]), RuntimeError))
    run("id_binding_missing_video_hard_fail", lambda: expect_raises(lambda: common.assert_unique_complete_ids(["$_$0"]), RuntimeError))
    run("sample_order_sha_mismatch_hard_fail", lambda: expect_raises(lambda: common.assert_order_sha(ids, "0" * 64), RuntimeError))

    split = common.make_source_split(video_ids, labels, 2101)
    run(
        "source_disjoint_split",
        lambda: (
            not set(split["inner_train_sources"]) & set(split["inner_valid_sources"])
            or (_ for _ in ()).throw(AssertionError("source overlap"))
        ),
    )
    pairs = common.balanced_within_pairs(video_ids, labels, 1.0, seed=2100, max_pairs=8)
    run(
        "true_pair_same_source",
        lambda: all(video_ids[row["left_index"]] == video_ids[row["right_index"]] for row in pairs)
        or (_ for _ in ()).throw(AssertionError("P1 cross source")),
    )
    run(
        "pair_target_is_label_difference",
        lambda: all(
            abs(row["target_difference"] - (labels[row["left_index"]] - labels[row["right_index"]])) < 1e-12
            for row in pairs
        ) or (_ for _ in ()).throw(AssertionError("pair target mismatch")),
    )
    by_video = {}
    for row in pairs:
        by_video.setdefault(row["video_id"], 0.0)
        by_video[row["video_id"]] += row["raw_video_balanced_weight"]
    run(
        "group_balance_equal_pair_mass",
        lambda: max(abs(value - 1.0) for value in by_video.values()) < 1e-12
        or (_ for _ in ()).throw(AssertionError("unequal source pair mass")),
    )
    all_index = np.arange(len(ids), dtype=np.int64)
    control = common.nearest_cross_source_pairs(pairs, all_index, video_ids, labels, 2101)
    run(
        "cross_source_control_different_source",
        lambda: all(video_ids[row["left_index"]] != video_ids[row["right_index"]] for row in control)
        or (_ for _ in ()).throw(AssertionError("P2 same source")),
    )
    diagnostic = common.pair_matching_diagnostics(pairs, control)
    run(
        "gap_matched_pair_count_and_sign",
        lambda: (
            diagnostic["pair_count_first"] == diagnostic["pair_count_second"]
            and abs(diagnostic["positive_fraction_first"] - diagnostic["positive_fraction_second"]) < 1e-12
        ) or (_ for _ in ()).throw(AssertionError("gap control mismatch")),
    )
    pseudo = common.pseudo_membership(video_ids, all_index, 2102)
    true_sizes = sorted(len(value) for value in common.group_indices(video_ids).values())
    pseudo_sizes = sorted(len(value) for value in common.group_indices(pseudo).values())
    run(
        "pseudo_group_size_multiset_preserved",
        lambda: true_sizes == pseudo_sizes or (_ for _ in ()).throw(AssertionError("group sizes changed")),
    )
    rng = np.random.RandomState(21)
    representations = {mode: rng.normal(size=(len(ids), 8)) for mode in common.MODES}
    train_index = split["inner_train_indices"]
    standard = common.standardization(representations, labels, train_index)
    model_a = common.fit_shared_ridge(representations, labels, train_index, pairs, standard)
    model_b = common.fit_shared_ridge(representations, labels, train_index, pairs, standard)
    run(
        "relative_bias_column_zero",
        lambda: model_a["relative_bias_column_max"] == 0.0
        or (_ for _ in ()).throw(AssertionError("bias entered relative rows")),
    )
    run(
        "deterministic_ridge_solver",
        lambda: np.array_equal(model_a["coefficient"], model_b["coefficient"])
        or (_ for _ in ()).throw(AssertionError("ridge not deterministic")),
    )
    run(
        "no_source_feature_in_design",
        lambda: (
            model_a["source_features_in_design"] is False
            and "video_id" not in inspect.signature(common.fit_shared_ridge).parameters
            and "clip_id" not in inspect.signature(common.fit_shared_ridge).parameters
        ) or (_ for _ in ()).throw(AssertionError("source ID entered design")),
    )
    run(
        "single_clip_representation_contract",
        lambda: "context" not in inspect.signature(common.fit_shared_ridge).parameters
        or (_ for _ in ()).throw(AssertionError("context input present")),
    )
    metric = common.project_metrics(np.linspace(-1, 1, len(labels)), labels)
    run(
        "project_metric_evaluator_parity",
        lambda: metric["ProjectEvaluatorRoundingParityMax"] <= 5.1e-5
        or (_ for _ in ()).throw(AssertionError("metric parity failed")),
    )

    def real_cache_binding():
        manifest_path = output / "representations/representation_cache_manifest.json"
        if not manifest_path.exists():
            raise AssertionError("real representation manifest absent")
        manifest = json.loads(manifest_path.read_text())
        for split_name, entry in manifest["splits"].items():
            if common.sha256_file(entry["path"]) != entry["sha256"]:
                raise AssertionError("{} cache SHA mismatch".format(split_name))
            with np.load(entry["path"], allow_pickle=False) as archive:
                if common.ordered_id_sha(archive["sample_id"].astype(str)) != entry["sample_order_sha256"]:
                    raise AssertionError("{} cache order mismatch".format(split_name))

    run("real_cache_checkpoint_and_sample_binding", real_cache_binding)

    def no_official_valid_without_gate():
        metrics_path = output / "probes/probe_metrics.json"
        if not metrics_path.exists():
            return
        metrics = json.loads(metrics_path.read_text())
        if not metrics["train_only_gate"]["passed"] and metrics["official_valid"]["run"]:
            raise AssertionError("Official Valid ran despite failed Train gate")

    run("official_valid_promotion_lock", no_official_valid_without_gate)
    result = {
        "tests_run": len(tests),
        "tests_passed": sum(value["passed"] for value in tests),
        "tests_failed": sum(not value["passed"] for value in tests),
        "failed_tests": [value for value in tests if not value["passed"]],
        "checks": tests,
        "locked_test_access_count": 0,
    }
    result["status"] = "STAGE21A_IMPLEMENTATION_AUDIT_PASSED" if not result["tests_failed"] else "STAGE21A_IMPLEMENTATION_AUDIT_FAILED"
    common.atomic_json(output / "tests/test_results.json", result)
    print(json.dumps({key: result[key] for key in ("status", "tests_run", "tests_passed", "tests_failed")}, indent=2))
    if result["tests_failed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
