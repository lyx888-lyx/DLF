import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pandas as pd
import torch

import aggregate_cf_compat_multiseed as aggregate
import run_cf_compat_multiseed as runner
import train_cf_compat_kd
from trains.singleTask.cf_compat_kd_utils import (
    CACHE_COLUMNS,
    MULTISEED_CACHE_VERSION,
    cache_paths,
    load_counterfactual_cache,
    locate_stage1_evaluator,
    write_counterfactual_cache,
)


def tiny_cache():
    rows = []
    for i in range(4):
        row = {"sample_index": i, "sample_id": "id{}".format(i), "label": float(i),
               "evaluator_LAV_pred": 1., "evaluator_LA_pred": .8,
               "evaluator_LV_pred": .7, "evaluator_L_pred": .5}
        for mode in ("LA", "LV", "L"):
            row["delta_{}".format(mode)] = i + .1
            row["rank_{}".format(mode)] = i + 1.
            row["q_{}".format(mode)] = (i + .5) / 4
            row["compat_{}".format(mode)] = 1 - (i + .5) / 4
        rows.append(row)
    return pd.DataFrame(rows).loc[:, CACHE_COLUMNS]


def fake_main_row(seed, j, checkpoint):
    row = {"Seed": seed, "BestValidEpoch": 2, "J_valid": j,
           "J_test_at_valid_best": j + .1, "BestObservedTestEpoch": 1,
           "BestObservedTestJ": j + .05, "MainCheckpoint": str(checkpoint)}
    for mode in aggregate.MODES:
        for metric in aggregate.METRICS:
            row["test_at_valid_best_{}_{}".format(mode, metric)] = .7
    return pd.Series(row)


class MultiseedProtocolTests(unittest.TestCase):
    def test_fixed_seed_lists_and_locked_seed(self):
        self.assertEqual(runner.ALL_SEEDS, (1111, 1112, 1113, 1114, 1115))
        self.assertEqual(runner.NEW_SEEDS, (1112, 1113, 1114, 1115))
        self.assertNotIn(1111, runner.NEW_SEEDS)

    def test_cli_rejects_seed1111_and_nonfixed_seed(self):
        for seed in (1111, 1116):
            with self.assertRaises(SystemExit):
                runner.parse_args(["--action", "moddrop", "--seed", str(seed)])

    def test_cli_rejects_reliability_gate_and_tuning(self):
        for args in (["--gate-mode", "reliability_compat"], ["--eta", ".5"], ["--lambda-kd", ".5"]):
            with self.assertRaises(SystemExit):
                runner.parse_args(["--action", "moddrop", "--seed", "1112", *args])

    def test_cli_allows_exactly_one_new_seed_process(self):
        cli = runner.parse_args(["--action", "cfcompat", "--seed", "1112"])
        self.assertEqual(cli.seed, 1112)
        self.assertEqual(cli.gate_mode, "compat")

    def test_smoke_is_exactly_two_epochs(self):
        cli = runner.parse_args(["--action", "moddrop", "--seed", "1112", "--smoke-test"])
        self.assertEqual(cli.max_epochs, 2)

    def test_gate3_log_parser_maps_runs_to_fixed_seed_order(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "gate.log"
            lines = []
            for index in range(1, 6):
                lines += ["Epoch: 1 TRAIN -(DLF) [1/1/{}]".format(index), "VAL-(DLF) >> Loss: 1.0",
                          "Epoch: 2 TRAIN -(DLF) [1/2/{}]".format(index), "VAL-(DLF) >> Loss: 0.5"]
            path.write_text("\n".join(lines))
            self.assertEqual(runner.parse_gate3_best_epochs(path), {seed: 2 for seed in runner.ALL_SEEDS})

    def test_seeded_cache_path_is_isolated_from_stage3b(self):
        old = cache_paths("root", "mosi")
        new = cache_paths("root", "mosi", MULTISEED_CACHE_VERSION, 1112)
        self.assertNotEqual(old["directory"], new["directory"])
        self.assertIn("seed1112", str(new["directory"]))

    def test_multiseed_cache_metadata_and_sha_binding(self):
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = Path(tmp) / "e.pth"; torch.save({"x": torch.tensor(1)}, checkpoint)
            paths, config = write_counterfactual_cache(tiny_cache(), tmp, "mosi", checkpoint, 3,
                                                        version=MULTISEED_CACHE_VERSION, seed=1112)
            self.assertEqual(config["seed"], 1112)
            self.assertTrue(config["created_from_train_only"])
            self.assertTrue(config["rng_state_preserved"])
            self.assertEqual(len(config["cache_sha256"]), 64)
            frame, _ = load_counterfactual_cache(tmp, "mosi", MULTISEED_CACHE_VERSION, 1112,
                                                  expected_evaluator_sha=config["evaluator_sha256"])
            self.assertEqual(len(frame), 4)
            with self.assertRaises(ValueError):
                load_counterfactual_cache(tmp, "mosi", MULTISEED_CACHE_VERSION, 1112, "wrong")

    def test_cache_manifest_records_locked_and_same_seed_cache_provenance(self):
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = Path(tmp) / "e.pth"; torch.save({"x": torch.tensor(1)}, checkpoint)
            for seed, version, cache_seed in ((1111, "cf_compat_v1", None),
                                               (1112, MULTISEED_CACHE_VERSION, 1112)):
                _, config = write_counterfactual_cache(
                    pd.concat([tiny_cache()] * 321, ignore_index=True).assign(sample_index=range(1284)),
                    tmp, "mosi", checkpoint, 3, version=version, seed=cache_seed)
                self.assertEqual(config["train_sample_count"], 1284)
            rows = runner.refresh_cache_manifest(tmp, "mosi")
            self.assertEqual([row["Seed"] for row in rows], [1111, 1112])
            manifest = pd.read_csv(Path(tmp) / "counterfactual_compatibility" /
                                   MULTISEED_CACHE_VERSION / "mosi" / "cache_manifest.csv")
            self.assertEqual(manifest.TrainSampleCount.tolist(), [1284, 1284])
            self.assertTrue(manifest.CreatedFromTrainOnly.all())

    def test_cache_requires_unique_complete_sample_indices(self):
        source = Path("trains/singleTask/cf_compat_kd_utils.py").read_text()
        self.assertIn("frame.sample_index.duplicated().any()", source)
        self.assertIn("np.arange(len(frame))", source)

    def test_same_seed_moddrop_evaluator_is_located_from_main_csv(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); checkpoint = root / "valid.pth"; torch.save({"x": torch.tensor(1)}, checkpoint)
            source = root / "missing_baseline/moddrop_benchmark_multiseed_v1/seed1112"
            source.mkdir(parents=True)
            pd.DataFrame([{"Seed": 1112, "BestValidEpoch": 7, "MainCheckpoint": str(checkpoint)}]).to_csv(source / "mosi_per_seed.csv", index=False)
            got, epoch, _ = locate_stage1_evaluator(root, "mosi", 1112, multiseed=True)
            self.assertEqual(got, checkpoint); self.assertEqual(epoch, 7)

    def test_test_best_evaluator_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); checkpoint = root / "diagnostic/best_test.pth"
            checkpoint.parent.mkdir(); torch.save({"x": torch.tensor(1)}, checkpoint)
            source = root / "missing_baseline/moddrop_benchmark_multiseed_v1/seed1112"
            source.mkdir(parents=True)
            pd.DataFrame([{"Seed": 1112, "BestValidEpoch": 7, "MainCheckpoint": str(checkpoint)}]).to_csv(source / "mosi_per_seed.csv", index=False)
            with self.assertRaises(ValueError):
                locate_stage1_evaluator(root, "mosi", 1112, multiseed=True)

    def test_moddrop_and_cf_paths_are_seed_isolated(self):
        cli = SimpleNamespace(result_root="result", model_save_dir="pt", smoke_test=False)
        r1, m1, d1 = runner.moddrop_paths(cli, 1112)
        r2, m2, _ = runner.moddrop_paths(cli, 1113)
        self.assertNotEqual(r1, r2); self.assertNotEqual(m1, m2); self.assertIn("diagnostic", str(d1))
        cf = SimpleNamespace(gate_mode="compat", result_root="result", model_save_dir="pt",
                             smoke_test=False, multiseed_replication=True, seeds=[1112])
        result, main, diagnostic = train_cf_compat_kd.method_paths(cf, "mosi", 1112)
        self.assertIn("benchmark_multiseed", str(result)); self.assertIn("seed1112", str(main))
        self.assertNotEqual(main, diagnostic)

    def test_seed1111_locked_manifest_entry_is_reused_not_copied(self):
        source = Path("run_cf_compat_multiseed.py").read_text()
        self.assertIn('"Status": "locked_existing"', source)
        self.assertIn('"Source": "reused_locked_existing_result"', source)
        self.assertNotIn("train_moddrop_benchmark(cli, logger, 1111)", source)

    def test_teacher_student_same_seed_initialization_guard_remains(self):
        source = Path("train_cf_compat_kd.py").read_text()
        self.assertIn("clean_checkpoint_path(cli.model_save_dir, args.dataset_name, seed)", source)
        self.assertIn("assert_initial_lav_equivalence", source)
        self.assertIn("build_frozen_teacher", source)

    def test_teacher_and_evaluator_freeze_guards_remain(self):
        cf_source = Path("train_cf_compat_kd.py").read_text()
        util_source = Path("trains/singleTask/cf_compat_kd_utils.py").read_text()
        self.assertIn("assert_teacher_not_in_optimizer", cf_source)
        self.assertIn("teacher_grad_count(teacher)", cf_source)
        self.assertIn("freeze_teacher(evaluator)", util_source)
        self.assertIn("torch.inference_mode()", util_source)

    def test_missing_rng_formula_is_fixed_for_both_methods(self):
        for path in ("run_cf_compat_multiseed.py", "train_cf_compat_kd.py"):
            self.assertIn("104729", Path(path).read_text())

    def test_first_epoch_counts_are_recorded_not_hardcoded_for_new_seeds(self):
        source = Path("run_cf_compat_multiseed.py").read_text()
        self.assertIn('"FirstEpochCount_LA"', source)
        self.assertNotIn('counts != Counter({"LA": 435', source)
        cf_source = Path("train_cf_compat_kd.py").read_text()
        self.assertIn("int(seed) == 1111", cf_source)

    def test_stage3b_formula_and_rank_transform_are_unchanged(self):
        source = Path("trains/singleTask/cf_compat_kd_utils.py").read_text()
        self.assertIn('RANK_TRANSFORM = "(rank-0.5)/N"', source)
        self.assertIn("compat = 1.0 - q", source)
        self.assertIn('smooth_l1_loss(student, teacher, reduction="none")', source)
        self.assertIn("torch.sum(weight * each) / (torch.sum(weight) + 1e-8)", source)

    def test_moddrop_formula_has_no_kd_or_lds(self):
        source = runner.train_moddrop_benchmark.__code__.co_names
        self.assertNotIn("gated_kd_loss", source)
        self.assertNotIn("lds", source)
        text = Path("run_cf_compat_multiseed.py").read_text()
        self.assertIn("loss = full_loss + missing_loss", text)

    def test_main_and_diagnostic_selection_are_separate(self):
        source = Path("run_cf_compat_multiseed.py").read_text()
        for token in ("is_best_valid", "is_best_test", "main_checkpoint", "diagnostic_checkpoint",
                      '"selected_by"] = "validation"', '"selected_by"] = "test"'):
            self.assertIn(token, source)

    def test_cfcompat_multiseed_rejects_multi_seed_and_wrong_gate(self):
        with mock.patch.object(sys, "argv", ["x", "--multiseed-replication", "--seeds", "1112", "1113"]):
            with self.assertRaises(SystemExit):
                train_cf_compat_kd.parse_args()
        with mock.patch.object(sys, "argv", ["x", "--multiseed-replication", "--seeds", "1112", "--gate-mode", "reliability_compat"]):
            with self.assertRaises(SystemExit):
                train_cf_compat_kd.parse_args()

    def test_formal_phase_order_requires_all_controls_before_cache(self):
        manifest = {"Seeds": [{"Seed": 1111, "Status": "locked_existing"}] +
                             [{"Seed": s, "Status": "pending", "ModDropCheckpoint": None,
                               "CachePath": None} for s in runner.NEW_SEEDS]}
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(runner, "RUN_MANIFEST", Path(tmp) / "run.json"):
            runner.RUN_MANIFEST.write_text(json.dumps(manifest))
            with self.assertRaises(RuntimeError):
                runner._formal_phase_guard("cache", 1112)

    def test_aggregator_rejects_duplicate_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "x.csv"
            pd.DataFrame([{"Seed": 1112}, {"Seed": 1112}]).to_csv(path, index=False)
            with self.assertRaises(ValueError):
                aggregate.one_row(path, 1112)

    def test_markdown_export_has_no_optional_tabulate_dependency(self):
        rendered = aggregate.markdown_table(pd.DataFrame([{"Seed": 1112, "Delta": -0.125}]))
        self.assertIn("| Seed | Delta |", rendered)
        self.assertIn("| 1112 | -0.125 |", rendered)
        self.assertNotIn("to_markdown", Path("aggregate_cf_compat_multiseed.py").read_text())

    def test_aggregator_rejects_diagnostic_as_main(self):
        row = pd.Series({"MainCheckpoint": "diagnostic/best_test.pth", "BestValidEpoch": 1,
                         "J_valid": .5, "J_test_at_valid_best": .6})
        with self.assertRaises(ValueError):
            aggregate.validate_main_row(row, "x", 1112)

    def test_paired_delta_uses_matching_seed_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = Path(tmp) / "x.pth"; checkpoint.write_bytes(b"x")
            controls = {s: fake_main_row(s, .6 + s / 100000, checkpoint) for s in runner.ALL_SEEDS}
            cfs = {s: fake_main_row(s, controls[s].J_valid - .01, checkpoint) for s in runner.ALL_SEEDS}
            with mock.patch.object(aggregate, "load_control", side_effect=lambda s: controls[s]), \
                    mock.patch.object(aggregate, "load_cf", side_effect=lambda s: cfs[s]):
                frame = aggregate.build_paired_rows()
            self.assertEqual(frame.Seed.tolist(), list(runner.ALL_SEEDS))
            np.testing.assert_allclose(frame.Delta_J_valid, -.01)

    def test_aggregator_refuses_incomplete_manifest(self):
        manifest = {"Seeds": [{"Seed": s, "Status": "pending"} for s in runner.ALL_SEEDS]}
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(aggregate, "RUN_MANIFEST", Path(tmp) / "run.json"):
            aggregate.RUN_MANIFEST.write_text(json.dumps(manifest))
            with self.assertRaises(RuntimeError):
                aggregate.build_checkpoint_manifest()

    def test_primary_summary_never_substitutes_diagnostic(self):
        source = Path("aggregate_cf_compat_multiseed.py").read_text()
        self.assertIn("Delta_J_test_main", source)
        self.assertIn("J_test_at_valid_best", source)
        self.assertNotIn("Delta_J_test_diagnostic.mean", source)

    def test_required_outputs_and_gate_stability_fields_are_registered(self):
        source = Path("aggregate_cf_compat_multiseed.py").read_text()
        for name in ("paired_seed_results.csv", "multiseed_summary.json", "multiseed_summary.md",
                     "gate_stability_across_seeds.csv", "checkpoint_manifest.csv",
                     "delta_p90", "delta_p99", "tie_fraction", "ESS_fraction",
                     "Spearman_delta_compat", "leave_one_seed_out_mean_delta_J_test"):
            self.assertIn(name, source)

    def test_run_outputs_required_files_per_seed(self):
        source = Path("run_cf_compat_multiseed.py").read_text()
        for name in ("mosi_per_seed.csv", "mosi_epoch_metrics.csv", "mosi_best_valid_predictions.csv",
                     "mosi_best_test_diagnostic_predictions.csv", "mosi_gate_summary.csv",
                     "mosi_gate_quartiles.csv"):
            self.assertIn(name, source)

    def test_formal_existing_outputs_are_not_overwritten(self):
        source = Path("run_cf_compat_multiseed.py").read_text()
        self.assertIn("Formal ModDrop output already exists", source)
        self.assertIn("Formal CFCompat output already exists", source)

    def test_stage3c_is_explicitly_not_started(self):
        source = Path("aggregate_cf_compat_multiseed.py").read_text(encoding="utf-8")
        self.assertIn("Stage 3C/recoverability 尚未开始", source)
        self.assertIn('"stage3c_recoverability_started": False', source)


if __name__ == "__main__":
    unittest.main()
