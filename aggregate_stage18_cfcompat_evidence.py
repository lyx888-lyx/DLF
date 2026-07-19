"""Build the Stage 18G novelty files and final evidence audit."""

import json
import subprocess
from pathlib import Path

import pandas as pd


ROOT = Path("result/missing_baseline/cfcompat_evidence_v1/mosi")
STAGE_A = ROOT / "stage18a_training_recovery"
STAGE_B = ROOT / "stage18b_student_learnability"
STAGE_C = ROOT / "stage18c_seed1114_controls"
STAGE_D = ROOT / "stage18d_seed1111_replication"
STAGE_E = ROOT / "stage18e_transfer_mechanism"
STAGE_F = ROOT / "stage18f_five_seed_benchmark"
STAGE_G = ROOT / "stage18g_novelty"
FINAL = ROOT / "final"


LITERATURE_COLUMNS = [
    "paper",
    "venue_year",
    "dataset",
    "backbone_text_features",
    "complete_teacher_type",
    "student_count",
    "response_kd",
    "feature_kd",
    "relation_kd",
    "prototype_kd",
    "self_distillation",
    "reliability_difficulty_weighting",
    "student_state_weighting",
    "sample_level_weighting",
    "missing_pattern_level_weighting",
    "train_label_constructs_transferability",
    "explicit_negative_transfer_test",
    "oracle",
    "reported_metrics",
    "fair_direct_comparison",
    "source_url",
    "notes",
]


def literature_rows():
    # UNKNOWN is deliberate whenever the inspected primary source did not
    # establish the requested detail.
    return [
        {
            "paper": "CFCompatKD (this Stage 18 evidence study)",
            "venue_year": "UNPUBLISHED LOCAL STUDY, 2026",
            "dataset": "CMU-MOSI",
            "backbone_text_features": "DLF; repository-fixed processed features",
            "complete_teacher_type": "same-seed frozen clean-LAV DLF",
            "student_count": "one unified student per seed",
            "response_kd": "YES",
            "feature_kd": "NO",
            "relation_kd": "NO",
            "prototype_kd": "NO",
            "self_distillation": "NO",
            "reliability_difficulty_weighting": "compatibility gate",
            "student_state_weighting": "YES; evaluator counterfactual state",
            "sample_level_weighting": "YES",
            "missing_pattern_level_weighting": "YES",
            "train_label_constructs_transferability": "YES; frozen evaluator was train-only",
            "explicit_negative_transfer_test": "YES; Stage 18E",
            "oracle": "YES; train-label-only control",
            "reported_metrics": "J_valid, MAE, Corr, Acc7/5/2, F1; transfer diagnostics",
            "fair_direct_comparison": "TARGET STUDY",
            "source_url": "LOCAL",
            "notes": "Two-seed core gate failed; no five-seed or Locked Test.",
        },
        {
            "paper": "Correlation-Decoupled Knowledge Distillation for Multimodal Sentiment Analysis with Incomplete Modalities (CorrKD)",
            "venue_year": "CVPR 2024",
            "dataset": "MOSI; MOSEI; IEMOCAP",
            "backbone_text_features": "Transformer; GloVe/COVAREP/Facet",
            "complete_teacher_type": "frozen complete-modality teacher; same architecture as student",
            "student_count": "one",
            "response_kd": "YES; response-disentangled consistency",
            "feature_kd": "YES; sample contrastive",
            "relation_kd": "YES; cross-sample/cross-category/cross-target",
            "prototype_kd": "YES; category-guided",
            "self_distillation": "NO",
            "reliability_difficulty_weighting": "NO CONFIRMED",
            "student_state_weighting": "NO CONFIRMED",
            "sample_level_weighting": "sample-level contrastive, not confirmed scalar weighting",
            "missing_pattern_level_weighting": "NO CONFIRMED",
            "train_label_constructs_transferability": "NO",
            "explicit_negative_transfer_test": "NO CONFIRMED",
            "oracle": "NO CONFIRMED",
            "reported_metrics": "weighted F1 for MOSI/MOSEI; per-class F1 for IEMOCAP",
            "fair_direct_comparison": "NO; different backbone, features, objective, and protocol",
            "source_url": "https://openaccess.thecvf.com/content/CVPR2024/html/Li_Correlation-Decoupled_Knowledge_Distillation_for_Multimodal_Sentiment_Analysis_with_Incomplete_Modalities_CVPR_2024_paper.html",
            "notes": "Direct missing-modality MSA KD neighbor.",
        },
        {
            "paper": "CMAD: Correlation-Aware and Modalities-Aware Distillation for Multimodal Sentiment Analysis with Missing Modalities",
            "venue_year": "ICCV 2025",
            "dataset": "MOSEI; IEMOCAP; MUStARD; UR-FUNNY; CHERMA",
            "backbone_text_features": "Perceiver/Transformer fusion; feature details UNKNOWN",
            "complete_teacher_type": "teacher and student use identical architectures",
            "student_count": "one",
            "response_kd": "YES",
            "feature_kd": "YES; multi-level alignment",
            "relation_kd": "YES; feature similarity and high-order correlation",
            "prototype_kd": "NO CONFIRMED",
            "self_distillation": "NO CONFIRMED",
            "reliability_difficulty_weighting": "YES; modality difficulty curriculum",
            "student_state_weighting": "NO CONFIRMED",
            "sample_level_weighting": "NO CONFIRMED",
            "missing_pattern_level_weighting": "modality-aware",
            "train_label_constructs_transferability": "NO CONFIRMED",
            "explicit_negative_transfer_test": "NO CONFIRMED",
            "oracle": "NO CONFIRMED",
            "reported_metrics": "dataset-specific classification metrics",
            "fair_direct_comparison": "NO; different datasets/backbone/protocol",
            "source_url": "https://openaccess.thecvf.com/content/ICCV2025/html/Zhuang_CMAD_Correlation-Aware_and_Modalities-Aware_Distillation_for_Multimodal_Sentiment_Analysis_with_ICCV_2025_paper.html",
            "notes": "Direct modality-aware KD neighbor.",
        },
        {
            "paper": "Enhance-then-Balance Modality Collaboration for Robust Multimodal Sentiment Analysis (EBMC)",
            "venue_year": "CVPR 2026",
            "dataset": "MOSI; MOSEI; IEMOCAP",
            "backbone_text_features": "BERT-base text features; custom EBMC",
            "complete_teacher_type": "modality teacher outputs; not a single clean-LAV teacher",
            "student_count": "fused student",
            "response_kd": "YES; confidence-weighted KL",
            "feature_kd": "YES",
            "relation_kd": "NO CONFIRMED",
            "prototype_kd": "NO CONFIRMED",
            "self_distillation": "NO CONFIRMED",
            "reliability_difficulty_weighting": "YES; confidence and normalized reliability",
            "student_state_weighting": "NO CONFIRMED",
            "sample_level_weighting": "YES; instance-aware modality trust",
            "missing_pattern_level_weighting": "YES; per modality/sample",
            "train_label_constructs_transferability": "NO CONFIRMED",
            "explicit_negative_transfer_test": "NO CONFIRMED",
            "oracle": "NO CONFIRMED",
            "reported_metrics": "Acc-2 and F1; robustness under modality missingness",
            "fair_direct_comparison": "NO; different method/backbone/protocol",
            "source_url": "https://openaccess.thecvf.com/content/CVPR2026/html/He_Enhance-then-Balance_Modality_Collaboration_for_Robust_Multimodal_Sentiment_Analysis_CVPR_2026_paper.html",
            "notes": "Direct prior art against broad sample-level reliability KD novelty.",
        },
        {
            "paper": "Contrastive Knowledge Distillation for Robust Multimodal Sentiment Analysis (MM-CKD)",
            "venue_year": "arXiv:2410.08692, 2024",
            "dataset": "incomplete video sentiment; exact datasets in inspected abstract UNKNOWN",
            "backbone_text_features": "UNKNOWN",
            "complete_teacher_type": "complete multimodal teacher",
            "student_count": "multiple incomplete-modality students",
            "response_kd": "NO CONFIRMED",
            "feature_kd": "YES; multi-view supervised contrastive",
            "relation_kd": "YES; cross-sample",
            "prototype_kd": "NO CONFIRMED",
            "self_distillation": "online joint teacher/student learning",
            "reliability_difficulty_weighting": "NO CONFIRMED",
            "student_state_weighting": "NO CONFIRMED",
            "sample_level_weighting": "NO CONFIRMED",
            "missing_pattern_level_weighting": "separate students by available modality subset",
            "train_label_constructs_transferability": "NO",
            "explicit_negative_transfer_test": "NO CONFIRMED",
            "oracle": "NO CONFIRMED",
            "reported_metrics": "UNKNOWN from inspected abstract",
            "fair_direct_comparison": "NO; preprint and different objective/protocol",
            "source_url": "https://arxiv.org/abs/2410.08692",
            "notes": "Direct incomplete-MSA distillation neighbor.",
        },
        {
            "paper": "MissModal: Increasing Robustness to Missing Modality in Multimodal Sentiment Analysis",
            "venue_year": "TACL 2023",
            "dataset": "two public MSA datasets; names UNKNOWN from inspected abstract",
            "backbone_text_features": "fusion-agnostic representation constraints",
            "complete_teacher_type": "NONE",
            "student_count": "N/A",
            "response_kd": "NO",
            "feature_kd": "NO; representation alignment rather than teacher KD",
            "relation_kd": "geometric/distribution/semantic constraints",
            "prototype_kd": "NO CONFIRMED",
            "self_distillation": "NO CONFIRMED",
            "reliability_difficulty_weighting": "NO CONFIRMED",
            "student_state_weighting": "NO",
            "sample_level_weighting": "NO CONFIRMED",
            "missing_pattern_level_weighting": "handles various missing scenarios",
            "train_label_constructs_transferability": "NO",
            "explicit_negative_transfer_test": "NO CONFIRMED",
            "oracle": "NO",
            "reported_metrics": "UNKNOWN from inspected abstract",
            "fair_direct_comparison": "NO; non-KD method/protocol",
            "source_url": "https://aclanthology.org/2023.tacl-1.94/",
            "notes": "Strong non-KD missing-modality MSA neighbor.",
        },
        {
            "paper": "Missing Modality meets Meta Sampling (M3S)",
            "venue_year": "AACL-IJCNLP 2022",
            "dataset": "IEMOCAP; SIMS; CMU-MOSI",
            "backbone_text_features": "model-agnostic MAML add-on; exact features UNKNOWN",
            "complete_teacher_type": "NONE",
            "student_count": "N/A",
            "response_kd": "NO",
            "feature_kd": "NO",
            "relation_kd": "NO",
            "prototype_kd": "NO",
            "self_distillation": "NO",
            "reliability_difficulty_weighting": "meta-sampling",
            "student_state_weighting": "meta-learning state",
            "sample_level_weighting": "sampling rather than KD weighting",
            "missing_pattern_level_weighting": "YES; missing-modality sampling",
            "train_label_constructs_transferability": "NO",
            "explicit_negative_transfer_test": "NO CONFIRMED",
            "oracle": "NO",
            "reported_metrics": "UNKNOWN from inspected abstract",
            "fair_direct_comparison": "NO; non-KD meta-learning/protocol",
            "source_url": "https://aclanthology.org/2022.aacl-main.10/",
            "notes": "Missing-pattern training neighbor.",
        },
        {
            "paper": "Mitigating Inconsistencies in Multimodal Sentiment Analysis under Uncertain Missing Modalities (EMMR)",
            "venue_year": "EMNLP 2022",
            "dataset": "CMU-MOSI; IEMOCAP",
            "backbone_text_features": "encoder-decoder ensemble; exact features UNKNOWN",
            "complete_teacher_type": "NONE",
            "student_count": "N/A",
            "response_kd": "NO",
            "feature_kd": "NO",
            "relation_kd": "semantic-consistency detection",
            "prototype_kd": "NO",
            "self_distillation": "NO",
            "reliability_difficulty_weighting": "key-missing-modality consistency check",
            "student_state_weighting": "NO",
            "sample_level_weighting": "sample decision routing, not KD weighting",
            "missing_pattern_level_weighting": "YES",
            "train_label_constructs_transferability": "NO CONFIRMED",
            "explicit_negative_transfer_test": "targets inconsistency; no KD negative-transfer oracle",
            "oracle": "NO",
            "reported_metrics": "UNKNOWN from inspected abstract",
            "fair_direct_comparison": "NO; reconstruction ensemble/protocol",
            "source_url": "https://aclanthology.org/2022.emnlp-main.189/",
            "notes": "Decision-consistency and missingness-routing neighbor.",
        },
        {
            "paper": "Multimodal Prompt Learning with Missing Modalities for Sentiment Analysis and Emotion Recognition",
            "venue_year": "ACL 2024",
            "dataset": "MSA and emotion recognition benchmarks; names UNKNOWN from inspected abstract",
            "backbone_text_features": "multimodal Transformer with prompts",
            "complete_teacher_type": "NONE",
            "student_count": "N/A",
            "response_kd": "NO",
            "feature_kd": "NO",
            "relation_kd": "NO CONFIRMED",
            "prototype_kd": "NO",
            "self_distillation": "NO",
            "reliability_difficulty_weighting": "missing-type prompts",
            "student_state_weighting": "NO",
            "sample_level_weighting": "NO CONFIRMED",
            "missing_pattern_level_weighting": "YES; missing-type prompts",
            "train_label_constructs_transferability": "NO",
            "explicit_negative_transfer_test": "NO CONFIRMED",
            "oracle": "NO",
            "reported_metrics": "all evaluation metrics; exact list UNKNOWN from abstract",
            "fair_direct_comparison": "NO; prompt model/protocol",
            "source_url": "https://aclanthology.org/2024.acl-long.94/",
            "notes": "Non-KD missing-modality MSA neighbor.",
        },
        {
            "paper": "MSD: Saliency-aware Knowledge Distillation for Multimodal Understanding",
            "venue_year": "Findings of EMNLP 2021",
            "dataset": "four vision-language multimodal datasets; not MSA",
            "backbone_text_features": "task-specific multimodal teachers/students",
            "complete_teacher_type": "multimodal teacher",
            "student_count": "one student with modality-specific auxiliaries",
            "response_kd": "YES; modality-specific teacher predictions",
            "feature_kd": "NO CONFIRMED",
            "relation_kd": "NO CONFIRMED",
            "prototype_kd": "NO",
            "self_distillation": "NO",
            "reliability_difficulty_weighting": "YES; modality saliency",
            "student_state_weighting": "NO CONFIRMED",
            "sample_level_weighting": "NO CONFIRMED",
            "missing_pattern_level_weighting": "modality-specific, not missing-pattern-specific",
            "train_label_constructs_transferability": "NO",
            "explicit_negative_transfer_test": "NO CONFIRMED",
            "oracle": "NO",
            "reported_metrics": "task-specific metrics",
            "fair_direct_comparison": "NO; different tasks/backbones/features",
            "source_url": "https://aclanthology.org/2021.findings-emnlp.302/",
            "notes": "Adjacent multimodal KD weighting prior art.",
        },
        {
            "paper": "C2KD: Bridging the Modality Gap for Cross-Modal Knowledge Distillation",
            "venue_year": "CVPR 2024",
            "dataset": "audio-visual, image-text, and RGB-depth tasks; not MSA",
            "backbone_text_features": "task-specific; UNKNOWN",
            "complete_teacher_type": "cross-modal teacher/proxy",
            "student_count": "proxy plus target student",
            "response_kd": "YES; soft-label selection",
            "feature_kd": "YES",
            "relation_kd": "cross-modal",
            "prototype_kd": "NO CONFIRMED",
            "self_distillation": "bidirectional/progressive distillation",
            "reliability_difficulty_weighting": "on-the-fly sample selection",
            "student_state_weighting": "YES; proxy/student knowledge",
            "sample_level_weighting": "YES; filters misaligned soft labels",
            "missing_pattern_level_weighting": "NO",
            "train_label_constructs_transferability": "NO CONFIRMED",
            "explicit_negative_transfer_test": "motivated by misaligned labels; explicit oracle UNKNOWN",
            "oracle": "NO CONFIRMED",
            "reported_metrics": "task-specific metrics",
            "fair_direct_comparison": "NO; adjacent cross-modal KD only",
            "source_url": "https://openaccess.thecvf.com/content/CVPR2024/html/Huo_C2KD_Bridging_the_Modality_Gap_for_Cross-Modal_Knowledge_Distillation_CVPR_2024_paper.html",
            "notes": "Adjacent selective cross-modal KD prior art.",
        },
    ]


def find_row(frame, seed, method):
    return frame.loc[
        frame.Seed.astype(int).eq(seed) & frame.Method.eq(method)
    ].iloc[0]


def main():
    for directory in (STAGE_F, STAGE_G, FINAL):
        directory.mkdir(parents=True, exist_ok=True)
    literature = pd.DataFrame(literature_rows(), columns=LITERATURE_COLUMNS)
    literature.to_csv(
        STAGE_G / "CFCompatKD_DISTILLATION_LITERATURE_MATRIX.csv", index=False
    )

    stage_a = pd.read_csv(STAGE_A / "stage18a_replay_runs.csv")
    gaps = pd.read_csv(STAGE_B / "stage18b_gap_decomposition.csv")
    metrics = pd.concat(
        [
            pd.read_csv(STAGE_C / "stage18c_seed1114_metrics.csv"),
            pd.read_csv(STAGE_D / "stage18d_seed1111_metrics.csv"),
        ],
        ignore_index=True,
    )
    transfer = pd.read_csv(STAGE_E / "stage18e_transfer_quadrants.csv")
    calibration = pd.read_csv(
        STAGE_E / "stage18e_compatibility_calibration.csv"
    )
    benefit = pd.read_csv(STAGE_E / "stage18e_teacher_benefit.csv")
    gate = pd.read_csv(STAGE_E / "stage18e_continuation_gate.csv")
    manifest = json.loads((STAGE_E / "stage18e_manifest.json").read_text())
    if manifest["continuation_gate_passed"]:
        raise RuntimeError("This aggregator is only for the observed failed gate.")

    def mean_delta(method, reference):
        return sum(
            float(find_row(metrics, seed, method).J_valid)
            - float(find_row(metrics, seed, reference).J_valid)
            for seed in (1114, 1111)
        ) / 2.0

    pooled_transfer = transfer.loc[
        transfer.Seed.astype(str).eq("POOLED") & transfer.Mode.eq("ALL")
    ].set_index("Method")
    pooled_cal = calibration.loc[
        calibration.Seed.astype(str).eq("POOLED")
        & calibration.Mode.eq("ALL")
    ].set_index("Target")
    pooled_benefit = benefit.loc[
        benefit.Seed.astype(str).eq("POOLED")
        & benefit.GroupType.eq("all")
    ].iloc[0]
    cf_equal = mean_delta("cfcompat", "equal_mass")
    cf_mode = mean_delta("cfcompat", "mode_mean")
    innovation_level = 2 if cf_equal < 0 and cf_mode >= 0 else 0
    if innovation_level != 2:
        raise RuntimeError("Observed innovation rule changed unexpectedly.")

    novelty = """# CFCompatKD Novelty Positioning

## Evidence-based position

The maximum supported position is **Level 2 (mode-level selectivity; provisional
two-seed ceiling)**. CFCompatKD improves over Equal-Mass Global KD on mean
validation J ({cf_equal:+.9f}), but does not improve over Mode-Mean KD
({cf_mode:+.9f}). It is also worse than Uniform KD on mean J
({cf_uniform:+.9f}).

This is not evidence for sample-level counterfactual transferability-aware
distillation. The compatibility score is below chance for TeacherBetter
(AUROC {tb_auc:.6f}) and DirectionCorrect (AUROC {dc_auc:.6f}); the
Shuffled-Teacher control is equivalent to CFCompatKD within the preregistered
0.001 tolerance. Although CFCompatKD reduces pooled harmful imitation relative
to Uniform by {harmful_delta:+.6f}, its pooled mean label-error change is
{cf_dy:+.6f}, versus {uniform_dy:+.6f} for Uniform.

## Literature boundary

Direct neighbors already cover complete-to-incomplete MSA distillation,
sample-level contrastive distillation, prototypes, modality-difficulty
weighting, and instance-aware reliability weighting. In particular, CorrKD
(CVPR 2024), CMAD (ICCV 2025), and EBMC (CVPR 2026) overlap important parts of
the broad claim. C2KD (CVPR 2024) is adjacent selective cross-modal KD.

Therefore, do not claim:

- "the first sample-level multimodal distillation";
- "the first reliability-weighted missing-modality KD";
- "counterfactual transferability is validated";
- "negative transfer is solved";
- "state of the art" or superiority to the cited methods without a matched
  reproduction.

Allowed positioning:

> A same-backbone diagnostic study of compatibility-routed response
> distillation under missing modalities, with evidence that mode-level KD
> strength and training instability explain more of the observed behavior than
> validated sample-level teacher transferability.

Different backbones, features and missing protocols; literature reference only.
""".format(
        cf_equal=cf_equal,
        cf_mode=cf_mode,
        cf_uniform=mean_delta("cfcompat", "uniform"),
        tb_auc=float(pooled_cal.loc["TeacherBetter", "AUROC"]),
        dc_auc=float(pooled_cal.loc["DirectionCorrect", "AUROC"]),
        harmful_delta=float(
            pooled_transfer.loc["cfcompat", "HarmfulImitationRate_Q2"]
            - pooled_transfer.loc["uniform", "HarmfulImitationRate_Q2"]
        ),
        cf_dy=float(pooled_transfer.loc["cfcompat", "MeanDeltaY"]),
        uniform_dy=float(pooled_transfer.loc["uniform", "MeanDeltaY"]),
    )
    (STAGE_G / "CFCompatKD_NOVELTY_POSITIONING.md").write_text(novelty)
    (STAGE_G / "CFCompatKD_INNOVATION_LEVEL.md").write_text(
        "# CFCompatKD Innovation Level\n\n"
        "**Level 2 — Mode-level selectivity (provisional two-seed ceiling).**\n\n"
        "CFCompatKD beats Equal-Mass Global KD in mean validation J but not "
        "Mode-Mean KD. It fails the Stage 18F continuation gate and therefore "
        "does not support Level 3–5 or a major-innovation claim.\n"
    )

    (STAGE_F / "stage18f_five_seed_audit.md").write_text(
        "# Stage 18F Five-Seed Audit\n\n"
        "Status: `NOT_EXECUTED_CORE_GATE_FAILED`\n\n"
        "Stage 18E passed 3/5 continuation criteria. The worst two-seed "
        "CFCompat-vs-control degradation was +0.002843 (> +0.001), and "
        "Shuffled-Teacher was equivalent within the preregistered tolerance. "
        "Five-seed training and all new Test access therefore remain prohibited.\n"
    )
    (STAGE_F / "stage18f_manifest.json").write_text(
        json.dumps(
            {
                "status": "NOT_EXECUTED_CORE_GATE_FAILED",
                "five_seed_training_executed": False,
                "test_unlock_generated": False,
                "locked_test_access_count": 0,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    if (ROOT / "TEST_UNLOCK_MANIFEST.json").exists():
        raise RuntimeError("Test unlock manifest must not exist after failed gate.")

    head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True
    ).strip()
    final = """# Stage 18 CFCompatKD Distillation Evidence Closure

Final status: `STAGE18_CORE_DISTILLATION_CLAIM_UNSUPPORTED`

1. Fair no-Test protocol recovered: **YES** (Stage 18A hard gate passed).
2. New CFCompat vs historical quality: J, LAV MAE, and MissingMacro MAE deltas
   are exactly 0 for seed 1114.
3. Duplicate stability: exact checkpoint/prediction replay within each method;
   J deltas are 0.
4. Specialists vs Unified: mixed by mode/seed; not uniformly better.
5. Optimization/Sharing Gap: pooled mean `{gopt:+.9f}`; strongly seed/mode
   dependent.
6. Information Gap: pooled mean `{ginfo:+.9f}`; mostly negative, so missing
   information is not the dominant explanation in this audit.
7. Uniform KD vs ModDrop: **improved**, mean Delta J `{uniform_mod:+.9f}`.
8. CFCompat vs Uniform: **not improved**, mean Delta J `{cf_uniform:+.9f}`.
9. CFCompat vs Equal-Mass: **improved on mean**, Delta J `{cf_equal:+.9f}`.
10. CFCompat vs Mode-Mean: **not improved**, Delta J `{cf_mode:+.9f}`.
11. CFCompat vs Shuffled-Gate: mean Delta J `{cf_shuffle:+.9f}`, but worst seed
    is +0.002843 and violates the +0.001 gate.
12. CFCompat vs Shuffled-Teacher: **not distinguishable**, mean Delta J
    `{cf_st:+.9f}` within ±0.001.
13. Oracle vs Uniform: **not improved**, mean Delta J `{oracle_uniform:+.9f}`.
14. Oracle vs CFCompat: Oracle is better by J `{oracle_cf:+.9f}` on mean.
15. TeacherBetter rate: `{teacher_better:.6f}`.
16. DirectionCorrect rate: `{direction_correct:.6f}`.
17. Compatibility AUROC/AUPRC: TeacherBetter `{tb_auc:.6f}/{tb_auprc:.6f}`;
    DirectionCorrect `{dc_auc:.6f}/{dc_auprc:.6f}`; OracleGate
    `{og_auc:.6f}/{og_auprc:.6f}`.
18. Uniform harmful imitation rate: `{uniform_harm:.6f}`.
19. CFCompat harmful imitation rate: `{cf_harm:.6f}`.
20. Helpful transfer rate: Uniform `{uniform_help:.6f}`; CFCompat
    `{cf_help:.6f}`.
21. Improvement from true Teacher knowledge: **not supported**; shuffled
    Teacher is equivalent and compatibility discrimination is below chance.
22. Five-seed Stage 18F executed: **NO; prohibited by failed core gate**.
23. Five-seed improvement count: **N/A**.
24. Worst observed mechanism seed: **1111** (CFCompat vs Shuffled-Gate
    Delta J +0.002843).
25. Remove-best-seed five-seed result: **N/A; Stage 18F not executed**.
26. Valid statistical evidence: two-seed descriptive evidence only; no
    five-seed permutation/bootstrap claim.
27. Test accessed: **NO**.
28. Locked Test access count: **0**.
29. MOSEI affected: **NO**; terminal completed state was read-only monitored.
30. Innovation level: **Level {innovation_level} (provisional ceiling)**.
31. Recommended positioning: same-backbone diagnostic of mode-level KD
    strength/instability, not a validated major method innovation.
32. Forbidden novelty wording: no "first", "SOTA", "validated
    counterfactual transferability", or "all-metric superiority" claims.
33. Branch: `experiment/cfcompat-distillation-evidence-v1`.
34. Commit at report generation: `{head}` (final mechanism commit may be the
    immediate descendant when this ignored result report is regenerated).
35. Final report: `{final_path}`.

Continuation gate:

```
{gate}
```

Final decision:

`CFCompatKD MECHANISM CHARACTERIZED; MAJOR-INNOVATION CLAIM NOT SUPPORTED`
""".format(
        gopt=float(gaps.G_opt.mean()),
        ginfo=float(gaps.G_info.mean()),
        uniform_mod=mean_delta("uniform", "moddrop"),
        cf_uniform=mean_delta("cfcompat", "uniform"),
        cf_equal=cf_equal,
        cf_mode=cf_mode,
        cf_shuffle=mean_delta("cfcompat", "shuffled_gate"),
        cf_st=mean_delta("cfcompat", "shuffled_teacher"),
        oracle_uniform=mean_delta("oracle", "uniform"),
        oracle_cf=mean_delta("oracle", "cfcompat"),
        teacher_better=float(pooled_benefit.TeacherBetterRate),
        direction_correct=float(pooled_benefit.DirectionCorrectRate),
        tb_auc=float(pooled_cal.loc["TeacherBetter", "AUROC"]),
        tb_auprc=float(pooled_cal.loc["TeacherBetter", "AUPRC"]),
        dc_auc=float(pooled_cal.loc["DirectionCorrect", "AUROC"]),
        dc_auprc=float(pooled_cal.loc["DirectionCorrect", "AUPRC"]),
        og_auc=float(pooled_cal.loc["OracleGate", "AUROC"]),
        og_auprc=float(pooled_cal.loc["OracleGate", "AUPRC"]),
        uniform_harm=float(
            pooled_transfer.loc["uniform", "HarmfulImitationRate_Q2"]
        ),
        cf_harm=float(
            pooled_transfer.loc["cfcompat", "HarmfulImitationRate_Q2"]
        ),
        uniform_help=float(
            pooled_transfer.loc["uniform", "HelpfulTransferRate_Q1"]
        ),
        cf_help=float(
            pooled_transfer.loc["cfcompat", "HelpfulTransferRate_Q1"]
        ),
        innovation_level=innovation_level,
        head=head,
        final_path=str(FINAL / "stage18_cfcompat_evidence_final_audit.md"),
        gate=gate.to_string(index=False),
    )
    (FINAL / "stage18_cfcompat_evidence_final_audit.md").write_text(final)
    print(final)


if __name__ == "__main__":
    main()
