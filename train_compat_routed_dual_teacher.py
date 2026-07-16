"""Stage 5A compatibility-routed dual-teacher distillation benchmark."""
import argparse
import hashlib
import json
import logging
import math
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

from config import get_config_regression
from data_loader import MMDataLoader
from trains.singleTask.HingeLoss import HingeLoss
from trains.singleTask.cf_compat_kd_utils import locate_stage1_evaluator, modes_from_masks
from trains.singleTask.compat_routed_dual_teacher_utils import (
    TEACHER_ROUTES, build_dual_teacher_suitability_audit, cache_table,
    compatibility_targets, distribution_stats, dual_teacher_audit_paths,
    locate_stage3_cache, mode_teacher_targets, route_alpha,
    routed_dual_teacher_loss,
)
from trains.singleTask.fixed_kd_utils import (
    assert_initial_lav_equivalence, assert_teacher_not_in_optimizer,
    build_frozen_teacher, checkpoint_sha256, teacher_grad_count,
    teacher_lav_prediction,
)
from trains.singleTask.missing_utils import (
    MISSING_MODES, MissingModalityWrapper, build_single_split_loader,
    clean_checkpoint_path, compute_full_dlf_loss, compute_task_loss,
    count_missing_modes, evaluate_all_modes, flatten_mode_metrics,
    mode_to_mask, sample_missing_masks, validation_objective,
    write_result_csvs,
)
from trains.singleTask.model.DLF import DLF
from utils.functions import assign_gpu, setup_seed


EXPECTED_GATE3_SHA = "f2597c1c81529ac03710bd631ce8da2f121c3c2be7f5e06b15c399cd9d771ad2"


def parse_args():
    parser = argparse.ArgumentParser(description="Stage 5A compatibility-routed dual-teacher KD.")
    parser.add_argument("--dataset", choices=("mosi",), default="mosi")
    parser.add_argument("--seeds", nargs="+", type=int, default=[1111])
    parser.add_argument("--teacher-route", choices=tuple(TEACHER_ROUTES), default="compatibility_routed")
    parser.add_argument("--lambda-route", type=float, default=1.0)
    parser.add_argument("--build-dual-teacher-audit-only", action="store_true")
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--max-epochs", type=int)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--gpu-ids", nargs="*", type=int, default=[0])
    parser.add_argument("--model-save-dir", default="pt")
    parser.add_argument("--result-root", default="result")
    parser.add_argument("--log-dir", default="log/missing_baseline")
    parser.add_argument("--config-file", default="config/config.json")
    args = parser.parse_args()
    if args.lambda_route != 1.0:
        parser.error("Stage 5A fixes --lambda-route at 1.0.")
    if args.max_epochs is not None and args.max_epochs < 1:
        parser.error("--max-epochs must be positive.")
    if args.smoke_test:
        args.max_epochs = 2 if args.max_epochs is None else min(2, args.max_epochs)
    if args.seeds != [1111]:
        parser.error("Stage 5A pre-registration permits seed1111 only.")
    return args


def build_config(cli, seed):
    args = get_config_regression("DLF", cli.dataset, cli.config_file)
    args.mode = "train"; args.feature_T = args.feature_A = args.feature_V = ""
    args.is_training = True; args.train_mode = "regression"
    args.seed = args.cur_seed = int(seed); args.device = assign_gpu(list(cli.gpu_ids))
    return args


def batch_to_device(batch, device):
    return (batch["text"].to(device), batch["audio"].to(device), batch["vision"].to(device),
            batch["labels"]["M"].to(device).view(-1, 1))


def method_paths(cli, dataset):
    _, version, _ = TEACHER_ROUTES[cli.teacher_route]
    result = Path(cli.result_root) / "missing_baseline" / version / "benchmark_train"
    main = Path(cli.model_save_dir) / "missing_baseline" / version / "DLF_{}_seed{{}}_best_valid.pth".format(dataset)
    if cli.smoke_test:
        result = result / "smoke"; main = main.parent / "smoke" / main.name
    diagnostic = main.parent / "diagnostic" / main.name.replace("_best_valid.pth", "_best_test_diagnostic.pth")
    return result, main, diagnostic


def create_logger(cli):
    _, _, tag = TEACHER_ROUTES[cli.teacher_route]
    directory = Path(cli.log_dir); directory.mkdir(parents=True, exist_ok=True)
    kind = "smoke" if cli.smoke_test else "train"
    path = directory / "DLF-{}-{}-seed1111-{}-{}.log".format(cli.dataset, tag, kind, datetime.now().strftime("%Y%m%d-%H%M%S"))
    logger = logging.getLogger("compat_routed_dual_teacher"); logger.handlers.clear(); logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    for handler in (logging.FileHandler(path), logging.StreamHandler()):
        handler.setFormatter(formatter); logger.addHandler(handler)
    return logger, path


def initialize_models(args, cli, seed, validation_loader=None):
    checkpoint = clean_checkpoint_path(cli.model_save_dir, args.dataset_name, seed)
    if not checkpoint.is_file():
        raise FileNotFoundError("Gate 3 checkpoint missing: {}".format(checkpoint))
    sha = checkpoint_sha256(checkpoint)
    if int(seed) == 1111 and sha != EXPECTED_GATE3_SHA:
        raise ValueError("Locked Gate 3 SHA mismatch.")
    teacher = build_frozen_teacher(DLF, args, checkpoint)
    backbone = DLF(args).to(args.device); backbone.load_state_dict(torch.load(checkpoint, map_location=args.device), strict=True)
    student = MissingModalityWrapper(backbone, args.feature_dims[1], args.feature_dims[2]).to(args.device)
    if validation_loader is not None:
        student.eval(); batch = next(iter(validation_loader)); text, audio, vision, _ = batch_to_device(batch, args.device)
        assert_initial_lav_equivalence(teacher, student, text, audio, vision)
    return teacher, student, checkpoint, sha


def prediction_rows(model, loader, device):
    rows = []; model.eval()
    with torch.no_grad():
        for batch in loader:
            text, audio, vision, labels = batch_to_device(batch, device); predictions = {}
            for mode in ("LAV",) + MISSING_MODES:
                mask = mode_to_mask(mode, labels.size(0), device, audio.dtype)
                predictions[mode] = model(text, audio, vision, mask)["output_logit"].view(-1).cpu().numpy()
            indices = batch["index"].view(-1).cpu().numpy().astype(int); ids = list(batch["id"])
            for pos, index in enumerate(indices):
                rows.append({"sample_index": int(index), "sample_id": str(ids[pos]), "label": float(labels[pos]),
                             **{"{}_pred".format(mode): float(values[pos]) for mode, values in predictions.items()}})
    return pd.DataFrame(rows).sort_values("sample_index", kind="mergesort")


def _flatten(metrics, prefix):
    values = {"{}_{}".format(prefix, key): value for key, value in flatten_mode_metrics(metrics).items()}
    keys = next(iter(metrics.values())).keys()
    for key in keys:
        values["{}_MissingMacro_{}".format(prefix, key)] = float(np.mean([metrics[m][key] for m in MISSING_MODES]))
    return values


def missing_sequence_sha(seed, sample_count=1284, epochs=1000):
    generator = torch.Generator().manual_seed(int(seed) + 104729)
    digest = hashlib.sha256()
    for _ in range(epochs):
        digest.update(torch.randint(0, len(MISSING_MODES), (sample_count,), generator=generator, dtype=torch.int8).numpy().tobytes())
    return digest.hexdigest()


def _epoch_diagnostics(records, seed, epoch, method, route):
    data = pd.DataFrame(records)
    if data.empty or not np.isfinite(data.select_dtypes(include=[np.number])).all().all():
        raise RuntimeError("Training diagnostics are empty or non-finite.")
    full_sum = float(data.weighted_full.sum()); mode_sum = float(data.weighted_mode.sum()); denominator = full_sum + mode_sum + 1e-8
    alpha_stats = distribution_stats(data.alpha)
    disagreement = distribution_stats(data.teacher_disagreement)
    summary = {"Seed": seed, "Epoch": epoch, "Method": method, "TeacherRoute": route, "SampleCount": len(data),
               "FullKDRawMean": float(data.d_full.mean()), "ModeKDRawMean": float(data.d_mode.mean()),
               "WeightedFullContribution": float(data.weighted_full.mean()), "WeightedModeContribution": float(data.weighted_mode.mean()),
               "RouteLoss": float(data.route_loss.mean()), "FullContributionFraction": full_sum/denominator,
               "ModeContributionFraction": 1-full_sum/denominator,
               **{"Alpha{}".format(k.capitalize()): v for k, v in alpha_stats.items()},
               "TeacherDisagreementMean": disagreement["mean"], "TeacherDisagreementStd": disagreement["std"],
               "TeacherDisagreementP50": disagreement["median"], "TeacherDisagreementP90": disagreement["p90"],
               "TeacherDisagreementP95": disagreement["p95"], "TeacherDisagreementMax": disagreement["max"],
               "StudentFullTeacherGap": float(data.student_full_gap.mean()), "StudentModeTeacherGap": float(data.student_mode_gap.mean())}
    mode_rows = []
    for mode, local in data.groupby("mode"):
        mode_rows.append({"Seed": seed, "Epoch": epoch, "Method": method, "TeacherRoute": route, "Mode": mode, "count": len(local),
                          "mean_alpha": float(local.alpha.mean()), "full_KD": float(local.d_full.mean()), "mode_KD": float(local.d_mode.mean()),
                          "full_contribution": float(local.weighted_full.mean()), "mode_contribution": float(local.weighted_mode.mean()),
                          "student_label_MAE": float(local.student_label_error.mean()), "student_full_gap": float(local.student_full_gap.mean()),
                          "student_mode_gap": float(local.student_mode_gap.mean()), "teacher_disagreement": float(local.teacher_disagreement.mean())})
    ordered = data.sort_values(["compatibility", "sample_index"], kind="mergesort").copy()
    ordered["Quartile"] = pd.qcut(np.arange(len(ordered)), 4, labels=["Q1_low", "Q2", "Q3", "Q4_high"])
    quartile_rows = []
    for quartile, local in ordered.groupby("Quartile", observed=False):
        quartile_rows.append({"Seed": seed, "Epoch": epoch, "Method": method, "TeacherRoute": route, "Quartile": str(quartile),
                              "count": len(local), "mean_compatibility": float(local.compatibility.mean()),
                              "mean_full_teacher_label_error": float(local.full_teacher_error.mean()),
                              "mean_mode_teacher_label_error": float(local.mode_teacher_error.mean()),
                              "fraction_mode_teacher_closer_to_label": float((local.mode_teacher_error < local.full_teacher_error).mean()),
                              "mean_student_full_gap": float(local.student_full_gap.mean()), "mean_student_mode_gap": float(local.student_mode_gap.mean()),
                              "fraction_student_closer_to_mode_teacher": float((local.student_mode_gap < local.student_full_gap).mean()),
                              "mean_full_KD": float(local.d_full.mean()), "mean_mode_KD": float(local.d_mode.mean()),
                              "mean_weighted_full_contribution": float(local.weighted_full.mean()),
                              "mean_weighted_mode_contribution": float(local.weighted_mode.mean()),
                              "mean_student_label_error": float(local.student_label_error.mean()),
                              **{"{}_count".format(mode): int((local["mode"] == mode).sum()) for mode in MISSING_MODES}})
    return summary, mode_rows, quartile_rows


def build_audit_only(cli, seed, logger):
    setup_seed(seed); args = build_config(cli, seed)
    cache_csv, cache_config_path, config, frame = locate_stage3_cache(cli.result_root, cli.dataset)
    mode_checkpoint, _, mode_source = locate_stage1_evaluator(cli.result_root, cli.dataset, seed)
    if checkpoint_sha256(mode_checkpoint) != config.get("evaluator_sha256"):
        raise ValueError("Mode Teacher differs from the Stage 3 cache manifest.")
    train_loader = build_single_split_loader(args, "train", cli.num_workers)
    teacher, student, checkpoint, _ = initialize_models(args, cli, seed)
    paths, summary = build_dual_teacher_suitability_audit(student, teacher, train_loader, args.device, frame,
                                                          checkpoint, mode_checkpoint, config, cli.result_root, cli.dataset, seed)
    logger.info("train-only suitability audit complete samples=%s cache=%s config=%s mode_source=%s outputs=%s",
                summary["train_sample_count"], cache_csv, cache_config_path, mode_source, paths["directory"])
    return paths


def train_one_seed(cli, seed, logger):
    setup_seed(seed); args = build_config(cli, seed)
    cache_csv, cache_config_path, cache_config, cache_frame = locate_stage3_cache(cli.result_root, cli.dataset)
    table = cache_table(cache_frame)
    mode_checkpoint, mode_epoch, mode_source = locate_stage1_evaluator(cli.result_root, cli.dataset, seed)
    mode_sha = checkpoint_sha256(mode_checkpoint)
    if mode_sha != cache_config.get("evaluator_sha256"):
        raise ValueError("Mode Teacher checkpoint/cache SHA binding failed.")
    audit_paths = dual_teacher_audit_paths(cli.result_root, cli.dataset, seed)
    if not audit_paths["targets"].is_file() or not audit_paths["manifest"].is_file():
        raise FileNotFoundError("Run the train-only suitability audit before training.")
    loaders = MMDataLoader(args, cli.num_workers)
    if set(loaders) != {"train", "valid"}: raise RuntimeError("Benchmark loader must expose train and valid only.")
    test_loader = build_single_split_loader(args, "test", cli.num_workers)
    teacher, student, init_checkpoint, init_sha = initialize_models(args, cli, seed, loaders["valid"])
    optimizer = optim.Adam(student.parameters(), lr=args.learning_rate); assert_teacher_not_in_optimizer(teacher, optimizer)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=.5, patience=args.patience)
    criterion, cosine, hinge = nn.L1Loss(), nn.CosineEmbeddingLoss(), HingeLoss()
    missing_generator = torch.Generator().manual_seed(int(seed) + 104729)
    result_dir, main_template, diagnostic_template = method_paths(cli, cli.dataset)
    main_checkpoint = Path(str(main_template).format(seed)); diagnostic_checkpoint = Path(str(diagnostic_template).format(seed))
    main_checkpoint.parent.mkdir(parents=True, exist_ok=True); diagnostic_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    method, _, _ = TEACHER_ROUTES[cli.teacher_route]
    best_valid_j = best_test_j = float("inf"); best_valid_epoch = best_test_epoch = 0
    epoch_rows=[]; route_rows=[]; quartile_rows=[]; mode_rows=[]
    logger.info("method=%s route=%s seed=%s main checkpoint selected by J_valid only", method, cli.teacher_route, seed)
    logger.info("full_teacher=%s sha=%s mode_teacher=%s sha=%s epoch=%s source=%s cache=%s manifest=%s",
                init_checkpoint, init_sha, mode_checkpoint, mode_sha, mode_epoch, mode_source, cache_csv, cache_config_path)
    for epoch in range(1, (cli.max_epochs or 1000)+1):
        student.train(); optimizer.zero_grad(); counts=Counter({"LA":0,"LV":0,"L":0}); records=[]
        full_losses=[]; missing_losses=[]
        for step, batch in enumerate(loaders["train"], 1):
            text,audio,vision,labels=batch_to_device(batch,args.device)
            full_mask=mode_to_mask("LAV",labels.size(0),args.device,audio.dtype)
            full_loss,_=compute_full_dlf_loss(student(text,audio,vision,full_mask),labels,criterion,cosine,hinge)
            missing_mask=sample_missing_masks(labels.size(0),missing_generator,args.device,audio.dtype)
            modes=modes_from_masks(missing_mask); counts.update(count_missing_modes(missing_mask))
            missing_output=student(text,audio,vision,missing_mask); missing_loss,_=compute_task_loss(missing_output,labels,criterion)
            full_target=teacher_lav_prediction(teacher,text,audio,vision).detach().view(-1)
            indices=batch["index"].view(-1).cpu().numpy().astype(int).tolist()
            mode_target=mode_teacher_targets(table,indices,modes,args.device,labels.dtype)
            compatibility=compatibility_targets(table,indices,modes,args.device,labels.dtype)
            alpha=route_alpha(cli.teacher_route,compatibility)
            route_loss,d_full,d_mode,routed=routed_dual_teacher_loss(missing_output["output_logit"],full_target,mode_target,alpha)
            if step == 1:
                grads=torch.autograd.grad(route_loss,[p for p in student.parameters() if p.requires_grad],retain_graph=True,allow_unused=True)
                if math.sqrt(sum(float(g.detach().pow(2).sum()) for g in grads if g is not None)) <= 0:
                    raise RuntimeError("Route loss did not reach Student parameters.")
            total_loss=full_loss+missing_loss+route_loss
            if not torch.isfinite(total_loss): raise FloatingPointError("NaN/Inf in Stage 5A loss.")
            total_loss.backward()
            if teacher_grad_count(teacher): raise RuntimeError("Frozen Full Teacher received gradients.")
            if step % args.update_epochs == 0 or step == len(loaders["train"]): optimizer.step(); optimizer.zero_grad()
            full_losses.append(float(full_loss.detach())); missing_losses.append(float(missing_loss.detach()))
            student_pred=missing_output["output_logit"].detach().view(-1); label=labels.view(-1)
            for pos,index in enumerate(indices):
                records.append({"sample_index":index,"mode":modes[pos],"compatibility":float(compatibility[pos]),"alpha":float(alpha[pos]),
                                "d_full":float(d_full[pos].detach()),"d_mode":float(d_mode[pos].detach()),"route_loss":float(routed[pos].detach()),
                                "weighted_full":float((alpha[pos]*d_full[pos]).detach()),"weighted_mode":float(((1-alpha[pos])*d_mode[pos]).detach()),
                                "full_teacher_error":float(torch.abs(full_target[pos]-label[pos])),"mode_teacher_error":float(torch.abs(mode_target[pos]-label[pos])),
                                "student_label_error":float(torch.abs(student_pred[pos]-label[pos])),"student_full_gap":float(torch.abs(student_pred[pos]-full_target[pos])),
                                "student_mode_gap":float(torch.abs(student_pred[pos]-mode_target[pos])),"teacher_disagreement":float(torch.abs(full_target[pos]-mode_target[pos]))})
        if epoch==1 and counts != Counter({"LA":435,"LV":430,"L":419}): raise RuntimeError("Epoch1 counts must be LA=435 LV=430 L=419.")
        valid=evaluate_all_modes(student,loaders["valid"],args.device,"moddrop",criterion); test=evaluate_all_modes(student,test_loader,args.device,"moddrop",criterion)
        j_valid,j_test=validation_objective(valid),validation_objective(test)
        if not (math.isfinite(j_valid) and math.isfinite(j_test)): raise FloatingPointError("Non-finite benchmark metric.")
        scheduler.step(j_valid); is_best_valid=j_valid <= best_valid_j-1e-6; is_best_test=j_test <= best_test_j-1e-6
        if is_best_valid: best_valid_j,best_valid_epoch=j_valid,epoch; torch.save(student.state_dict(),main_checkpoint)
        if is_best_test: best_test_j,best_test_epoch=j_test,epoch; torch.save(student.state_dict(),diagnostic_checkpoint)
        summary,modes_out,quartiles_out=_epoch_diagnostics(records,seed,epoch,method,cli.teacher_route)
        route_rows.append(summary); mode_rows.extend(modes_out); quartile_rows.extend(quartiles_out)
        epoch_rows.append({"Seed":seed,"Epoch":epoch,"Method":method,"TeacherRoute":cli.teacher_route,"J_valid":j_valid,"J_test":j_test,
                           "IsBestValid":is_best_valid,"IsBestTestDiagnostic":is_best_test,"FullLoss":float(np.mean(full_losses)),
                           "MissingLoss":float(np.mean(missing_losses)),**{k:v for k,v in summary.items() if k not in ("Seed","Epoch","Method","TeacherRoute")},
                           **_flatten(valid,"valid"),**_flatten(test,"test")})
        logger.info("epoch=%s LA=%s LV=%s L=%s J_valid=%.6f J_test=%.6f route=%.6f alpha=%.6f",epoch,counts["LA"],counts["LV"],counts["L"],j_valid,j_test,summary["RouteLoss"],summary["AlphaMean"])
        if epoch-best_valid_epoch >= args.early_stop: break
    if not main_checkpoint.is_file() or not diagnostic_checkpoint.is_file(): raise RuntimeError("Main and diagnostic checkpoints must both exist.")
    total_epochs=len(epoch_rows)
    student.load_state_dict(torch.load(main_checkpoint,map_location=args.device),strict=True)
    final_valid=evaluate_all_modes(student,loaders["valid"],args.device,"moddrop",criterion); final_test=evaluate_all_modes(student,test_loader,args.device,"moddrop",criterion)
    valid_predictions=prediction_rows(student,loaders["valid"],args.device); valid_predictions["selected_by"]="valid"; valid_predictions["diagnostic_only"]=False
    student.load_state_dict(torch.load(diagnostic_checkpoint,map_location=args.device),strict=True)
    diagnostic_test=evaluate_all_modes(student,test_loader,args.device,"moddrop",criterion)
    diagnostic_predictions=prediction_rows(student,test_loader,args.device); diagnostic_predictions["selected_by"]="test"; diagnostic_predictions["diagnostic_only"]=True; diagnostic_predictions["not_main_result"]=True
    result={"Seed":seed,"Method":method,"TeacherRoute":cli.teacher_route,"BestValidEpoch":best_valid_epoch,
            "J_valid":validation_objective(final_valid),"J_test_at_valid_best":validation_objective(final_test),"BestObservedTestEpoch":best_test_epoch,
            "BestObservedTestJ":best_test_j,"SelectionRegret":validation_objective(final_test)-best_test_j,"TotalEpochs":total_epochs,
            "MainCheckpoint":str(main_checkpoint),"MainCheckpointSHA256":checkpoint_sha256(main_checkpoint),"DiagnosticCheckpoint":str(diagnostic_checkpoint),
            "DiagnosticCheckpointSHA256":checkpoint_sha256(diagnostic_checkpoint),"FullTeacherCheckpoint":str(init_checkpoint),"FullTeacherSHA256":init_sha,
            "ModeTeacherCheckpoint":str(mode_checkpoint),"ModeTeacherSHA256":mode_sha,"StudentInitCheckpoint":str(init_checkpoint),"StudentInitSHA256":init_sha,
            "CompatibilityCache":str(cache_csv),"CompatibilityCacheSHA256":checkpoint_sha256(cache_csv),
            "DualTeacherAuditCache":str(audit_paths["targets"]),"DualTeacherAuditCacheSHA256":checkpoint_sha256(audit_paths["targets"]),
            "MissingSequenceSHA256":missing_sequence_sha(seed),"EvaluatorForwardDuringTraining":False,"ValidTestUsedTeacher":False,
            "ValidTestUsedEvaluator":False,"ValidTestUsedTrainCache":False,"TeacherGradientDetected":False,"TestBasedMainSelection":False,
            **_flatten(final_valid,"valid"),**_flatten(final_test,"test_at_valid_best"),**_flatten(diagnostic_test,"test_diagnostic")}
    return result,epoch_rows,route_rows,quartile_rows,mode_rows,valid_predictions,diagnostic_predictions


def main():
    cli=parse_args(); logger,log_path=create_logger(cli)
    if cli.build_dual_teacher_audit_only:
        for seed in cli.seeds: build_audit_only(cli,seed,logger)
        logger.info("audit-only complete; no valid/test loader or training checkpoint created; log=%s",log_path); return
    rows=[]; epochs=[]; routes=[]; quartiles=[]; modes=[]; valid_predictions=[]; diagnostic_predictions=[]
    for seed in cli.seeds:
        result,e,r,q,m,v,d=train_one_seed(cli,seed,logger); rows.append(result); epochs.extend(e); routes.extend(r); quartiles.extend(q); modes.extend(m); valid_predictions.append(v); diagnostic_predictions.append(d)
    result_dir,_,_=method_paths(cli,cli.dataset); write_result_csvs(rows,result_dir,cli.dataset)
    pd.DataFrame(epochs).to_csv(result_dir/"{}_epoch_metrics.csv".format(cli.dataset),index=False)
    pd.DataFrame(routes).to_csv(result_dir/"{}_route_summary.csv".format(cli.dataset),index=False)
    pd.DataFrame(quartiles).to_csv(result_dir/"{}_route_quartiles.csv".format(cli.dataset),index=False)
    pd.DataFrame(modes).to_csv(result_dir/"{}_mode_teacher_metrics.csv".format(cli.dataset),index=False)
    pd.concat(valid_predictions,ignore_index=True).to_csv(result_dir/"{}_best_valid_predictions.csv".format(cli.dataset),index=False)
    pd.concat(diagnostic_predictions,ignore_index=True).to_csv(result_dir/"{}_best_test_diagnostic_predictions.csv".format(cli.dataset),index=False)
    logger.info("complete results=%s log=%s",result_dir,log_path)


if __name__ == "__main__": main()
