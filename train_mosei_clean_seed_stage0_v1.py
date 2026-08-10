"""Windows-safe Stage-0 clean DLF training for MOSEI.

This entry point exists because Windows DataLoader multiprocessing cannot respawn
an inline <stdin> program.  It is a real .py module with a guarded main entry.

Protocol:
- train/valid only; MMDataLoader(mode=train) never constructs Test;
- one seed per process;
- canonical validation-best checkpoint path expected by downstream CFCompat;
- no hyperparameter search;
- optional TF32/high matmul precision is recorded explicitly.
"""
from __future__ import annotations

import argparse
import hashlib
import inspect
import json
from pathlib import Path

import torch
import torch.optim.lr_scheduler as lr_scheduler


def _install_reduce_on_plateau_verbose_compat() -> bool:
    """Ignore legacy verbose= on new PyTorch without changing scheduler logic."""
    original = lr_scheduler.ReduceLROnPlateau
    if "verbose" in inspect.signature(original).parameters:
        return False

    class CompatReduceLROnPlateau(original):
        def __init__(self, *args, verbose=None, **kwargs):
            super().__init__(*args, **kwargs)

    lr_scheduler.ReduceLROnPlateau = CompatReduceLROnPlateau
    return True


# Must happen before importing run -> trains.singleTask.DLF, which imports the
# scheduler symbol into its module namespace.
SCHEDULER_COMPAT_INSTALLED = _install_reduce_on_plateau_verbose_compat()

from run import DLF_run  # noqa: E402


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_args():
    p = argparse.ArgumentParser(description="Train one clean MOSEI DLF seed on Windows")
    p.add_argument("--seed", type=int, choices=(1111, 1112, 1113, 1114, 1115), default=1111)
    p.add_argument("--gpu-id", type=int, default=0)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--update-epochs", type=int, default=10)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--matmul-precision", choices=("highest", "high", "medium"), default="high")
    p.add_argument("--model-save-dir", default="./pt")
    p.add_argument("--result-root", default="./result")
    p.add_argument("--log-dir", default="./log")
    p.add_argument("--config-file", default="config/config.json")
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()
    if args.batch_size < 1 or args.update_epochs < 1 or args.num_workers < 0:
        p.error("batch-size/update-epochs must be positive and num-workers non-negative")
    return args


def main():
    args = parse_args()
    checkpoint = Path(args.model_save_dir) / f"DLF_mosei_seed{args.seed}_best.pth"
    if checkpoint.exists() and not args.overwrite:
        raise FileExistsError(
            f"Validation-best checkpoint already exists: {checkpoint}. "
            "Inspect it or pass --overwrite intentionally."
        )

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this MOSEI training run.")

    torch.cuda.set_device(args.gpu_id)
    torch.set_float32_matmul_precision(args.matmul_precision)
    # Keep conv behavior explicit as well.  'high'/'medium' are speed-oriented;
    # 'highest' keeps matmul at full FP32 internal precision.
    torch.backends.cudnn.allow_tf32 = args.matmul_precision != "highest"

    free_bytes, total_bytes = torch.cuda.mem_get_info(args.gpu_id)
    props = torch.cuda.get_device_properties(args.gpu_id)
    print("================ MOSEI clean Stage-0 configuration ================")
    print("torch:", torch.__version__)
    print("cuda runtime:", torch.version.cuda)
    print("gpu:", props.name)
    print("gpu total MiB:", total_bytes / 1024**2)
    print("gpu free  MiB:", free_bytes / 1024**2)
    print("seed:", args.seed)
    print("batch_size:", args.batch_size)
    print("update_epochs:", args.update_epochs)
    print("nominal examples per optimizer step:", args.batch_size * args.update_epochs)
    print("num_workers:", args.num_workers)
    print("matmul_precision:", args.matmul_precision)
    print("legacy scheduler compat installed:", SCHEDULER_COMPAT_INSTALLED)
    print("TEST CONSTRUCTED BY TRAIN RUN: False")

    DLF_run(
        model_name="DLF",
        dataset_name="mosei",
        config_file=args.config_file,
        config={
            "batch_size": int(args.batch_size),
            "update_epochs": int(args.update_epochs),
        },
        seeds=[int(args.seed)],
        is_tune=False,
        model_save_dir=args.model_save_dir,
        res_save_dir=args.result_root,
        log_dir=args.log_dir,
        gpu_ids=[int(args.gpu_id)],
        num_workers=int(args.num_workers),
        mode="train",
        is_training=True,
    )

    if not checkpoint.is_file():
        raise RuntimeError(f"Training returned without validation-best checkpoint: {checkpoint}")

    out = Path(args.result_root) / "missing_baseline" / "mosei_clean_stage0_v1" / f"seed{args.seed}"
    out.mkdir(parents=True, exist_ok=True)
    manifest = {
        "dataset": "mosei",
        "seed": int(args.seed),
        "selected_by": "validation_Loss",
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256(checkpoint),
        "checkpoint_size_bytes": checkpoint.stat().st_size,
        "batch_size": int(args.batch_size),
        "update_epochs": int(args.update_epochs),
        "nominal_examples_per_optimizer_step": int(args.batch_size * args.update_epochs),
        "num_workers": int(args.num_workers),
        "matmul_precision": args.matmul_precision,
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "gpu_name": props.name,
        "scheduler_verbose_compat_installed": bool(SCHEDULER_COMPAT_INSTALLED),
        "test_constructed": False,
    }
    (out / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print("================ MOSEI clean Stage-0 complete ======================")
    print("checkpoint:", checkpoint)
    print("sha256:", manifest["checkpoint_sha256"])
    print("manifest:", out / "run_manifest.json")


if __name__ == "__main__":
    main()
