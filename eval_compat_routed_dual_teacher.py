"""Independent Student-only Stage 5A checkpoint evaluation."""
import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn

from train_compat_routed_dual_teacher import build_config, method_paths
from trains.singleTask.compat_routed_dual_teacher_utils import TEACHER_ROUTES
from trains.singleTask.missing_utils import MissingModalityWrapper, build_single_split_loader, evaluate_all_modes, validation_objective
from trains.singleTask.model.DLF import DLF


def parse_args():
    parser=argparse.ArgumentParser(description="Evaluate a Stage 5A Student checkpoint without either Teacher or a train cache.")
    parser.add_argument("--dataset",choices=("mosi",),default="mosi"); parser.add_argument("--seed",type=int,default=1111)
    parser.add_argument("--teacher-route",choices=tuple(TEACHER_ROUTES),default="compatibility_routed")
    parser.add_argument("--checkpoint-kind",choices=("valid","diagnostic"),default="valid")
    parser.add_argument("--smoke-test",action="store_true")
    parser.add_argument("--gpu-ids",nargs="*",type=int,default=[0]); parser.add_argument("--num-workers",type=int,default=0)
    parser.add_argument("--model-save-dir",default="pt"); parser.add_argument("--result-root",default="result")
    parser.add_argument("--config-file",default="config/config.json"); parser.add_argument("--output")
    return parser.parse_args()


def main():
    cli=parse_args(); args=build_config(cli,cli.seed)
    _,main_template,diagnostic_template=method_paths(cli,cli.dataset)
    checkpoint=Path(str(main_template if cli.checkpoint_kind=="valid" else diagnostic_template).format(cli.seed))
    if not checkpoint.is_file(): raise FileNotFoundError(checkpoint)
    model=MissingModalityWrapper(DLF(args).to(args.device),args.feature_dims[1],args.feature_dims[2]).to(args.device)
    state=torch.load(checkpoint,map_location=args.device)
    if any("teacher" in key.lower() or "evaluator" in key.lower() or "residual" in key.lower() for key in state):
        raise ValueError("Student checkpoint contains forbidden Teacher/Evaluator/residual state.")
    model.load_state_dict(state,strict=True); criterion=nn.L1Loss()
    valid=evaluate_all_modes(model,build_single_split_loader(args,"valid",cli.num_workers),args.device,"moddrop",criterion)
    test=evaluate_all_modes(model,build_single_split_loader(args,"test",cli.num_workers),args.device,"moddrop",criterion)
    payload={"checkpoint":str(checkpoint),"checkpoint_kind":cli.checkpoint_kind,"student_only":True,
             "J_valid":validation_objective(valid),"J_test":validation_objective(test),"valid":valid,"test":test}
    text=json.dumps(payload,indent=2,sort_keys=True)
    if cli.output: Path(cli.output).write_text(text+"\n")
    print(text)


if __name__=="__main__": main()
