"""Student-only benchmark evaluator for DLF-ReliabilityKD-v1."""
import argparse
from pathlib import Path
import torch
import torch.nn as nn
from config import get_config_regression
from trains.singleTask.model.DLF import DLF
from trains.singleTask.missing_utils import MissingModalityWrapper,build_single_split_loader,evaluate_all_modes,flatten_mode_metrics,clean_checkpoint_path,write_result_csvs
from trains.singleTask.reliability_kd_utils import reliability_checkpoint_path
from utils.functions import assign_gpu,setup_seed

def main():
 p=argparse.ArgumentParser();p.add_argument('--dataset',choices=('mosi',),default='mosi');p.add_argument('--seeds',nargs='+',type=int,default=[1111]);p.add_argument('--split',choices=('valid','test'),default='valid');p.add_argument('--num-workers',type=int,default=0);p.add_argument('--gpu-ids',nargs='*',type=int,default=[0]);p.add_argument('--model-save-dir',default='pt');p.add_argument('--result-dir',default='result/missing_baseline/reliability_kd_v1');p.add_argument('--config-file',default='config/config.json');c=p.parse_args();rows=[]
 for seed in c.seeds:
  setup_seed(seed);a=get_config_regression('DLF',c.dataset,c.config_file);a.feature_T=a.feature_A=a.feature_V='';a.mode=c.split;a.is_training=False;a.seed=seed;a.device=assign_gpu(list(c.gpu_ids));clean=clean_checkpoint_path(c.model_save_dir,a.dataset_name,seed);path=reliability_checkpoint_path(c.model_save_dir,a.dataset_name,seed);b=DLF(a).to(a.device);b.load_state_dict(torch.load(clean,map_location=a.device),strict=True);m=MissingModalityWrapper(b,a.feature_dims[1],a.feature_dims[2]).to(a.device);m.load_state_dict(torch.load(path,map_location=a.device),strict=True);metrics=evaluate_all_modes(m,build_single_split_loader(a,c.split,c.num_workers),a.device,'moddrop',nn.L1Loss());r={'Seed':seed,'Checkpoint':str(path)};r.update(flatten_mode_metrics(metrics));rows.append(r);print('seed={} split={} checkpoint={}'.format(seed,c.split,path))
 write_result_csvs(rows,Path(c.result_dir)/'eval'/c.split,c.dataset)
if __name__=='__main__':main()
