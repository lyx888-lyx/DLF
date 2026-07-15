"""Stage 3A benchmark-track reliability-gated prediction KD."""
import argparse, logging, math
from collections import Counter
from datetime import datetime
from pathlib import Path
import pandas as pd
import torch
import torch.nn as nn
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from config import get_config_regression
from data_loader import MMDataLoader
from trains.singleTask.HingeLoss import HingeLoss
from trains.singleTask.model.DLF import DLF
from trains.singleTask.missing_utils import MissingModalityWrapper,clean_checkpoint_path,compute_full_dlf_loss,compute_task_loss,count_missing_modes,evaluate_all_modes,flatten_mode_metrics,mode_to_mask,sample_missing_masks,validation_objective,write_result_csvs,build_single_split_loader
from trains.singleTask.fixed_kd_utils import build_frozen_teacher,checkpoint_sha256,assert_initial_lav_equivalence,assert_teacher_not_in_optimizer,teacher_grad_count,teacher_lav_prediction,compute_validation_gaps
from trains.singleTask.reliability_kd_utils import reliability_weights,reliability_kd_loss,reliability_checkpoint_path,stats
from utils.functions import assign_gpu,setup_seed

def parse_args():
 p=argparse.ArgumentParser();p.add_argument('--dataset',choices=('mosi',),default='mosi');p.add_argument('--seeds',nargs='+',type=int,default=[1111]);p.add_argument('--eta',type=float,default=1.);p.add_argument('--lambda-kd',type=float,default=1.);p.add_argument('--smoke-test',action='store_true');p.add_argument('--max-epochs',type=int);p.add_argument('--num-workers',type=int,default=1);p.add_argument('--gpu-ids',nargs='*',type=int,default=[0]);p.add_argument('--model-save-dir',default='pt');p.add_argument('--result-dir',default='result/missing_baseline/reliability_kd_v1/benchmark_train');p.add_argument('--log-dir',default='log/missing_baseline');p.add_argument('--config-file',default='config/config.json');a=p.parse_args()
 if a.eta!=1 or a.lambda_kd!=1:p.error('Stage 3A fixes eta and lambda-kd at 1.0.')
 if a.smoke_test:a.max_epochs=2 if a.max_epochs is None else min(2,a.max_epochs)
 return a
def config(c,seed):
 a=get_config_regression('DLF',c.dataset,c.config_file);a.mode='train';a.feature_T=a.feature_A=a.feature_V='';a.is_training=True;a.train_mode='regression';a.seed=a.cur_seed=seed;a.device=assign_gpu(list(c.gpu_ids));return a
def batch(b,d):return b['text'].to(d),b['audio'].to(d),b['vision'].to(d),b['labels']['M'].to(d).view(-1,1)
def ckpt(c,name,seed):
 p=reliability_checkpoint_path(c.model_save_dir,name,seed);p=(p.parent/'smoke'/p.name) if c.smoke_test else p;p.parent.mkdir(parents=True,exist_ok=True);return p
def logger(c):
 p=Path(c.log_dir);p.mkdir(parents=True,exist_ok=True);f=p/f'DLF-{c.dataset}-reliabilitykd-{("smoke" if c.smoke_test else "train")}-{datetime.now():%Y%m%d-%H%M%S}.log';l=logging.getLogger('reliabilitykd');l.handlers.clear();l.setLevel(logging.INFO);h=logging.FileHandler(f);s=logging.StreamHandler();[x.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')) for x in(h,s)];l.addHandler(h);l.addHandler(s);return l,f
def train(c,seed,l):
 setup_seed(seed);a=config(c,seed);loaders=MMDataLoader(a,c.num_workers);test_loader=build_single_split_loader(a,'test',c.num_workers)
 if set(loaders)!={'train','valid'}:raise RuntimeError('train/valid loader protocol failed')
 clean=clean_checkpoint_path(c.model_save_dir,a.dataset_name,seed);sha=checkpoint_sha256(clean);teacher=build_frozen_teacher(DLF,a,clean);backbone=DLF(a).to(a.device);backbone.load_state_dict(torch.load(clean,map_location=a.device),strict=True);student=MissingModalityWrapper(backbone,a.feature_dims[1],a.feature_dims[2]).to(a.device);student.eval();x,au,v,_=batch(next(iter(loaders['valid'])),a.device);assert_initial_lav_equivalence(teacher,student,x,au,v)
 opt=optim.Adam(student.parameters(),lr=a.learning_rate);assert_teacher_not_in_optimizer(teacher,opt);sched=ReduceLROnPlateau(opt,mode='min',factor=.5,patience=a.patience);crit=nn.L1Loss();cos=nn.CosineEmbeddingLoss();hinge=HingeLoss();gen=torch.Generator().manual_seed(seed+104729);path=ckpt(c,a.dataset_name,seed);best=float('inf');best_epoch=0;bestvalid=besttest=None;rows=[]
 l.info('benchmark/original-protocol track: test is evaluated every epoch; checkpoint selection uses valid J only')
 for epoch in range(1,(c.max_epochs or 1000)+1):
  student.train();opt.zero_grad();counts=Counter({'LA':0,'LV':0,'L':0});es=[];ws=[];kds=[];gaps=[]
  for step,b in enumerate(loaders['train'],1):
   text,audio,vision,y=batch(b,a.device);full,_=compute_full_dlf_loss(student(text,audio,vision,mode_to_mask('LAV',y.size(0),a.device,audio.dtype)),y,crit,cos,hinge);mask=sample_missing_masks(y.size(0),gen,a.device,audio.dtype);counts.update(count_missing_modes(mask));missing=student(text,audio,vision,mask);ml,_=compute_task_loss(missing,y,crit)
   tp=teacher_lav_prediction(teacher,text,audio,vision)
   w=reliability_weights(tp,y);kd,each=reliability_kd_loss(missing['output_logit'],tp,w);loss=full+ml+kd
   if step==1 and float(torch.autograd.grad(kd,[p for p in student.parameters() if p.requires_grad],retain_graph=True,allow_unused=True)[0].norm())<=0:raise RuntimeError('KD gradient missing')
   loss.backward();
   if teacher_grad_count(teacher):raise RuntimeError('Teacher gradient detected')
   if step%a.update_epochs==0 or step==len(loaders['train']):opt.step();opt.zero_grad()
   es.extend(torch.abs(tp-y).view(-1).cpu().tolist());ws.extend(w.cpu().tolist());kds.extend(each.detach().cpu().tolist());gaps.extend(torch.abs(missing['output_logit'].detach()-tp).view(-1).cpu().tolist())
  valid=evaluate_all_modes(student,loaders['valid'],a.device,'moddrop',crit);test=evaluate_all_modes(student,test_loader,a.device,'moddrop',crit);jv=validation_objective(valid);jt=validation_objective(test);sched.step(jv);isbest=jv<=best-1e-6
  if isbest:best,best_epoch,bestvalid,besttest=jv,epoch,valid,test;torch.save(student.state_dict(),path)
  st=stats(es);rw=stats(ws);ess=sum(ws)**2/sum(x*x for x in ws);row={'Seed':seed,'Epoch':epoch,'J_valid':jv,'J_test':jt,'IsBestValid':isbest,'KD_loss':sum(kds)/len(kds),'train_teacher_student_abs_gap':sum(gaps)/len(gaps),'ESS':ess,'ESS_fraction':ess/len(ws),**{'reliability_'+k:v for k,v in rw.items()},**{'teacher_error_'+k:v for k,v in st.items()}};row.update({'%s_valid_%s'%(m,k):v for m,z in valid.items() for k,v in z.items()});row.update({'%s_test_%s'%(m,k):v for m,z in test.items() for k,v in z.items()});rows.append(row);l.info('epoch=%s LA=%s LV=%s L=%s J_valid=%.6f J_test=%.6f KD=%.6f rel_mean=%.6f ESS_fraction=%.6f',epoch,counts['LA'],counts['LV'],counts['L'],jv,jt,row['KD_loss'],rw['mean'],row['ESS_fraction'])
  if epoch-best_epoch>=a.early_stop:break
 student.load_state_dict(torch.load(path,map_location=a.device));fv=evaluate_all_modes(student,loaders['valid'],a.device,'moddrop',crit);ft=evaluate_all_modes(student,test_loader,a.device,'moddrop',crit);besttestrow=min(rows,key=lambda r:r['J_test']);result={'Seed':seed,'BestEpoch':best_epoch,'J_valid':validation_objective(fv),'J_test_at_valid_best':validation_objective(ft),'BestObservedTestEpoch':besttestrow['Epoch'],'BestObservedTestJ':besttestrow['J_test'],'Checkpoint':str(path),'TeacherCheckpoint':str(clean),'TeacherCheckpointSHA256':sha,'StudentInitCheckpoint':str(clean),'StudentInitCheckpointSHA256':sha};result.update({'%s_valid_%s'%(m,k):v for m,z in fv.items() for k,v in z.items()});result.update({'%s_test_%s'%(m,k):v for m,z in ft.items() for k,v in z.items()});return result,rows
def main():
 c=parse_args();l,p=logger(c);allrows=[];epochs=[]
 for s in c.seeds:r,e=train(c,s,l);allrows.append(r);epochs+=e
 out=Path(c.result_dir)/('smoke' if c.smoke_test else '');write_result_csvs(allrows,out,c.dataset);pd.DataFrame(epochs).to_csv(out/f'{c.dataset}_epoch_metrics.csv',index=False);pd.DataFrame(epochs).to_csv(out/f'{c.dataset}_reliability_summary.csv',index=False);l.info('results=%s',out)
if __name__=='__main__':main()
