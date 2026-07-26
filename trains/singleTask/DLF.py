import logging
import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
from ..utils import MetricsTop, dict_to_str
from .HingeLoss import HingeLoss
from .expert_analysis import (
    analyze_expert_pool,
    extract_expert_logits,
    normalize_batch_ids,
)


logger = logging.getLogger('MMSA')


class MSE(nn.Module):
    def __init__(self):
        super(MSE, self).__init__()

    def forward(self, pred, real):
        diffs = torch.add(real, -pred)
        n = torch.numel(diffs.data)
        mse = torch.sum(diffs.pow(2)) / n
        return mse


class DLF():
    def __init__(self, args):
        self.args = args
        self.criterion = nn.L1Loss()
        self.cosine = nn.CosineEmbeddingLoss()
        self.metrics = MetricsTop(args.train_mode).getMetics(args.dataset_name)
        self.MSE = MSE()
        self.sim_loss = HingeLoss()

    def do_train(self, model, dataloader, return_epoch_results=False):

        # 0: DLF model
        params = model[0].parameters()

        optimizer = optim.Adam(params, lr=self.args.learning_rate)
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            verbose=True,
            patience=self.args.patience,
        )

        epochs, best_epoch = 0, 0
        if return_epoch_results:
            epoch_results = {
                'train': [],
                'valid': [],
                'test': [],
            }
        min_or_max = 'min' if self.args.KeyEval in ['Loss'] else 'max'
        best_valid = 1e8 if min_or_max == 'min' else 0

        net = []
        net_DLF = model[0]
        net.append(net_DLF)
        model = net

        while True:
            epochs += 1
            y_pred, y_true = [], []
            for mod in model:
                mod.train()

            train_loss = 0.0
            left_epochs = self.args.update_epochs
            with tqdm(dataloader['train']) as td:
                for batch_data in td:

                    if left_epochs == self.args.update_epochs:
                        optimizer.zero_grad()
                    left_epochs -= 1
                    vision = batch_data['vision'].to(self.args.device)
                    audio = batch_data['audio'].to(self.args.device)
                    text = batch_data['text'].to(self.args.device)
                    labels = batch_data['labels']['M'].to(self.args.device)
                    labels = labels.view(-1, 1)

                    output = model[0](text, audio, vision)

                    # task loss
                    loss_task_all = self.criterion(output['output_logit'], labels)

                    loss_task_l_hetero = self.criterion(
                        output['logits_l_hetero'], labels
                    )
                    loss_task_v_hetero = self.criterion(
                        output['logits_v_hetero'], labels
                    )
                    loss_task_a_hetero = self.criterion(
                        output['logits_a_hetero'], labels
                    )
                    loss_task_c = self.criterion(output['logits_c'], labels)

                    # total MSA loss L_msa
                    loss_task = 1 * (
                        1 * loss_task_all
                        + 1 * loss_task_c
                        + 3 * loss_task_l_hetero
                        + 1 * loss_task_v_hetero
                        + 1 * loss_task_a_hetero
                    )

                    # reconstruction loss L_r
                    loss_recon_l = self.MSE(output['recon_l'], output['origin_l'])
                    loss_recon_v = self.MSE(output['recon_v'], output['origin_v'])
                    loss_recon_a = self.MSE(output['recon_a'], output['origin_a'])
                    loss_recon = loss_recon_l + loss_recon_v + loss_recon_a

                    # specific loss L_s
                    loss_sl_slr = self.MSE(
                        output['s_l'].permute(1, 2, 0), output['s_l_r']
                    )
                    loss_sv_slv = self.MSE(
                        output['s_v'].permute(1, 2, 0), output['s_v_r']
                    )
                    loss_sa_sla = self.MSE(
                        output['s_a'].permute(1, 2, 0), output['s_a_r']
                    )
                    loss_s_sr = loss_sl_slr + loss_sv_slv + loss_sa_sla

                    # ort loss L_o
                    if self.args.dataset_name == 'mosi':
                        num = 50
                    elif self.args.dataset_name == 'mosei':
                        num = 10

                    cosine_similarity_s_c_l = self.cosine(
                        output['s_l'].reshape(-1, num),
                        output['c_l'].reshape(-1, num),
                        torch.tensor([-1]).cuda(),
                    )
                    cosine_similarity_s_c_v = self.cosine(
                        output['s_v'].reshape(-1, num),
                        output['c_v'].reshape(-1, num),
                        torch.tensor([-1]).cuda(),
                    )
                    cosine_similarity_s_c_a = self.cosine(
                        output['s_a'].reshape(-1, num),
                        output['c_a'].reshape(-1, num),
                        torch.tensor([-1]).cuda(),
                    )

                    loss_ort = (
                        cosine_similarity_s_c_l
                        + cosine_similarity_s_c_v
                        + cosine_similarity_s_c_a
                    )

                    # triplet margin loss L_m
                    c_l = output['c_l_sim']
                    c_v = output['c_v_sim']
                    c_a = output['c_a_sim']
                    ids, feats = [], []
                    for i in range(labels.size(0)):
                        feats.append(c_l[i].view(1, -1))
                        feats.append(c_v[i].view(1, -1))
                        feats.append(c_a[i].view(1, -1))
                        ids.append(labels[i].view(1, -1))
                        ids.append(labels[i].view(1, -1))
                        ids.append(labels[i].view(1, -1))
                    feats = torch.cat(feats, dim=0)
                    ids = torch.cat(ids, dim=0)
                    loss_sim = self.sim_loss(ids, feats)

                    # overall loss L_DLF
                    combined_loss = loss_task + (
                        loss_s_sr + loss_recon + (loss_sim + loss_ort) * 0.1
                    ) * 0.1

                    combined_loss.backward()

                    if self.args.grad_clip != -1.0:
                        params = list(model[0].parameters())
                        nn.utils.clip_grad_value_(params, self.args.grad_clip)

                    train_loss += combined_loss.item()

                    y_pred.append(output['output_logit'].cpu())
                    y_true.append(labels.cpu())
                    if not left_epochs:
                        optimizer.step()
                        left_epochs = self.args.update_epochs
                if not left_epochs:
                    optimizer.step()

            train_loss = train_loss / len(dataloader['train'])
            pred, true = torch.cat(y_pred), torch.cat(y_true)
            train_results = self.metrics(pred, true)
            logger.info(
                f">> Epoch: {epochs} "
                f"TRAIN -({self.args.model_name}) "
                f"[{epochs - best_epoch}/{epochs}/{self.args.cur_seed}] "
                f">> total_loss: {round(train_loss, 4)} "
                f"{dict_to_str(train_results)}"
            )

            # validation
            val_results = self.do_test(model[0], dataloader['valid'], mode='VAL')
            test_results = self.do_test(model[0], dataloader['test'], mode='TEST')
            cur_valid = val_results[self.args.KeyEval]
            scheduler.step(val_results['Loss'])

            # save best model
            isBetter = (
                cur_valid <= (best_valid - 1e-6)
                if min_or_max == 'min'
                else cur_valid >= (best_valid + 1e-6)
            )
            if isBetter:
                best_valid, best_epoch = cur_valid, epochs
                model_save_path = './pt/DLF' + str(self.args.dataset_name) + '.pth'
                torch.save(model[0].state_dict(), model_save_path)

            if return_epoch_results:
                train_results['Loss'] = train_loss
                epoch_results['train'].append(train_results)
                epoch_results['valid'].append(val_results)
                epoch_results['test'].append(test_results)

            # early stop
            if epochs - best_epoch >= self.args.early_stop:
                return epoch_results if return_epoch_results else None

    def do_test(
        self,
        model,
        dataloader,
        mode='VAL',
        return_sample_results=False,
        expert_analysis=False,
        expert_save_dir=None,
    ):

        model.eval()
        y_pred, y_true = [], []
        eval_loss = 0.0

        if expert_analysis:
            expert_buffers = None
            expert_sample_ids = []

        if return_sample_results:
            ids, sample_results = [], []
            all_labels = []
            features = {
                'Feature_t': [],
                'Feature_a': [],
                'Feature_v': [],
                'Feature_f': [],
            }

        with torch.no_grad():
            with tqdm(dataloader) as td:
                for batch_data in td:
                    vision = batch_data['vision'].to(self.args.device)
                    audio = batch_data['audio'].to(self.args.device)
                    text = batch_data['text'].to(self.args.device)
                    labels = batch_data['labels']['M'].to(self.args.device)
                    labels = labels.view(-1, 1)
                    output = model(text, audio, vision)
                    loss = self.criterion(output['output_logit'], labels)
                    eval_loss += loss.item()
                    y_pred.append(output['output_logit'].cpu())
                    y_true.append(labels.cpu())

                    if expert_analysis:
                        batch_experts = extract_expert_logits(output)
                        if expert_buffers is None:
                            expert_buffers = {
                                name: [] for name in batch_experts.keys()
                            }
                        for name, prediction in batch_experts.items():
                            expert_buffers[name].append(prediction.detach().cpu())
                        expert_sample_ids.extend(
                            normalize_batch_ids(batch_data.get('id'))
                        )

        eval_loss = eval_loss / len(dataloader)
        pred, true = torch.cat(y_pred), torch.cat(y_true)

        eval_results = self.metrics(pred, true)
        eval_results['Loss'] = round(eval_loss, 4)
        logger.info(f"{mode}-({self.args.model_name}) >> {dict_to_str(eval_results)}")

        if expert_analysis:
            if expert_save_dir is None:
                raise ValueError(
                    'expert_save_dir must be provided when expert_analysis=True.'
                )
            expert_predictions = {
                name: torch.cat(buffer, dim=0)
                for name, buffer in expert_buffers.items()
            }
            analyze_expert_pool(
                expert_predictions=expert_predictions,
                targets=true,
                metrics_fn=self.metrics,
                save_dir=expert_save_dir,
                sample_ids=expert_sample_ids,
            )

        if return_sample_results:
            eval_results['Ids'] = ids
            eval_results['SResults'] = sample_results
            for key in features.keys():
                features[key] = np.concatenate(features[key], axis=0)
            eval_results['Features'] = features
            eval_results['Labels'] = all_labels

        return eval_results
