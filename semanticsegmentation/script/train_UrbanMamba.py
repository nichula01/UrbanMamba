import os
import sys

main_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(main_dir)

import argparse
import time
import math
import copy

import numpy as np

from UrbanMamba.semanticsegmentation.configs.config import get_config

import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.nn.utils as nn_utils
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from tqdm import tqdm
from UrbanMamba.semanticsegmentation.datasets.make_data_loader import SemanticDatasetLOVEDA, make_data_loader
from UrbanMamba.semanticsegmentation.models.UrbanMamba import UrbanMamba
import UrbanMamba.semanticsegmentation.utils_func.lovasz_loss as L
from UrbanMamba.semanticsegmentation.utils_func.eval import Evaluator
from UrbanMamba.semanticsegmentation.utils_func.postprocess import (
    tta_logits,
    clean_mask,
    crf_refine,
)
from UrbanMamba.semanticsegmentation.datasets import imutils


class ModelEMA(object):
    """Exponential moving average of model parameters for evaluation stability."""

    def __init__(self, model, decay=0.999):
        self.ema = copy.deepcopy(model)
        self.ema.eval()
        self.decay = decay
        for p in self.ema.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        d = self.decay
        msd = model.state_dict()
        esd = self.ema.state_dict()
        for k in esd.keys():
            if k in msd:
                esd[k].data.mul_(d).add_(msd[k].data, alpha=1.0 - d)


def build_optimizer(cfg, model):
    base_lr = cfg.TRAIN.BASE_LR
    wd = cfg.TRAIN.WEIGHT_DECAY
    bb_mult = cfg.TRAIN.BACKBONE_LR_MULT

    backbone_params = []
    other_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "spatial_encoder" in name or "backbone" in name:
            backbone_params.append(param)
        else:
            other_params.append(param)

    param_groups = [
        {"params": backbone_params, "lr": base_lr * bb_mult, "weight_decay": wd},
        {"params": other_params, "lr": base_lr, "weight_decay": wd},
    ]
    optimizer = optim.AdamW(param_groups, lr=base_lr, betas=(0.9, 0.999))
    return optimizer


def build_scheduler(cfg, optimizer, num_steps_per_epoch):
    total_epochs = cfg.TRAIN.EPOCHS
    warmup_epochs = cfg.TRAIN.WARMUP_EPOCHS
    total_steps = total_epochs * num_steps_per_epoch
    warmup_steps = warmup_epochs * num_steps_per_epoch

    scheduler_type = cfg.TRAIN.SCHEDULER.lower()
    min_lr = cfg.TRAIN.MIN_LR
    base_lr = cfg.TRAIN.BASE_LR
    poly_power = cfg.TRAIN.POLY_POWER

    def lr_lambda(current_step):
        if current_step < warmup_steps and warmup_steps > 0:
            return float(current_step) / float(max(1, warmup_steps))
        if current_step >= total_steps:
            return min_lr / base_lr
        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        progress = min(max(progress, 0.0), 1.0)
        if scheduler_type == "cosine":
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            return (min_lr / base_lr) + (1.0 - min_lr / base_lr) * cosine
        if scheduler_type == "poly":
            poly = (1.0 - progress) ** poly_power
            return (min_lr / base_lr) + (1.0 - min_lr / base_lr) * poly
        return 1.0

    scheduler = LambdaLR(optimizer, lr_lambda)
    return scheduler, total_steps

class Trainer(object):
    def __init__(self, args):
        self.args = args
        config = get_config(args)
        self.config = config

        self.train_data_loader = make_data_loader(args, config)

        self.deep_model = UrbanMamba(
            output_clf = args.num_classes,
            pretrained=args.pretrained_weight_path,
            use_nsst=config.FREQ.USE_NSST,
            freq_encoder_type=config.FREQ.ENCODER_TYPE,
            patch_size=config.MODEL.VSSM.PATCH_SIZE, 
            in_chans=config.MODEL.VSSM.IN_CHANS, 
            num_classes=config.MODEL.NUM_CLASSES, 
            depths=config.MODEL.VSSM.DEPTHS, 
            dims=config.MODEL.VSSM.EMBED_DIM, 
            # ===================
            ssm_d_state=config.MODEL.VSSM.SSM_D_STATE,
            ssm_ratio=config.MODEL.VSSM.SSM_RATIO,
            ssm_rank_ratio=config.MODEL.VSSM.SSM_RANK_RATIO,
            ssm_dt_rank=("auto" if config.MODEL.VSSM.SSM_DT_RANK == "auto" else int(config.MODEL.VSSM.SSM_DT_RANK)),
            ssm_act_layer=config.MODEL.VSSM.SSM_ACT_LAYER,
            ssm_conv=config.MODEL.VSSM.SSM_CONV,
            ssm_conv_bias=config.MODEL.VSSM.SSM_CONV_BIAS,
            ssm_drop_rate=config.MODEL.VSSM.SSM_DROP_RATE,
            ssm_init=config.MODEL.VSSM.SSM_INIT,
            forward_type=config.MODEL.VSSM.SSM_FORWARDTYPE,
            # ===================
            mlp_ratio=config.MODEL.VSSM.MLP_RATIO,
            mlp_act_layer=config.MODEL.VSSM.MLP_ACT_LAYER,
            mlp_drop_rate=config.MODEL.VSSM.MLP_DROP_RATE,
            # ===================
            drop_path_rate=config.MODEL.DROP_PATH_RATE,
            patch_norm=config.MODEL.VSSM.PATCH_NORM,
            norm_layer=config.MODEL.VSSM.NORM_LAYER,
            downsample_version=config.MODEL.VSSM.DOWNSAMPLE,
            patchembed_version=config.MODEL.VSSM.PATCHEMBED,
            gmlp=config.MODEL.VSSM.GMLP,
            use_checkpoint=config.TRAIN.USE_CHECKPOINT,
            cfg=config,
            ) 
        self.deep_model = self.deep_model.cuda()
        self.model_save_path = os.path.join(args.model_param_path, args.dataset,
                                            args.model_type + '_' + str(time.time()))
        self.lr = config.TRAIN.BASE_LR
        self.epoch = config.TRAIN.EPOCHS

        if not os.path.exists(self.model_save_path):
            os.makedirs(self.model_save_path)

        if args.resume is not None:
            if not os.path.isfile(args.resume):
                raise RuntimeError("=> no checkpoint found at '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            model_dict = {}
            state_dict = self.deep_model.state_dict()
            for k, v in checkpoint.items():
                if k in state_dict:
                    model_dict[k] = v
            state_dict.update(model_dict)
            self.deep_model.load_state_dict(state_dict)

        self.optim = build_optimizer(config, self.deep_model)
        self.scheduler, self.total_steps = build_scheduler(config, self.optim, len(self.train_data_loader))
        self.global_step = 0
        self.ema = ModelEMA(self.deep_model, decay=config.TRAIN.EMA_DECAY) if getattr(config.TRAIN, "USE_EMA", False) else None

        self.evaluator = Evaluator(args.num_classes)


    def training(self):
        best_iou = 0.0
        best_round = []
        torch.cuda.empty_cache()
        cfg = self.config
        num_epochs = cfg.TRAIN.EPOCHS

        for epoch in range(num_epochs):
            self.deep_model.train()
            for itera, data in enumerate(tqdm(self.train_data_loader, desc=f"Epoch {epoch+1}/{num_epochs}")):
                imgs = data['image']
                labels = data['label']

                imgs = imgs.cuda()
                labels = labels.cuda().long()

                logits = self.deep_model(imgs)

                self.optim.zero_grad()

                class_weights = None
                if hasattr(cfg, "LOSS") and getattr(cfg.LOSS, "CLASS_WEIGHTS", None) is not None:
                    w = torch.tensor(cfg.LOSS.CLASS_WEIGHTS, device=logits.device, dtype=torch.float32)
                    class_weights = w

                ce_loss = F.cross_entropy(
                    logits,
                    labels,
                    weight=class_weights,
                    ignore_index=cfg.TRAIN.IGNORE_LABEL if hasattr(cfg.TRAIN, "IGNORE_LABEL") else 255,
                )

                loss = ce_loss
                if hasattr(cfg, "LOSS") and getattr(cfg.LOSS, "USE_LOVASZ", False):
                    prob = F.softmax(logits, dim=1)
                    lovasz = L.lovasz_softmax(
                        prob,
                        labels,
                        ignore=cfg.TRAIN.IGNORE_LABEL if hasattr(cfg.TRAIN, "IGNORE_LABEL") else 255,
                    )
                    lambda_lovasz = getattr(cfg.LOSS, "LOVASZ_WEIGHT", 0.5)
                    loss = loss + lambda_lovasz * lovasz

                loss.backward()

                max_norm = getattr(cfg.TRAIN, "CLIP_GRAD_NORM", 0.0)
                if max_norm and max_norm > 0:
                    nn_utils.clip_grad_norm_(self.deep_model.parameters(), max_norm)

                self.optim.step()
                self.scheduler.step()
                self.global_step += 1

                if self.ema is not None:
                    self.ema.update(self.deep_model)

            # validation each epoch
            self.deep_model.eval()
            f1, iou, oa = self.validation()
            if iou > best_iou:
                torch.save(self.deep_model.state_dict(),
                           os.path.join(self.model_save_path, f'epoch{epoch + 1}_best.pth'))
                best_iou = iou
            self.deep_model.train()

        print('The accuracy of the best round is ', best_round)

    def validation(self):
        print('---------starting evaluation-----------')
        cfg = self.config
        eval_model = self.ema.ema if getattr(self, "ema", None) is not None else self.deep_model
        if hasattr(cfg.DATA, "VAL_CROP_SIZE"):
            val_crop_size = cfg.DATA.VAL_CROP_SIZE
        else:
            val_crop_size = self.args.crop_size

        dataset = SemanticDatasetLOVEDA(self.args.test_dataset_path, self.args.test_data_name_list, val_crop_size, mode='test', cfg=cfg)
        val_data_loader = DataLoader(dataset, batch_size=1, num_workers=4, drop_last=False)
        torch.cuda.empty_cache()

        self.evaluator.reset()

        preds_all = []
        labels_all = []
        with torch.no_grad():
            for itera, data in enumerate(val_data_loader):
                imgs = data['image']
                labels = data['label']

                imgs = imgs.cuda()
                labels = labels.cuda().long()

                if cfg.POSTPROCESS.ENABLE:
                    logits = tta_logits(eval_model, imgs, use_tta=cfg.POSTPROCESS.TTA)
                else:
                    logits = eval_model(imgs)

                prob = torch.softmax(logits, dim=1)
                preds = torch.argmax(prob, dim=1)

                if cfg.POSTPROCESS.ENABLE and cfg.POSTPROCESS.MORPH:
                    refined_list = []
                    for b in range(preds.size(0)):
                        refined = clean_mask(
                            preds[b],
                            min_areas=cfg.POSTPROCESS.MIN_AREA,
                            apply_open_close=True,
                        )
                        refined_list.append(refined)
                    preds = torch.stack(refined_list, dim=0)

                if cfg.POSTPROCESS.ENABLE and cfg.POSTPROCESS.CRF:
                    refined = []
                    mean = torch.tensor(imutils.MEAN_RGB, device=imgs.device).view(1, 3, 1, 1)
                    std = torch.tensor(imutils.STD_RGB, device=imgs.device).view(1, 3, 1, 1)
                    for b in range(preds.size(0)):
                        img_denorm = imgs[b:b+1] * std + mean
                        img_np = img_denorm[0].permute(1, 2, 0).clamp(0, 255).byte().cpu().numpy()
                        prob_np = prob[b].detach().cpu().numpy()
                        crf_mask = crf_refine(img_np, prob_np)
                        refined.append(torch.from_numpy(crf_mask).to(device=preds.device, dtype=preds.dtype))
                    preds = torch.stack(refined, dim=0)

                labels_np = labels.cpu().numpy()
                preds_np = preds.cpu().numpy()
                self.evaluator.add_batch(gt_image=labels_np, pre_image=preds_np)

        iou_per_class = self.evaluator.Intersection_over_Union()
        f1_per_class = self.evaluator.F1()
        OA = self.evaluator.OA()
        print('mF1:{}, mIOU:{}, OA:{}'.format(np.nanmean(f1_per_class), np.nanmean(iou_per_class), OA))        

        return np.nanmean(f1_per_class), np.nanmean(iou_per_class), OA


def main():
    parser = argparse.ArgumentParser(description="Training on SECOND dataset")
    parser.add_argument('--cfg', type=str, default='/home/songjian/project/UrbanMamba/VMamba/classification/configs/vssm1/vssm_base_224.yaml')
    parser.add_argument(
        "--opts",
        help="Modify config options by adding 'KEY VALUE' pairs. ",
        default=None,
        nargs='+',
    )
    parser.add_argument('--pretrained_weight_path', type=str)

    parser.add_argument('--dataset', type=str, default='SECOND')
    parser.add_argument('--type', type=str, default='train')
    parser.add_argument('--train_dataset_path', type=str, default='/data/ggeoinfo/datasets/xBD/train')
    parser.add_argument('--train_data_list_path', type=str, default='/data/ggeoinfo/datasets/xBD/xBD_list/train_all.txt')
    parser.add_argument('--test_dataset_path', type=str, default='/data/ggeoinfo/datasets/xBD/test')
    parser.add_argument('--test_data_list_path', type=str, default='/data/ggeoinfo/datasets/xBD/xBD_list/val_all.txt')
    parser.add_argument('--shuffle', type=bool, default=True)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--crop_size', type=int, default=512)
    parser.add_argument('--train_data_name_list', type=list)
    parser.add_argument('--test_data_name_list', type=list)
    parser.add_argument('--start_iter', type=int, default=0)
    parser.add_argument('--cuda', type=bool, default=True)
    parser.add_argument('--max_iters', type=int, default=240000)
    parser.add_argument('--model_type', type=str, default='ChangeMambaSCD')
    parser.add_argument('--model_param_path', type=str, default='../saved_models')

    parser.add_argument('--resume', type=str)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=5e-4)

    args = parser.parse_args()
    with open(args.train_data_list_path, "r") as f:
        # data_name_list = f.read()
        data_name_list = [data_name.strip() for data_name in f]
    args.train_data_name_list = data_name_list

    with open(args.test_data_list_path, "r") as f:
        # data_name_list = f.read()
        test_data_name_list = [data_name.strip() for data_name in f]
    args.test_data_name_list = test_data_name_list

    trainer = Trainer(args)
    trainer.training()


if __name__ == "__main__":
    main()
