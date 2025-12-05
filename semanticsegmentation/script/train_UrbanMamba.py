import os
import sys

main_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(main_dir)

import argparse
import time

import numpy as np

from UrbanMamba.semanticsegmentation.configs.config import get_config

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from UrbanMamba.semanticsegmentation.datasets.make_data_loader import SemanticDatasetLOVEDA, make_data_loader
from UrbanMamba.semanticsegmentation.models.UrbanMamba import UrbanMamba
import UrbanMamba.semanticsegmentation.utils_func.lovasz_loss as L
from torch.optim.lr_scheduler import StepLR
from UrbanMamba.semanticsegmentation.utils_func.eval import Evaluator
from UrbanMamba.semanticsegmentation.utils_func.postprocess import (
    tta_logits,
    clean_mask,
    crf_refine,
)
from UrbanMamba.semanticsegmentation.datasets import imutils

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
            ) 
        self.deep_model = self.deep_model.cuda()
        self.model_save_path = os.path.join(args.model_param_path, args.dataset,
                                            args.model_type + '_' + str(time.time()))
        self.lr = args.learning_rate
        self.epoch = args.max_iters // args.batch_size

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

        self.optim = optim.AdamW(self.deep_model.parameters(),
                                 lr=args.learning_rate,
                                 weight_decay=args.weight_decay)



        self.scheduler = StepLR(self.optim, step_size=10000, gamma=0.5)

        self.evaluator = Evaluator(args.num_classes)


    def training(self):
        best_iou = 0.0
        best_round = []
        torch.cuda.empty_cache()
        elem_num = len(self.train_data_loader)
        train_enumerator = enumerate(self.train_data_loader)
        cfg = self.config
        for _ in tqdm(range(elem_num)):
            itera, data = train_enumerator.__next__()
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

            self.optim.step()
            self.scheduler.step()

            if (itera + 1) % 10 == 0:
                if (itera + 1) % 500 == 0:
                    self.deep_model.eval()
                    f1, iou, oa = self.validation()
                    if iou > best_iou:
                        torch.save(self.deep_model.state_dict(),
                                   os.path.join(self.model_save_path, f'{itera + 1}_model.pth'))
                        best_iou = iou
                    self.deep_model.train()

        print('The accuracy of the best round is ', best_round)

    def validation(self):
        print('---------starting evaluation-----------')
        cfg = self.config
        dataset = SemanticDatasetLOVEDA(self.args.test_dataset_path, self.args.test_data_name_list, 256, mode='test', cfg=cfg)
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
                    logits = tta_logits(self.deep_model, imgs, use_tta=cfg.POSTPROCESS.TTA)
                else:
                    logits = self.deep_model(imgs)

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
