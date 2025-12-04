import os, sys, pdb
import csv
import random
import argparse
import datetime
import numpy as np
import time
import json
from pathlib import Path

import torch
from torchvision import transforms
import torch.nn.functional as F

import timm
from timm.data.mixup import Mixup
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.utils import ModelEma

from model.resnet import resnet50
from data.dataset import TrainDataset
from engine_finetune import train_one_epoch, evaluate

import utils
from utils import NativeScalerWithGradNormCount as NativeScaler
from utils import str2bool, remap_checkpoint_keys

def get_args_parser():

    parser = argparse.ArgumentParser('Fine-tuning', add_help=False)
    parser.add_argument('--jpeg_factor', type=int, default=None)
    parser.add_argument('--blur_sigma', type=float, default=None)
    parser.add_argument('--mask_ratio', type=float, default=None)
    parser.add_argument('--mask_patch_size', type=int, default=None)
    parser.add_argument('--transform_mode', type=str, default='crop')

    # Training Config
    parser.add_argument('--batch_size', default=64, type=int,
                        help='Per GPU batch size')
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--update_freq', default=1, type=int,
                        help='gradient accumulation steps')

    # Model parameters
    parser.add_argument('--model', default='resnet50', type=str, metavar='MODEL',
                        help='model architecture')
    parser.add_argument('--input_size', default=256, type=int,
                        help='image input size')
    parser.add_argument('--layer_decay_type', type=str, choices=['single', 'group'], default='single',
                        help="""Layer decay strategies. The single strategy assigns a distinct decaying value for each layer,
                        whereas the group strategy assigns the same decaying value for three consecutive layers""")
    
    # EMA related parameters
    parser.add_argument('--model_ema', action='store_true')
    parser.add_argument('--model_ema_decay', type=float, default=0.9999)
    parser.add_argument('--model_ema_force_cpu', action='store_true')
    parser.add_argument('--model_ema_eval', action='store_true', help='Using ema to eval during training.')

    # Optimization parameters
    parser.add_argument('--clip_grad', type=float, default=None, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')
    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=5e-4, metavar='LR',
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--layer_decay', type=float, default=1.0)
    parser.add_argument('--min_lr', type=float, default=1e-6, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0 (1e-6)')
    parser.add_argument('--warmup_epochs', type=int, default=20, metavar='N',
                        help='epochs to warmup LR, if scheduler supports')
    
    parser.add_argument('--warmup_steps', type=int, default=-1, metavar='N',
                        help='num of steps to warmup LR, will overload warmup_epochs if set > 0')    
    parser.add_argument('--opt', default='adamw', type=str, metavar='OPTIMIZER',
                        help='Optimizer (default: "adamw"')
    parser.add_argument('--opt_eps', default=1e-8, type=float, metavar='EPSILON',
                        help='Optimizer Epsilon (default: 1e-8)')
    parser.add_argument('--opt_betas', default=None, type=float, nargs='+', metavar='BETA',
                        help='Optimizer Betas (default: None, use opt default)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight_decay_end', type=float, default=None, help="""Final value of the
        weight decay. We use a cosine schedule for WD and using a larger decay by
        the end of training improves performance for ViTs.""")

    parser.add_argument('--smoothing', type=float, default=0.1, help='Label smoothing (default: 0.1)')

    # Mixup params
    parser.add_argument('--mixup', type=float, default=0.,
                        help='mixup alpha, mixup enabled if > 0.')
    parser.add_argument('--cutmix', type=float, default=0.,
                        help='cutmix alpha, cutmix enabled if > 0.')
    parser.add_argument('--cutmix_minmax', type=float, nargs='+', default=None,
                        help='cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)')
    parser.add_argument('--mixup_prob', type=float, default=1.0,
                        help='Probability of performing mixup or cutmix when either/both is enabled')
    parser.add_argument('--mixup_switch_prob', type=float, default=0.5,
                        help='Probability of switching to cutmix when both mixup and cutmix enabled')
    parser.add_argument('--mixup_mode', type=str, default='batch',
                        help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"')

    # Finetuning params
    parser.add_argument('--pretrained', default=True, help='finetune from checkpoint')
    parser.add_argument('--global_pool', action='store_true')
    parser.set_defaults(global_pool=True)
    parser.add_argument('--head_init_scale', default=0.001, type=float,
                        help='classifier head initial scale, typically adjusted in fine-tuning')
    parser.add_argument('--model_key', default='model|module', type=str,
                        help='which key to load from saved state dict, usually model or model_ema')
    parser.add_argument('--model_prefix', default='', type=str)

    # Dataset parameters
    parser.add_argument('--num_train', default=10000000000, type=int,
                        help="Number of training images, incluing real and fake")
    parser.add_argument('--data_path', default='', type=str,
                        help='dataset path')
    parser.add_argument('--nb_classes', default=2, type=int,
                        help='number of the classification types')
    parser.add_argument('--output_dir', default='',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default=None,
                        help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=None, type=int)
    parser.add_argument('--resume', default='',
                        help='resume from checkpoint')
    
    parser.add_argument('--eval_data_path', default=None, type=str,
                        help='dataset path for evaluation')
    parser.add_argument('--imagenet_default_mean_and_std', type=str2bool, default=True)
    parser.add_argument('--data_set', default='IMNET', choices=['CIFAR', 'IMNET', 'image_folder'],
                        type=str, help='ImageNet dataset path')
    parser.add_argument('--auto_resume', type=str2bool, default=True)
    parser.add_argument('--save_ckpt', type=str2bool, default=True)
    parser.add_argument('--save_ckpt_freq', default=5, type=int)
    parser.add_argument('--save_ckpt_num', default=100, type=int)

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', type=str2bool, default=False,
                        help='Perform evaluation only')
    parser.add_argument('--dist_eval', type=str2bool, default=True,
                        help='Enabling distributed evaluation')
    parser.add_argument('--disable_eval', action='store_true',
                        help='Disabling evaluation during training')
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--pin_mem', type=str2bool, default=True,
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')

    # Evaluation parameters
    parser.add_argument('--crop_pct', type=float, default=None)

    # Distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--local-rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', type=str2bool, default=False)
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')
    
    parser.add_argument('--use_amp', action='store_true', 
                        help="Use apex AMP (Automatic Mixed Precision) or not")

    return parser

def main(args):
    device = torch.device(args.device)
    model = resnet50(num_classes=2)
    model.to(device)
    model.load_state_dict(torch.load(args.resume, weights_only=False)['model'])
    model.eval()
    dataset_val = TrainDataset(is_train=False, args=args, is_inference = True)
    sampler_val = torch.utils.data.SequentialSampler(dataset_val)
    data_loader_val = torch.utils.data.DataLoader(
                dataset_val, sampler=sampler_val,
                batch_size=1 if args.transform_mode == 'source' else args.batch_size,
                num_workers=args.num_workers,
                pin_memory=args.pin_mem,
                drop_last=False
            )
    print("Starting inference...")
    per_image_results = []
    for index, batch in enumerate(data_loader_val):
        images = batch[0]
        image_id = batch[1]
        # target = batch[-1]
        images = images.to(device)
        # target = target.to(device)
        output = model(images)
        if isinstance(output, dict):
            logits = output['logits']
        else:
            logits = output
        probabilities = F.softmax(logits, dim=1)
        confidence, predicted_classes = torch.max(probabilities, dim=1)
        # is_correct_tensor = (predicted_classes == target)

        predicted_classes_np = predicted_classes.cpu().numpy()
        confidence_np = confidence.cpu().detach().numpy()
        # is_correct_np = is_correct_tensor.cpu().numpy()

        for i in range(images.shape[0]):
            current_id = image_id[i] if isinstance(image_id, (list, tuple)) else image_id.item()
            class_name = ["real", "fake"]
            per_image_results.append({
                 'image_id': current_id, 
                 'predicted_class': class_name[int(predicted_classes_np[i])],
                 'confidence': float(confidence_np[i]),
                #  'is_correct': bool(is_correct_np[i])
            })
    print(per_image_results)

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Fine-tuning', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)