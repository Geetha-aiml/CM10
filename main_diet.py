# main.py  -- modified to save checkpoint snapshots, curves, classification reports & confusion matrices
# Based on original DeiT main script (integrated additions per user request)

# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
import argparse
import datetime
import numpy as np
import time
import torch
import torch.backends.cudnn as cudnn
import json
import os
import shutil
from pathlib import Path

# plotting & metrics additions
import matplotlib
matplotlib.use('Agg')  # headless
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

from timm.data import Mixup
from timm.models import create_model
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.scheduler import create_scheduler
from timm.optim import create_optimizer
from timm.utils import NativeScaler, get_state_dict, ModelEma

from datasets import build_dataset
from engine import train_one_epoch, evaluate
from losses import DistillationLoss
from samplers import RASampler
from augment import new_data_aug_generator

# import models_old
# import models_v2

import utils


def get_args_parser():
    parser = argparse.ArgumentParser('DeiT training and evaluation script', add_help=False)
    parser.add_argument('--batch-size', default=64, type=int)
    parser.add_argument('--epochs', default=250, type=int)
    parser.add_argument('--bce-loss', action='store_true')
    parser.add_argument('--unscale-lr', action='store_true')

    # Model parameters
    parser.add_argument('--model', default='deit_base_patch16_224', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--input-size', default=224, type=int, help='images input size')

    parser.add_argument('--drop', type=float, default=0.0, metavar='PCT',
                        help='Dropout rate (default: 0.)')
    parser.add_argument('--drop-path', type=float, default=0.1, metavar='PCT',
                        help='Drop path rate (default: 0.1)')

    parser.add_argument('--model-ema', action='store_true')
    parser.add_argument('--no-model-ema', action='store_false', dest='model_ema')
    parser.set_defaults(model_ema=True)
    parser.add_argument('--model-ema-decay', type=float, default=0.99996, help='')
    parser.add_argument('--model-ema-force-cpu', action='store_true', default=False, help='')

    # Optimizer parameters
    parser.add_argument('--opt', default='adamw', type=str, metavar='OPTIMIZER',
                        help='Optimizer (default: "adamw"')
    parser.add_argument('--opt-eps', default=1e-8, type=float, metavar='EPSILON',
                        help='Optimizer Epsilon (default: 1e-8)')
    parser.add_argument('--opt-betas', default=None, type=float, nargs='+', metavar='BETA',
                        help='Optimizer Betas (default: None, use opt default)')
    parser.add_argument('--clip-grad', type=float, default=None, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight-decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')
    # Learning rate schedule parameters
    parser.add_argument('--sched', default='cosine', type=str, metavar='SCHEDULER',
                        help='LR scheduler (default: "cosine"')
    parser.add_argument('--lr', type=float, default=5e-4, metavar='LR',
                        help='learning rate (default: 5e-4)')
    parser.add_argument('--lr-noise', type=float, nargs='+', default=None, metavar='pct, pct',
                        help='learning rate noise on/off epoch percentages')
    parser.add_argument('--lr-noise-pct', type=float, default=0.67, metavar='PERCENT',
                        help='learning rate noise limit percent (default: 0.67)')
    parser.add_argument('--lr-noise-std', type=float, default=1.0, metavar='STDDEV',
                        help='learning rate noise std-dev (default: 1.0)')
    parser.add_argument('--warmup-lr', type=float, default=1e-6, metavar='LR',
                        help='warmup learning rate (default: 1e-6)')
    parser.add_argument('--min-lr', type=float, default=1e-5, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')

    parser.add_argument('--decay-epochs', type=float, default=30, metavar='N',
                        help='epoch interval to decay LR')
    parser.add_argument('--warmup-epochs', type=int, default=5, metavar='N',
                        help='epochs to warmup LR, if scheduler supports')
    parser.add_argument('--cooldown-epochs', type=int, default=10, metavar='N',
                        help='epochs to cooldown LR at min_lr, after cyclic schedule ends')
    parser.add_argument('--patience-epochs', type=int, default=10, metavar='N',
                        help='patience epochs for Plateau LR scheduler (default: 10')
    parser.add_argument('--decay-rate', '--dr', type=float, default=0.1, metavar='RATE',
                        help='LR decay rate (default: 0.1)')

    # Augmentation parameters
    parser.add_argument('--color-jitter', type=float, default=0.0, metavar='PCT',
                        help='Color jitter factor (default: 0.3)')
    parser.add_argument('--aa', type=str, default='none', metavar='NAME',
                        help='Use AutoAugment policy. "v0" or "original". " + \
                             "(default: rand-m9-mstd0.5-inc1)'),
    parser.add_argument('--smoothing', type=float, default=0.0, help='Label smoothing (default: 0.1)')
    parser.add_argument('--train-interpolation', type=str, default='bicubic',
                        help='Training interpolation (random, bilinear, bicubic default: "bicubic")')

    parser.add_argument('--repeated-aug', action='store_true')
    parser.add_argument('--no-repeated-aug', action='store_false', dest='repeated_aug')
    parser.set_defaults(repeated_aug=True)
    
    parser.add_argument('--train-mode', action='store_true')
    parser.add_argument('--no-train-mode', action='store_false', dest='train_mode')
    parser.set_defaults(train_mode=True)
    
    parser.add_argument('--ThreeAugment', action='store_false') #3augment
    
    parser.add_argument('--src', action='store_false') #simple random crop
    
    # * Random Erase params
    parser.add_argument('--reprob', type=float, default=0.0, metavar='PCT',
                        help='Random erase prob (default: 0.25)')
    parser.add_argument('--remode', type=str, default='pixel',
                        help='Random erase mode (default: "pixel")')
    parser.add_argument('--recount', type=int, default=1,
                        help='Random erase count (default: 1)')
    parser.add_argument('--resplit', action='store_true', default=False,
                        help='Do not random erase first (clean) augmentation split')

    # * Mixup params
    parser.add_argument('--mixup', type=float, default=0.8,
                        help='mixup alpha, mixup enabled if > 0. (default: 0.8)')
    parser.add_argument('--cutmix', type=float, default=1.0,
                        help='cutmix alpha, cutmix enabled if > 0. (default: 1.0)')
    parser.add_argument('--cutmix-minmax', type=float, nargs='+', default=None,
                        help='cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)')
    parser.add_argument('--mixup-prob', type=float, default=1.0,
                        help='Probability of performing mixup or cutmix when either/both is enabled')
    parser.add_argument('--mixup-switch-prob', type=float, default=0.5,
                        help='Probability of switching to cutmix when both mixup and cutmix enabled')
    parser.add_argument('--mixup-mode', type=str, default='batch',
                        help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"')

    # Distillation parameters
    parser.add_argument('--teacher-model', default='regnety_160', type=str, metavar='MODEL',
                        help='Name of teacher model to train (default: "regnety_160"')
    parser.add_argument('--teacher-path', type=str, default='')
    parser.add_argument('--distillation-type', default='none', choices=['none', 'soft', 'hard'], type=str, help="")
    parser.add_argument('--distillation-alpha', default=0.5, type=float, help="")
    parser.add_argument('--distillation-tau', default=1.0, type=float, help="")
    
    # * Cosub params
    parser.add_argument('--cosub', action='store_true') 
    
    # * Finetuning params
    parser.add_argument('--finetune', default='', help='finetune from checkpoint')
    parser.add_argument('--attn-only', action='store_true') 
    
    # Dataset parameters
    parser.add_argument('--data-path', default='/datasets01/imagenet_full_size/061417/', type=str,
                        help='dataset path')
    parser.add_argument('--data-set', default='IMNET', choices=['CIFAR', 'IMNET', 'INAT', 'INAT19'],
                        type=str, help='Image Net dataset path')
    parser.add_argument('--inat-category', default='name',
                        choices=['kingdom', 'phylum', 'class', 'order', 'supercategory', 'family', 'genus', 'name'],
                        type=str, help='semantic granularity')

    parser.add_argument('--output_dir', default='',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
    parser.add_argument('--eval-crop-ratio', default=0.875, type=float, help="Crop ratio for evaluation")
    parser.add_argument('--dist-eval', action='store_true', default=False, help='Enabling distributed evaluation')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin-mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no-pin-mem', action='store_false', dest='pin_mem',
                        help='')
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument('--distributed', action='store_true', default=False, help='Enabling distributed training')
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    return parser


# ---------------------------
# Additional utilities added
# ---------------------------
# Epochs where we save snapshots + reports
SAVE_EPOCHS = {10, 20, 50, 100, 150, 200, 250}


def _safe_get_metric(dct, possible_keys):
    """Try multiple possible keys in dictionary `dct` and return first non-None."""
    if dct is None:
        return None
    for k in possible_keys:
        if k in dct and dct[k] is not None:
            return dct[k]
    # some train_stats values can be nested; try numeric values
    for v in dct.values():
        if isinstance(v, (int, float)):
            # best guess (not perfect)
            return v
    return None


def save_checkpoint_copy(output_dir: Path, epoch: int):
    """Copy the standard checkpoint.pth to checkpoint_epoch_{epoch}.pth"""
    src = output_dir / "checkpoint.pth"
    dst = output_dir / f"checkpoint_epoch_{epoch}.pth"
    try:
        if src.exists():
            shutil.copyfile(src, dst)
            print(f"[SAVE] Copied checkpoint -> {dst}")
        else:
            print(f"[WARN] No checkpoint.pth found to copy at epoch {epoch}")
    except Exception as e:
        print(f"[ERROR] copying checkpoint for epoch {epoch}: {e}")


def ensure_dir(path: Path):
    if not path.exists():
        path.mkdir(parents=True, exist_ok=True)


def evaluate_and_save_reports(model_for_eval, data_loader_val, device, outdir: Path, nb_classes=None):
    """
    Run inference on data_loader_val, produce classification report + confusion matrix and save them.
    model_for_eval : should be the nn.Module (not DDP wrapper) used for inference.
    """
    ensure_dir(outdir)
    model_for_eval.eval()
    preds_list = []
    targets_list = []
    with torch.no_grad():
        for images, labels in data_loader_val:
            images = images.to(device)
            outs = model_for_eval(images)
            # if model returns tuple (e.g. with distillation), handle it
            if isinstance(outs, (tuple, list)):
                outs = outs[0]
            _, preds = torch.max(outs, 1)
            preds_list.append(preds.cpu().numpy())
            targets_list.append(labels.cpu().numpy())
    if len(preds_list) == 0:
        print("[WARN] No predictions collected in evaluate_and_save_reports()")
        return
    preds = np.concatenate(preds_list)
    targets = np.concatenate(targets_list)

    # classification report
    clf_report = classification_report(targets, preds, output_dict=True, zero_division=0)
    with (outdir / "classification_report.json").open("w") as f:
        json.dump(clf_report, f, indent=2)

    # confusion matrix
    cm = confusion_matrix(targets, preds)
    np.save(outdir / "confusion_matrix.npy", cm)

    # plot confusion matrix heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=False)
    plt.title("Confusion matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(outdir / "confusion_matrix.png")
    plt.close()

    print(f"[EVAL] Saved classification_report and confusion matrix to {outdir}")


# ---------------------------
# End of additional utilities
# ---------------------------


def main(args):
    utils.init_distributed_mode(args)

    print(args)

    if args.distillation_type != 'none' and args.finetune and not args.eval:
        raise NotImplementedError("Finetuning with distillation not yet supported")

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    # random.seed(seed)

    cudnn.benchmark = True

    dataset_train, args.nb_classes = build_dataset(is_train=True, args=args)
    dataset_val, _ = build_dataset(is_train=False, args=args)

    if args.distributed:
        num_tasks = utils.get_world_size()
        global_rank = utils.get_rank()
        if args.repeated_aug:
            sampler_train = RASampler(
                dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
            )
        else:
            sampler_train = torch.utils.data.DistributedSampler(
                dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
            )
        if args.dist_eval:
            if len(dataset_val) % num_tasks != 0:
                print('Warning: Enabling distributed evaluation with an eval dataset not divisible by process number. '
                      'This will slightly alter validation results as extra duplicate entries are added to achieve '
                      'equal num of samples per-process.')
            sampler_val = torch.utils.data.DistributedSampler(
                dataset_val, num_replicas=num_tasks, rank=global_rank, shuffle=False)
        else:
            sampler_val = torch.utils.data.SequentialSampler(dataset_val)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
        persistent_workers=True if args.num_workers > 0 else False,
    )
    
    # data_loader_train.dataset.transform = new_data_aug_generator(args)

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, sampler=sampler_val,
        batch_size=int(1.5 * args.batch_size),
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False,
        persistent_workers=True if args.num_workers > 0 else False,
    )

    # lists to store metrics for plotting
    collected_epochs = []
    train_accs = []
    val_accs = []
    train_losses = []
    val_losses = []

    mixup_fn = None
    mixup_active=False
    if mixup_active:
        mixup_fn = Mixup(
            mixup_alpha=args.mixup, cutmix_alpha=args.cutmix, cutmix_minmax=args.cutmix_minmax,
            prob=args.mixup_prob, switch_prob=args.mixup_switch_prob, mode=args.mixup_mode,
            label_smoothing=args.smoothing, num_classes=args.nb_classes)

    print(f"Creating model: {args.model}")
    model = create_model(
        args.model,
        pretrained=False,
        num_classes=args.nb_classes,
        drop_rate=args.drop,
        # drop_path_rate=args.drop_path,
        drop_block_rate=None,
        # img_size=args.input_size
    )

                    
    if args.finetune:
        if args.finetune.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.finetune, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.finetune, map_location='cpu')

        checkpoint_model = checkpoint['model']
        state_dict = model.state_dict()
        for k in ['head.weight', 'head.bias', 'head_dist.weight', 'head_dist.bias']:
            if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint_model[k]

        # interpolate position embedding
        pos_embed_checkpoint = checkpoint_model['pos_embed']
        embedding_size = pos_embed_checkpoint.shape[-1]
        num_patches = model.patch_embed.num_patches
        num_extra_tokens = model.pos_embed.shape[-2] - num_patches
        # height (== width) for the checkpoint position embedding
        orig_size = int((pos_embed_checkpoint.shape[-2] - num_extra_tokens) ** 0.5)
        # height (== width) for the new position embedding
        new_size = int(num_patches ** 0.5)
        # class_token and dist_token are kept unchanged
        extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
        # only the position tokens are interpolated
        pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
        pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, embedding_size).permute(0, 3, 1, 2)
        pos_tokens = torch.nn.functional.interpolate(
            pos_tokens, size=(new_size, new_size), mode='bicubic', align_corners=False)
        pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
        new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
        checkpoint_model['pos_embed'] = new_pos_embed

        model.load_state_dict(checkpoint_model, strict=False)
        
    if args.attn_only:
        for name_p,p in model.named_parameters():
            if '.attn.' in name_p:
                p.requires_grad = True
            else:
                p.requires_grad = False
        try:
            model.head.weight.requires_grad = True
            model.head.bias.requires_grad = True
        except:
            model.fc.weight.requires_grad = True
            model.fc.bias.requires_grad = True
        try:
            model.pos_embed.requires_grad = True
        except:
            print('no position encoding')
        try:
            for p in model.patch_embed.parameters():
                p.requires_grad = False
        except:
            print('no patch embed')
            
    model.to(device)

    model_ema = None
    if args.model_ema:
        # Important to create EMA model after cuda(), DP wrapper, and AMP but before SyncBN and DDP wrapper
        model_ema = ModelEma(
            model,
            decay=args.model_ema_decay,
            device='cpu' if args.model_ema_force_cpu else '',
            resume='')

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)
    if not args.unscale_lr:
        linear_scaled_lr = args.lr * args.batch_size * utils.get_world_size() / 512.0
        args.lr = linear_scaled_lr
    optimizer = create_optimizer(args, model_without_ddp)
    loss_scaler = NativeScaler()

    lr_scheduler, _ = create_scheduler(args, optimizer)

    criterion = LabelSmoothingCrossEntropy()

    if mixup_active:
        # smoothing is handled with mixup label transform
        criterion = SoftTargetCrossEntropy()
    elif args.smoothing:
        criterion = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
    else:
        criterion = torch.nn.CrossEntropyLoss()
        
    if args.bce_loss:
        criterion = torch.nn.BCEWithLogitsLoss()
        
    teacher_model = None
    if args.distillation_type != 'none':
        assert args.teacher_path, 'need to specify teacher-path when using distillation'
        print(f"Creating teacher model: {args.teacher_model}")
        teacher_model = create_model(
            args.teacher_model,
            pretrained=False,
            num_classes=args.nb_classes,
            global_pool='avg',
        )
        if args.teacher_path.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.teacher_path, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.teacher_path, map_location='cpu')
        teacher_model.load_state_dict(checkpoint['model'])
        teacher_model.to(device)
        teacher_model.eval()

    # wrap the criterion in our custom DistillationLoss, which
    # just dispatches to the original criterion if args.distillation_type is 'none'
    criterion = DistillationLoss(
        criterion, teacher_model, args.distillation_type, args.distillation_alpha, args.distillation_tau
    )

    output_dir = Path(args.output_dir)
    if args.resume:
        if args.resume.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.resume, map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'])
        if not args.eval and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            args.start_epoch = checkpoint['epoch'] + 1
            if args.model_ema:
                utils._load_checkpoint_for_ema(model_ema, checkpoint['model_ema'])
            if 'scaler' in checkpoint:
                loss_scaler.load_state_dict(checkpoint['scaler'])
        lr_scheduler.step(args.start_epoch)
    if args.eval:
        test_stats = evaluate(data_loader_val, model, device)
        print(f"Accuracy of the network on the {len(dataset_val)} test images: {test_stats['acc1']:.1f}%")

        # Print table cleanly
        print("\n" + "-" * 75)
        print(f"{'Epoch':<10}{'Train Acc':<15}{'Val Acc':<15}{'Train Loss':<15}{'Val Loss':<15}")
        print("-" * 75)
        print(f"{args.start_epoch:<10}{0:<15}{test_stats['acc1']:<15.4f}{0:<15}{0:<15}")
        print("-" * 75 + "\n")

        return


        #         # ----------- WRITE TABLE LOG INTO log.txt -----------
        #         # -------- WRITE CLEAN TABLE FORMAT INTO log.txt --------
        # if args.output_dir and utils.is_main_process():
        #     log_file = output_dir / "log.txt"

        #     # Write header only once (epoch 0)
        #     if epoch == args.start_epoch:
        #         with log_file.open("a") as f:
        #             f.write("\n" + "-" * 85 + "\n")
        #             f.write(f"{'Epoch':<10}{'Train Acc':<15}{'Val Acc':<15}{'Train Loss':<15}{'Val Loss':<15}{'LR':<15}\n")
        #             f.write("-" * 85 + "\n")

        #     # Fetch LR
        #     current_lr = optimizer.param_groups[0]["lr"]

        #     # Write row
        #     with log_file.open("a") as f:
        #         f.write(f"{epoch:<10}{train_acc_disp:<15.4f}{val_acc_disp:<15.4f}"
        #                 f"{train_loss_disp:<15.4f}{val_loss_disp:<15.4f}{current_lr:<15.6f}\n")



        # return

    # Ensure output dir exists
    if args.output_dir:
        ensure_dir(output_dir)

    # results dir to put classification reports + confusion matrices + curves
    results_dir = output_dir / "results"
    ensure_dir(results_dir)

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    max_accuracy = 0.0

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)

        # regular training step (unchanged)
        train_stats = train_one_epoch(
            model, criterion, data_loader_train,
            optimizer, device, epoch, loss_scaler,
            args.clip_grad, model_ema, mixup_fn,
            set_training_mode=args.train_mode,
            args=args,
        )

        # ---------- QUICK TRAIN ACC ESTIMATE (keeps your max_batches approach) ----------
        # Use a small number of batches to estimate train accuracy quickly.
        model_without_ddp.eval()
        correct = 0
        total = 0
        max_batches = 10  # small quick-check; increase if you want a better estimate
        with torch.no_grad():
            for i, (images, labels) in enumerate(data_loader_train):
                if i >= max_batches:
                    break
                images = images.to(device)
                labels = labels.to(device)

                outputs = model_without_ddp(images)
                if isinstance(outputs, (tuple, list)):
                    outputs = outputs[0]
                _, preds = outputs.max(1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        train_acc = 100.0 * correct / total if total > 0 else float('nan')
        model_without_ddp.train()
        # ------------------------------------------------------------------------------

        # step scheduler and save regular checkpoint
        lr_scheduler.step(epoch)
        if args.output_dir:
            checkpoint_paths = [output_dir / 'checkpoint.pth']
            for checkpoint_path in checkpoint_paths:
                utils.save_on_master({
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch,
                    'model_ema': get_state_dict(model_ema),
                    'scaler': loss_scaler.state_dict(),
                    'args': args,
                }, checkpoint_path)

        # evaluate on validation set
        test_stats = evaluate(data_loader_val, model, device)
        print(f"Accuracy of the network on the {len(dataset_val)} test images: {test_stats.get('acc1', test_stats.get('accuracy', 'N/A')):.1f}%")

        # update best checkpoint when validation improves
        current_val_acc = test_stats.get("acc1") or test_stats.get("accuracy") or 0.0
        if max_accuracy < current_val_acc:
            max_accuracy = current_val_acc
            if args.output_dir:
                checkpoint_paths = [output_dir / 'best_checkpoint.pth']
                for checkpoint_path in checkpoint_paths:
                    utils.save_on_master({
                        'model': model_without_ddp.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'lr_scheduler': lr_scheduler.state_dict(),
                        'epoch': epoch,
                        'model_ema': get_state_dict(model_ema),
                        'scaler': loss_scaler.state_dict(),
                        'args': args,
                    }, checkpoint_path)
        print(f'Max accuracy: {max_accuracy:.2f}%')

        # ---------------- collect metrics ----------------
        # keep your manual train_acc (percentage) and get losses/val_acc safely
        train_acc_value = train_acc  # manual quick estimate (0-100)
        train_loss_value = _safe_get_metric(train_stats, ['loss', 'train_loss', 'avg_loss'])
        val_acc = _safe_get_metric(test_stats, ['acc1', 'acc', 'top1', 'accuracy', 'test_acc1'])
        val_loss = _safe_get_metric(test_stats, ['loss', 'test_loss', 'avg_loss'])

        collected_epochs.append(epoch)
        train_accs.append(train_acc_value)
        val_accs.append(val_acc if val_acc is not None else np.nan)
        train_losses.append(train_loss_value if train_loss_value is not None else np.nan)
        val_losses.append(val_loss if val_loss is not None else np.nan)
        # -------------------------------------------------

        # -------- PRINT CLEAN TABLE FORMAT (console) --------
        ta = train_acc_value if not np.isnan(train_acc_value) else float('nan')
        va = val_acc if val_acc is not None else float('nan')
        tl = train_loss_value if train_loss_value is not None else float('nan')
        vl = val_loss if val_loss is not None else float('nan')

        print("\n" + "-" * 85)
        print(f"{'Epoch':<10}{'Train Acc%':<15}{'Val Acc%':<15}{'Train Loss':<15}{'Val Loss':<15}{'LR':<15}")
        print("-" * 85)
        print(f"{epoch:<10}{ta:<15.2f}{va:<15.2f}{tl:<15.4f}{vl:<15.4f}{optimizer.param_groups[0]['lr']:<15.6f}")
        print("-" * 85 + "\n")

        # -------- WRITE TABLE LOG INTO log.txt (one row per epoch) --------
        if args.output_dir and utils.is_main_process():
            log_file = output_dir / "log.txt"
            # header if first epoch
            if epoch == args.start_epoch and not log_file.exists():
                with log_file.open("a") as f:
                    f.write("\n" + "-" * 120 + "\n")
                    f.write(f"{'Epoch':<10}{'Train Acc%':<15}{'Val Acc%':<15}{'Train Loss':<15}{'Val Loss':<15}{'LR':<15}{'Time(s)':<15}\n")
                    f.write("-" * 120 + "\n")

            # write row
            elapsed = int(time.time() - start_time)
            with log_file.open("a") as f:
                f.write(f"{epoch:<10}{ta:<15.2f}{va:<15.2f}{tl:<15.4f}{vl:<15.4f}{optimizer.param_groups[0]['lr']:<15.6f}{elapsed:<15d}\n")

        # ---- save checkpoint copy + evaluation + plots at target epochs ----
        if args.output_dir and epoch in SAVE_EPOCHS:
            epoch_outdir = results_dir / f"epoch_{epoch}"
            ensure_dir(epoch_outdir)
            try:
                save_checkpoint_copy(output_dir, epoch)
            except Exception as e:
                print(f"[WARN] Could not copy checkpoint for epoch {epoch}: {e}")

            try:
                model_for_eval = model_without_ddp
                evaluate_and_save_reports(model_for_eval, data_loader_val, device, epoch_outdir, nb_classes=args.nb_classes)
            except Exception as e:
                print(f"[ERROR] evaluate_and_save_reports failed for epoch {epoch}: {e}")

            # Save current curves (partial)
            try:
                plt.figure(figsize=(7,5))
                plt.plot(collected_epochs, train_accs, marker='o', label='train_acc')
                plt.plot(collected_epochs, val_accs, marker='o', label='val_acc')
                plt.xlabel('Epoch')
                plt.ylabel('Accuracy')
                plt.title('Train vs Val Accuracy (so far)')
                plt.legend()
                plt.grid(True)
                plt.tight_layout()
                plt.savefig(results_dir / f"accuracy_curve_until_epoch_{epoch}.png")
                plt.close()

                plt.figure(figsize=(7,5))
                plt.plot(collected_epochs, train_losses, marker='o', label='train_loss')
                plt.plot(collected_epochs, val_losses, marker='o', label='val_loss')
                plt.xlabel('Epoch')
                plt.ylabel('Loss')
                plt.title('Train vs Val Loss (so far)')
                plt.legend()
                plt.grid(True)
                plt.tight_layout()
                plt.savefig(results_dir / f"loss_curve_until_epoch_{epoch}.png")
                plt.close()
            except Exception as e:
                print(f"[WARN] could not save partial curves at epoch {epoch}: {e}")

    # ---- after training: save final curves and a summary ----
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))

    # Save final curves
    try:
        # full accuracy curve
        plt.figure(figsize=(7,5))
        plt.plot(collected_epochs, train_accs, marker='o', label='train_acc')
        plt.plot(collected_epochs, val_accs, marker='o', label='val_acc')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title('Train vs Val Accuracy')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(results_dir / "accuracy_curve.png")
        plt.close()

        # full loss curve
        plt.figure(figsize=(7,5))
        plt.plot(collected_epochs, train_losses, marker='o', label='train_loss')
        plt.plot(collected_epochs, val_losses, marker='o', label='val_loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Train vs Val Loss')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(results_dir / "loss_curve.png")
        plt.close()

        # Save a small JSON summary
        summary = {
            "total_time_seconds": int(total_time),
            "total_time_human": total_time_str,
            "n_parameters": int(n_parameters),
            "epochs_ran": collected_epochs,
            "train_accs": [None if np.isnan(x) else float(x) for x in train_accs],
            "val_accs": [None if np.isnan(x) else float(x) for x in val_accs],
        }
        with (results_dir / "training_summary.json").open("w") as f:
            json.dump(summary, f, indent=2)

        print(f"[SAVE] Final curves and summary saved to {results_dir}")
    except Exception as e:
        print(f"[WARN] could not save final curves/summary: {e}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser('DeiT training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
