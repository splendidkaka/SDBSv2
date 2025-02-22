# ------------------------------------------------------------------------------------
# Sparse DETR
# Copyright (c) 2021 KakaoBrain. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------------------
# Modified from Deformable DETR (https://github.com/fundamentalvision/Deformable-DETR)
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# ------------------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------------------
# Modified by Sihan Ge
# ------------------------------------------------------------------------------------


import argparse
import datetime
import json
import random
import time
from tabulate import tabulate
from pathlib import Path
from image import *
import os
import math
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

import torch.nn as nn

from utils import save_checkpoint
import datasets
import util.misc as utils
import datasets.samplers as samplers
from datasets import build_dataset, get_coco_api_from_dataset
from engine import evaluate, train_one_epoch
from models import build_model
from util.benchmark import compute_fps, compute_gflops

from utils import get_root_logger, setup_seed
from torch.utils.tensorboard import SummaryWriter
import dataset
from torchvision import transforms

def get_args_parser():
    parser = argparse.ArgumentParser('Deformable DETR Detector', add_help=False)
    parser.add_argument('--lr', default=0.0001, type=float)
    parser.add_argument('--lr_backbone_names', default=["backbone.0"], type=str, nargs='+')
    parser.add_argument('--lr_backbone', default=0.0001, type=float)
    parser.add_argument('--lr_linear_proj_names', default=['reference_points', 'sampling_offsets'], type=str, nargs='+')
    parser.add_argument('--lr_linear_proj_mult', default=0.1, type=float)
    parser.add_argument('--batch_size', default=8, type=int)
    parser.add_argument('--weight_decay', default=5 * 1e-4, type=float)
    parser.add_argument('--epochs', default=1000, type=int)
    parser.add_argument('--lr_drop', default=40, type=int)
    parser.add_argument('--lr_drop_epochs', default=None, type=int, nargs='+')
    parser.add_argument('--clip_max_norm', default=0.1, type=float,
                        help='gradient clipping max norm')


    parser.add_argument('--sgd', action='store_true')

    # Variants of Deformable DETR
    parser.add_argument('--with_box_refine', default=False, action='store_true')
    parser.add_argument('--two_stage', default=False, action='store_true')

    # Model parameters
    parser.add_argument('--frozen_weights', type=str, default=None,
                        help="Path to the pretrained model. If set, only the mask head will be trained")

    # * Backbone
    parser.add_argument('--backbone', default='resnet50', type=str,
                        help="Name of the convolutional backbone to use")
    parser.add_argument('--dilation', action='store_true',
                        help="If true, we replace stride with dilation in the last convolutional block (DC5)")
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                        help="Type of positional embedding to use on top of the image features")
    parser.add_argument('--position_embedding_scale', default=2 * np.pi, type=float,
                        help="position / size * scale")
    parser.add_argument('--num_feature_levels', default=4, type=int, help='number of feature levels')

    # * Modified architecture
    parser.add_argument('--backbone_from_scratch', default=False, action='store_true')
    parser.add_argument('--finetune_early_layers', default=False, action='store_true')
    parser.add_argument('--scrl_pretrained_path', default='', type=str)

    # * Transformer
    parser.add_argument('--enc_layers', default=6, type=int,
                        help="Number of encoding layers in the transformer")
    parser.add_argument('--dec_layers', default=6, type=int,
                        help="Number of decoding layers in the transformer")
    parser.add_argument('--dim_feedforward', default=1024, type=int,
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--hidden_dim', default=256, type=int,
                        help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--dropout', default=0.1, type=float,
                        help="Dropout applied in the transformer")
    parser.add_argument('--nheads', default=8, type=int,
                        help="Number of attention heads inside the transformer's attentions")
    # parser.add_argument('--num_queries', default=300, type=int,
    #                     help="Number of query slots")
    parser.add_argument('--dec_n_points', default=4, type=int)
    parser.add_argument('--enc_n_points', default=4, type=int)
    
    # * Efficient DETR
    parser.add_argument('--eff_query_init', default=False, action='store_true')
    parser.add_argument('--eff_specific_head', default=False, action='store_true')

    # * Sparse DETR
    parser.add_argument('--use_enc_aux_loss', default=False, action='store_true')
    parser.add_argument('--rho', default=0.2, type=float)

    # * Segmentation
    parser.add_argument('--masks', action='store_true',
                        help="Train segmentation head if the flag is provided")

    # Loss
    parser.add_argument('--no_aux_loss', dest='aux_loss', action='store_false',
                        help="Disables auxiliary decoding losses (loss at each layer)")

    # * Matcher
    parser.add_argument('--set_cost_class', default=2, type=float,
                        help="Class coefficient in the matching cost")
    parser.add_argument('--set_cost_bbox', default=5, type=float,
                        help="L1 box coefficient in the matching cost")
    parser.add_argument('--set_cost_point', default=5, type=float,
                        help="L1 box coefficient in the matching cost")
    parser.add_argument('--set_cost_giou', default=2, type=float,
                        help="giou box coefficient in the matching cost")

    # * Loss coefficients
    parser.add_argument('--mask_loss_coef', default=1, type=float)
    parser.add_argument('--dice_loss_coef', default=1, type=float)
    parser.add_argument('--cls_loss_coef', default=2, type=float)
#     parser.add_argument('--point_loss_coef', default=5, type=float)
    parser.add_argument('--giou_loss_coef', default=2, type=float)
    parser.add_argument('--mask_prediction_coef', default=1, type=float)
    parser.add_argument('--focal_alpha', default=0.25, type=float)

    # * dataset parameters
    parser.add_argument('--dataset_file', default='coco')
    parser.add_argument('--coco_path', default='/root/autodl-tmp', type=str)
    parser.add_argument('--coco_panoptic_path', type=str)
    parser.add_argument('--remove_difficult', action='store_true')

    parser.add_argument('--output_dir', default='res',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--num_workers', default=2, type=int)
    parser.add_argument('--cache_mode', default=False, action='store_true', help='whether to cache images on memory')
    
    # * benchmark
    parser.add_argument('--approx_benchmark_only', default=False, action='store_true')
    parser.add_argument('--benchmark_only', default=False, action='store_true')
    parser.add_argument('--no_benchmark', dest='benchmark', action='store_false')


    parser.add_argument('--dataset', default='jhu')
    parser.add_argument('--save_path', type=str, default='valid_res',
                        help='save checkpoint directory')
    parser.add_argument('--save_path_log', type=str, default='',
                        help='save checkpoint directory')
    parser.add_argument('--gray_aug', action='store_true',
                        help='using the gray augmentation')
    parser.add_argument('--gray_p', type=float, default=0.3,# JH default 0.3 NW 0.1
                        help='probability of gray')
    parser.add_argument('--crop_size', type=int, default=256,
                        help='crop size for training')
    parser.add_argument('--scale_aug', action='store_true',
                        help='using the scale augmentation')
    parser.add_argument('--scale_p', type=float, default=0.3,
                        help='probability of scaling')
    parser.add_argument('--scale_type', type=int, default=1,# default=0
                        help='scale type')
    parser.add_argument('--num_patch', type=int, default=1,
                        help='number of patches')
    parser.add_argument('--min_num', type=int, default=-1,
                        help='min_num')
    parser.add_argument('--num_queries', default=500, type=int,
                        help="Number of query slots")
    parser.add_argument('--channel_point', type=int, default=3,
                        help='number of boxes')
    parser.add_argument('--num_knn', type=int, default=4,
                        help='number of knn')

    # distributed training parameters !
    parser.add_argument('--local_rank', type=int, default=-1,
                        help='local local_rank')


    parser.add_argument('--pre', type=str, default=None,
                        help='pre-trained model directory')
    parser.add_argument('--best_pred', type=int, default=1e5,
                        help='best pred')
    parser.add_argument('--threshold', type=float, default=0.35,
                        help='threshold to filter the negative points')
    parser.add_argument('--point_loss_coef', default=5, type=float)
    parser.add_argument('--lr_step', type=int, default=1200,
                    help='lr_step')

    
    return parser

def collate_wrapper(batch):
    targets = []
    imgs = []
    fname = []

    for item in batch:

        #if return_args.train_patch:
        fname.append(item[0])

        for i in range(0, len(item[1])):
            imgs.append(item[1][i])

        for i in range(0, len(item[2])):
            targets.append(item[2][i])
        # else:
        # fname.append(item[0])
        # imgs.append(item[1])
        # targets.append(item[2])

    return  torch.stack(imgs, 0), targets

def main(args):
    utils.init_distributed_mode(args)
    print("git:\n  {}\n".format(utils.get_sha()))

    if args.frozen_weights is not None:
        assert args.masks, "Frozen training is meant for segmentation only"
    print(args)

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    model, criterion, postprocessors = build_model(args)
    model.to(device)
    model_without_ddp = model
    
    # dataset_val_org = build_dataset(image_set='val', args=args)
    

    path = 'log_file/debug/'
    args.save_path_log = path
    if not os.path.exists(args.save_path_log):
        os.makedirs(path)
    logger = get_root_logger(path + 'debug.log')
    writer = SummaryWriter(path)
    logger.info(args)
    
#     if args.approx_benchmark_only or args.benchmark_only:
#         assert not args.distributed and args.benchmark
    
#     if utils.is_main_process() and args.benchmark:
#         n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
#         if args.benchmark_only:
#             gflops = compute_gflops(model, dataset_val_org, approximated=False)
#         else:
#             gflops = compute_gflops(model, dataset_val_org, approximated=True)
#         fps = compute_fps(model, dataset_val_org, num_iters=20, batch_size=1)
#         bfps = compute_fps(model, dataset_val_org, num_iters=20, batch_size=4)
#         tab_keys = ["#Params(M)", "GFLOPs", "FPS", "B4FPS"]
#         tab_vals = [n_params / 10 ** 6, gflops, fps, bfps]
#         table = tabulate([tab_vals], headers=tab_keys, tablefmt="pipe",
#                         floatfmt=".3f", stralign="center", numalign="center")
#         print("===== Benchmark (Crude Approx.) =====\n" + table)
        
#     if args.approx_benchmark_only or args.benchmark_only:
#         import sys; sys.exit()
            



#     data_loader_train = DataLoader(dataset_train, batch_sampler=batch_sampler_train,
#                                    collate_fn=utils.collate_fn, num_workers=args.num_workers,
#                                    pin_memory=True)
#     data_loader_val = DataLoader(dataset_val, args.batch_size, sampler=sampler_val,
#                                  drop_last=False, collate_fn=utils.collate_fn, num_workers=args.num_workers,
#                                  pin_memory=True)

# jhu
    if args.dataset=='jhu':
        train_file = './npydata/jhu_train.npy'
        test_file = './npydata/jhu_val.npy'
# nwpu
    if args.dataset == 'nwpu':

        train_file = './npydata/nwpu_train.npy'
        test_file = './npydata/nwpu_val.npy'



    with open(train_file, 'rb') as outfile:
        train_data = np.load(outfile).tolist()
    with open(test_file, 'rb') as outfile:
        test_data = np.load(outfile).tolist()

    args = utils.scale_learning_rate(args)

    train_data = dataset.listDataset(train_data, args.save_path,
                                     shuffle=True,
                                     transform=transforms.Compose([
                                         transforms.RandomGrayscale(p=args.gray_p if args.gray_aug else 0),
                                         transforms.ToTensor(),

                                         transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                              std=[0.229, 0.224, 0.225]),
                                     ]),
                                     train=True,
                                     args=args)

    p_sampler_train = torch.utils.data.RandomSampler(train_data)  # 数据随机采样
    # p_sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    p_batch_sampler_train = torch.utils.data.BatchSampler(
        p_sampler_train, args.batch_size, drop_last=True)

    train_loader = DataLoader(
        train_data,
        # batch_size=args.batch_size,
        batch_sampler=p_batch_sampler_train,
        # drop_last=False,
        # collate_fn=utils.collate_fn,
        collate_fn=collate_wrapper,
        # sampler=None,
        num_workers=16,
        # prefetch_factor=2,
        pin_memory=True
    )

    # test_loader = torch.utils.data.DataLoader(
    #     test_data,
    #     batch_size=1,
    # )


    def match_name_keywords(n, name_keywords):
        out = False
        for b in name_keywords:
            if b in n:
                out = True
                break
        return out

    param_dicts = [
        {
            "params":
                [p for n, p in model_without_ddp.named_parameters()
                 if (not match_name_keywords(n, args.lr_backbone_names) 
                     and not match_name_keywords(n, args.lr_linear_proj_names) 
                     and p.requires_grad)],
            "lr": args.lr,
        },
        {
            "params": [p for n, p in model_without_ddp.named_parameters() 
                       if (match_name_keywords(n, args.lr_backbone_names) 
                           and not match_name_keywords(n, args.lr_linear_proj_names) 
                           and p.requires_grad)],
            "lr": args.lr_backbone,
        },
        {
            "params": [p for n, p in model_without_ddp.named_parameters() 
                       if match_name_keywords(n, args.lr_linear_proj_names) and p.requires_grad],
            "lr": args.lr * args.lr_linear_proj_mult,
        }
    ]

    
    
    # optimizer = torch.optim.Adam(
    #     [
    #         {'params': model.parameters(), 'lr': args.lr},
    #     ], lr=args.lr, weight_decay=args.weight_decay)
    #
    # lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[args.lr_step], gamma=0.1, last_epoch=-1)
    optimizer = torch.optim.Adam(param_dicts, lr=args.lr)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop)
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], 
                                                          find_unused_parameters=True)
        model_without_ddp = model.module



    if args.frozen_weights is not None:
        checkpoint = torch.load(args.frozen_weights, map_location='cpu')
        model_without_ddp.detr.load_state_dict(checkpoint['model'])

    output_dir = Path(args.output_dir)
    if args.resume:
        if args.resume.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.resume, map_location='cpu')
        missing_keys, unexpected_keys = model_without_ddp.load_state_dict(checkpoint['model'], strict=False)
        unexpected_keys = [k for k in unexpected_keys if not (k.endswith('total_params') or k.endswith('total_ops'))]
        if len(missing_keys) > 0:
            print('Missing Keys: {}'.format(missing_keys))
        if len(unexpected_keys) > 0:
            print('Unexpected Keys: {}'.format(unexpected_keys))
#         if not args.eval and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
#             import copy
#             p_groups = copy.deepcopy(optimizer.param_groups)
#             optimizer.load_state_dict(checkpoint['optimizer'])
#             for pg, pg_old in zip(optimizer.param_groups, p_groups):
#                 pg['lr'] = pg_old['lr']
#                 pg['initial_lr'] = pg_old['initial_lr']
#             print(optimizer.param_groups)
#             lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
#             # todo: this is a hack for doing experiment that resume from checkpoint 
#             # and also modify lr scheduler (e.g., decrease lr in advance).
#             args.override_resumed_lr_drop = True
#             if args.override_resumed_lr_drop:
#                 print('Warning: (hack) args.override_resumed_lr_drop is set to True, '
#                       'so args.lr_drop would override lr_drop in resumed lr_scheduler.')
#                 lr_scheduler.step_size = args.lr_drop
#                 lr_scheduler.base_lrs = list(map(lambda group: group['initial_lr'], optimizer.param_groups))
#             lr_scheduler.step(lr_scheduler.last_epoch)
#             args.start_epoch = checkpoint['epoch'] + 1
        # check the resumed model
        if not args.eval:
            test_stats, coco_evaluator = evaluate(
                model, criterion, postprocessors, data_loader_val, base_ds, device, args
            )
    
    if args.eval:
        print("Start evaluation")
        start_time = time.time()
        test_stats, coco_evaluator = evaluate(model, criterion, postprocessors,
                                              data_loader_val, base_ds, device, args)
        if args.output_dir:
            utils.save_on_master(coco_evaluator.coco_eval["bbox"].eval, output_dir / "eval.pth")
        print_final_result_on_master(model, dataset_val_org, args, test_stats, start_time)
        return

    if utils.is_main_process():
        writer = SummaryWriter(output_dir)
    else:
        writer = None
    total_iter = 0
    
    
    print("Start training")
    start_time = time.time()
    eval_epoch = 0
    
    print('best result:', args.best_pred)
    logger.info('best result = {:.3f}'.format(args.best_pred))
    
    for epoch in range(args.start_epoch, args.epochs):
        # if args.distributed:
        #     sampler_train.set_epoch(epoch)
        # train_stats, total_iter = train_one_epoch(
        #     model, criterion, data_loader_train, optimizer, device, epoch, args.clip_max_norm, writer, total_iter)
        train_stats, total_iter= train_one_epoch(
            model, criterion, train_loader, optimizer, device, epoch, args.clip_max_norm,writer,total_iter,logger,args)
        lr_scheduler.step()
        if args.output_dir:
            checkpoint_paths = [output_dir / 'checkpoint.pth']
            # extra checkpoint before LR drop and every 5 epochs
            if (epoch + 1) % args.lr_drop == 0 or (epoch + 1) % 100 == 0:
                checkpoint_paths.append(output_dir / f'checkpoint{epoch:04}.pth')
            for checkpoint_path in checkpoint_paths:
                utils.save_on_master({
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch,
                    'args': args,
                }, checkpoint_path)
        if  (epoch + 1) % 5 == 0:

            pred_mae, pred_mse, visi = validate(test_data, model,args,logger)

            writer.add_scalar('Metrcis/MAE', pred_mae, eval_epoch)
            writer.add_scalar('Metrcis/MSE', pred_mse, eval_epoch)

            # save_result
#             if args.save:
            is_best = pred_mae < args.best_pred
            args.best_pred = min(pred_mae, args.best_pred)
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.pre,
                'state_dict': model.state_dict(),
                'best_prec1': args.best_pred,
                'optimizer': optimizer.state_dict(),
            }, visi, is_best, args.save_path)

            end = time.time()
            logger.info(
                    'Testing Epoch:[{}/{}]\t mae={:.3f}\t mse={:.3f}\t best_mae={:.3f}\t'.format(
                        args.epochs,
                        epoch,
                        pred_mae, pred_mse,
                        args.best_pred))
        # test_stats, coco_evaluator = evaluate(
        #     model, criterion, postprocessors, data_loader_val, base_ds, device, args
        # )

        # write test status
        # if utils.is_main_process():
        #     writer.add_scalar('test/AP', test_stats['coco_eval_bbox'][0], epoch)
        #     writer.add_scalar('test/AP50', test_stats['coco_eval_bbox'][1], epoch)
        #     writer.add_scalar('test/AP75', test_stats['coco_eval_bbox'][2], epoch)
        #     writer.add_scalar('test/APs', test_stats['coco_eval_bbox'][3], epoch)
        #     writer.add_scalar('test/APm', test_stats['coco_eval_bbox'][4], epoch)
        #     writer.add_scalar('test/APl', test_stats['coco_eval_bbox'][5], epoch)
        #     writer.add_scalar('test/class_error', test_stats['class_error'], epoch)
        #     writer.add_scalar('test/loss', test_stats['loss'], epoch)
        #     writer.add_scalar('test/loss_ce', test_stats['loss_ce'], epoch)
        #     writer.add_scalar('test/loss_bbox', test_stats['loss_bbox'], epoch)
        #     writer.add_scalar('test/loss_giou', test_stats['loss_giou'], epoch)
        #     for key, value in test_stats.items():
        #         if "corr" in key:
        #             writer.add_scalar('test/'+key, value, epoch)
        #
        # log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
        #              **{f'test_{k}': v for k, v in test_stats.items()},
        #              'epoch': epoch}
        #
        # if args.output_dir and utils.is_main_process():
        #     if args.benchmark:
        #         log_stats.update({'params': n_params, 'gflops': gflops, 'fps': fps, 'bfps': bfps})
        #
        #     with (output_dir / "log.txt").open("a") as f:
        #         f.write(json.dumps(log_stats) + "\n")
        #
        #     # for evaluation logs
        #     if coco_evaluator is not None:
        #         (output_dir / 'eval').mkdir(exist_ok=True)
        #         if "bbox" in coco_evaluator.coco_eval:
        #             filenames = ['latest.pth']
        #             if epoch % 50 == 0:
        #                 filenames.append(f'{epoch:03}.pth')
        #             for name in filenames:
        #                 torch.save(coco_evaluator.coco_eval["bbox"].eval,
        #                            output_dir / "eval" / name)
        
    # print_final_result_on_master(model, dataset_val_org, args, test_stats, start_time)
def pre_data(train_list, args, train):
    print("Pre_load dataset ......")
    data_keys = {}
    count = 0
    for j in range(len(train_list)):
        Img_path = train_list[j]
        fname = os.path.basename(Img_path)
        # print(fname)
        img, fidt_map, kpoint = load_data_fidt(Img_path, args, train)

        if min(fidt_map.shape[0], fidt_map.shape[1]) < 256 and train == True:
            # ignore some small resolution images
            continue
#         print(img.size, fidt_map.shape)
        blob = {}
        blob['img'] = img
        blob['kpoint'] = np.array(kpoint)
        blob['fidt_map'] = fidt_map
        blob['fname'] = fname
        data_keys[count] = blob
        count += 1
#     print(data_keys)
#     print(type(data_keys))
    return data_keys

def validate(Pre_data, model, args,logger):
    logger.info('begin test')
    test_loader1 = torch.utils.data.DataLoader(
        dataset.listDataset(Pre_data, args.save_path,
                            shuffle=False,
                            transform=transforms.Compose([
                                transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                            std=[0.229, 0.224, 0.225]),

                            ]),
                            args=args, train=False),
        batch_size=1,
    )
    # test_file = './npydata/jhu_val.npy'
    # with open(test_file, 'rb') as outfile:
    #     test_list = np.load(outfile).tolist()
    # test_data2 = pre_data(test_list, args, train=False)
    #
    # test_loader2 = torch.utils.data.DataLoader(
    #     dataset.listDataset2(test_data2, args.save_path,
    #                         shuffle=False,
    #                         transform=transforms.Compose([
    #                             transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406],
    #                                                                         std=[0.229, 0.224, 0.225]),
    #
    #                         ]),
    #                         args=args, train=False),
    #     batch_size=1,
    # )

    model.eval()

    mae = 0.0
    mse = 0.0
    visi = []
    if not os.path.exists('local_eval/loc_file'):
        os.makedirs('local_eval/loc_file')
        
    
      
    '''output coordinates'''
    f_loc = open("local_eval/A_localization.txt", "w+")
       ###########################################################################
#     print(test_loader2)
#     for i, (fname, img, fidt_map, kpoint) in enumerate(test_loader2):

#         count = 0
# #         img = img.cuda()

#         if len(img.shape) == 5:
#             img = img.squeeze(0)
#         if len(fidt_map.shape) == 5:
#             fidt_map = fidt_map.squeeze(0)
#         if len(img.shape) == 3:
#             img = img.unsqueeze(0)
#         if len(fidt_map.shape) == 3:
#             fidt_map = fidt_map.unsqueeze(0)

#         with torch.no_grad():
#             img = img.cuda()
#             d6 = model(img)
#             print(d6)
# #             print(kpoint)
# #             print(type(d6))
#             count, pred_kpoint, f_loc = LMDS_counting(torch.tensor(d6), i + 1, f_loc, args)
#             point_map = generate_point_map(pred_kpoint, f_loc, rate=1)

            
#         '''return counting and coordinates'''
            

#         out_logits, out_point = outputs['pred_logits'], outputs['pred_points']
#         prob = out_logits.sigmoid()
#         prob = prob.view(1, -1, 2)
#         out_logits = out_logits.view(1, -1, 2)
#         topk_values, topk_indexes = torch.topk(prob.view(out_logits.shape[0], -1),
#                                                kpoint.shape[0] * args.num_queries, dim=1)
        
#         topk_points = topk_indexes // out_logits.shape[2]
        
#         out_point = torch.gather(out_point, 1, topk_points.unsqueeze(-1).repeat(1, 1, 2))
#         out_point = out_point * 256

#         value_points = torch.cat([topk_values.unsqueeze(2), out_point], 2)
        
#         crop_size = 256
#         for i in range(len(value_points)):

#             out_value = value_points[i].squeeze(0)[:, 0].data.cpu().numpy()
#             out_point = value_points[i].squeeze(0)[:, 1:3].data.cpu().numpy().tolist()
#             k = np.zeros((crop_size, crop_size))
#             print(k)
# #             c_map = np.zeros((crop_size, crop_size))

# #             '''get coordinate'''
#             for j in range(len(out_point)):
#                 if out_value[j] < 0.25:
#                     break
#                 x = int(out_point[j][0])
#                 y = int(out_point[j][1])
#                 k[x, y] = 1
        
#         kpoint_list.append(k)
# #         print(kpoint_list)
# # #         confidence_list.append(c_map)
#         pre_kpoint = torch.from_numpy(np.array(kpoint_list)).unsqueeze(0)

        
        
#         count = 0
#         gt_count = torch.sum(kpoint).item()
#         for k in range(topk_values.shape[0]):
#             sub_count = topk_values[k, :]
#             sub_count[sub_count < args.threshold] = 0
#             sub_count[sub_count > 0] = 1
#             sub_count = torch.sum(sub_count).item()
#             count += sub_count
#         mae += abs(count - gt_count)
#         mse += abs(count - gt_count) * abs(count - gt_count)
        
       
    
    
# #         pred_coor = np.nonzero(out_point)
# #         count = len(pred_coor[0])

#         f_loc.write('{} {} '.format(i+1, count))
#         point_map = generate_point_map(pre_kpoint, f_loc, rate=1)  
    
    
    
    ###########################################################################
    
    
    
    
    for i, (fname, img, kpoint, targets, patch_info) in enumerate(test_loader1):
        
        
        
        if len(img.shape) == 5:
            img = img.squeeze(0)
        if len(img.shape) == 3:
            img = img.unsqueeze(0)
        if len(kpoint.shape) == 5:
            kpoint = kpoint.squeeze(0)
        with torch.no_grad():
            img = img.cuda()
            outputs = model(img)
#             print(kpoint)
#             print(type(d6))
#             count, pred_kpoint, f_loc = LMDS_counting(d6, i + 1, f_loc, args)
            
            
        '''return counting and coordinates'''
            

        out_logits, out_point = outputs['pred_logits'], outputs['pred_points']
        prob = out_logits.sigmoid()
        prob = prob.view(1, -1, 2)
        out_logits = out_logits.view(1, -1, 2)
        topk_values, topk_indexes = torch.topk(prob.view(out_logits.shape[0], -1),
                                               kpoint.shape[0] * args.num_queries, dim=1)
        
        # topk_points = topk_indexes // out_logits.shape[2]
        #
        # out_point = torch.gather(out_point, 1, topk_points.unsqueeze(-1).repeat(1, 1, 2))
        # out_point = out_point * 256
        #
        # value_points = torch.cat([topk_values.unsqueeze(2), out_point], 2)
        # print(len(value_points))
        # crop_size = 256
        
#         for i in range(len(value_points)):
# #             print(i)
# #             print(len(value_points))
# #             print(value_points[i])
#             out_value = value_points[i].squeeze(0)[:, 0].data.cpu().numpy()
#             out_point = value_points[i].squeeze(0)[:, 1:3].data.cpu().numpy().tolist()
#             k = np.zeros((crop_size, crop_size))
#             print(k)
# #             c_map = np.zeros((crop_size, crop_size))

            # '''get coordinate'''
            # for j in range(len(out_point)):
            #     if out_value[j] < 0.25:
            #         break
            #     x = int(out_point[j][0])
            #     y = int(out_point[j][1])
            #     k[x, y] = 1
        
#         kpoint_list.append(k)
#         print(kpoint_list)
# #         confidence_list.append(c_map)
#         pre_kpoint = torch.from_numpy(np.array(kpoint_list)).unsqueeze(0)

        
        
        count = 0
        gt_count = torch.sum(kpoint).item()
        for k in range(topk_values.shape[0]):
            sub_count = topk_values[k, :]
            sub_count[sub_count < args.threshold] = 0
            sub_count[sub_count > 0] = 1
            sub_count = torch.sum(sub_count).item()
            count += sub_count
        mae += abs(count - gt_count)
        mse += abs(count - gt_count) * abs(count - gt_count)
        
       
    
    
#         pred_coor = np.nonzero(out_point)
#         count = len(pred_coor[0])

        # f_loc.write('{} {} '.format(i+1, count))
        # point_map = generate_point_map(pre_kpoint, f_loc, rate=1)
        
        

    mae = mae / len(test_loader1)
    mse = math.sqrt(mse / len(test_loader1))

    print('mae', mae, 'mse', mse)
    
    return mae, mse, visi

def print_final_result_on_master(model, dataset_val, args, test_stats, start_time=None):   
    if not utils.is_main_process():
        return False
    
    # training wallclock-time / gpus-hours
    num_gpus = args.world_size if args.distributed else 1
    if start_time is not None:
        total_time = time.time() - start_time
        gpu_hours = total_time / 3600 * num_gpus
        gpu_hours_per_epoch = gpu_hours / args.epochs
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    else:
        total_time_str, gpu_hours, gpu_hours_per_epoch = ["N/A"] * 3

    # make result table
    now = datetime.datetime.now().strftime("%h%d %H:%M")
    tab_keys =  ["Time", "output_dir", "epochs", "bsz", "#GPUs"]
    tab_vals =  [now, Path(args.output_dir), args.epochs, int(args.batch_size * num_gpus), num_gpus]
    tab_keys += ["AP", "AP50", "AP75", "APs", "APm", "APl"]
    tab_vals += [v * 100 for v in test_stats['coco_eval_bbox'][:6]]

    tab_keys += ["E/T", "GPU*hrs", "GPU*hrs/ep"]
    tab_vals += [total_time_str, gpu_hours, gpu_hours_per_epoch]
    
    # add benchmark
    if args.benchmark:
        gflops = compute_gflops(model, dataset_val, approximated=False)
        fps = compute_fps(model, dataset_val, num_iters=300, batch_size=1)
        bfps = compute_fps(model, dataset_val, num_iters=300, batch_size=4)
        tab_keys += ['GFLOPs', 'FPS', 'B4FPS']
        tab_vals += [gflops, fps, bfps]
        
    table = tabulate([tab_vals], headers=tab_keys, tablefmt="pipe",
                     floatfmt=".3f", stralign="center", numalign="center")
    
    # dump to the file
    with open("log_result.txt", "a") as f:
        f.write("\n" + table + "\n")
            
    print(f"Save the final result to ./log_result.txt\n{table}")

def LMDS_counting(input, w_fname, f_loc, args):
    input_max = torch.max(input).item()

  
    keep = nn.functional.max_pool2d(input, (3, 3), stride=1, padding=1)
    keep = (keep == input).float()
    input = keep * input

    '''set the pixel valur of local maxima as 1 for counting'''
    input[input < 100.0 / 255.0 * input_max] = 0
    input[input > 0] = 1

    ''' negative sample'''
    if input_max < 0.1:
        input = input * 0

    count = int(torch.sum(input).item())

    kpoint = input.data.squeeze(0).squeeze(0).cpu().numpy()

    f_loc.write('{} {} '.format(w_fname, count))
    return count, kpoint, f_loc
    
def generate_point_map(kpoint, f_loc, rate=1):
    '''obtain the location coordinates'''
    pred_coor = np.nonzero(kpoint)

    point_map = np.zeros((int(kpoint.shape[0] * rate), int(kpoint.shape[1] * rate), 3), dtype="uint8") + 255  # 22
    # count = len(pred_coor[0])
    coord_list = []
    for i in range(0, len(pred_coor[0])):
        h = int(pred_coor[0][i] * rate)
        w = int(pred_coor[1][i] * rate)
        coord_list.append([w, h])
        cv2.circle(point_map, (w, h), 2, (0, 0, 0), -1)

    for data in coord_list:
        f_loc.write('{} {} '.format(math.floor(data[0]), math.floor(data[1])))
    f_loc.write('\n')

    return point_map


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Sparse DETR training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
