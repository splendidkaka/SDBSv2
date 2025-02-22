


from __future__ import division


import os
import warnings
import torch
from config import return_args, args
# torch.cuda.set_device(int(args.gpu_id[0]))
# os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
import torch.nn as nn
from torchvision import transforms
import dataset
import math
from utils import get_root_logger, setup_seed
import nni
from nni.utils import merge_parameter
import time
import util.misc as utils
from utils import save_checkpoint
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter  # add tensoorboard


from models import build_model
def get_args_parser():
    parser = argparse.ArgumentParser('Deformable DETR Detector', add_help=False)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--lr_backbone_names', default=["backbone.0"], type=str, nargs='+')
    parser.add_argument('--lr_backbone', default=1e-4, type=float)
    parser.add_argument('--lr_linear_proj_names', default=['reference_points', 'sampling_offsets'], type=str, nargs='+')
    parser.add_argument('--lr_linear_proj_mult', default=0.1, type=float)
    parser.add_argument('--batch_size', default=8, type=int)
    parser.add_argument('--weight_decay', default=5 * 1e-4, type=float)
    parser.add_argument('--epochs', default=50, type=int)
    parser.add_argument('--lr_drop', default=10, type=int)
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


    parser.add_argument('--dataset', default='nwpu')
    parser.add_argument('--save_path', type=str, default='valid_res',
                        help='save checkpoint directory')
    parser.add_argument('--save_path_log', type=str, default='',
                        help='save checkpoint directory')
    parser.add_argument('--gray_aug', action='store_true',
                        help='using the gray augmentation')
    parser.add_argument('--gray_p', type=float, default=0.1,
                        help='probability of gray')
    parser.add_argument('--crop_size', type=int, default=256,
                        help='crop size for training')
    parser.add_argument('--scale_aug', action='store_true',
                        help='using the scale augmentation')
    parser.add_argument('--scale_p', type=float, default=0.3,
                        help='probability of scaling')
    parser.add_argument('--scale_type', type=int, default=0,
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

warnings.filterwarnings('ignore')
'''fixed random seed '''
setup_seed(args.seed)


def main(args):
    utils.init_distributed_mode(args)
    if args.dataset == 'jhu':
        test_file = './npydata/jhu_test.npy'
    if args.dataset == 'nwpu':
        test_file = './npydata/nwpu_val_2048.npy'
    # test_file = './npydata/nwpu_val.npy'

    # test_file = './npydata/qnrf_test.npy'
    # test_file = './npydata/ShanghaiB_test.npy'
    with open(test_file, 'rb') as outfile:
        test_list = np.load(outfile).tolist()

#     utils.init_distributed_mode(return_args)
    model, criterion, postprocessors = build_model(args)

    model = model.cuda()


    optimizer = torch.optim.Adam(
        [
            {'params': model.parameters(), 'lr': args.lr},
        ], lr=args.lr, weight_decay=args.weight_decay)
    if args.pre:
        if os.path.isfile(args.pre):
            logger.info("=> loading checkpoint '{}'".format(args.pre))
    checkpoint = torch.load(args.pre,map_location=torch.device('cuda:0'))
    model.load_state_dict(checkpoint['state_dict'], strict=False)
    # args.start_epoch = checkpoint['epoch']
    # args.best_pred = checkpoint['best_prec1']
            
    print('best result:', args.best_pred)
    
    torch.set_num_threads(args.num_workers)
    test_data = test_list
    eval_epoch = 0

    pred_mae, pred_mse, visi = validate(test_data, model,args)

    print('Metrcis/MAE'+str(pred_mae))
    print('Metrcis/MSE'+str(pred_mse))

def collate_wrapper(batch):
    targets = []



    imgs = []
    fname = []

    for item in batch:

        if return_args.train_patch:
            fname.append(item[0])

            for i in range(0, len(item[1])):
                imgs.append(item[1][i])

            for i in range(0, len(item[2])):
                targets.append(item[2][i])
        else:
            fname.append(item[0])
            imgs.append(item[1])
            targets.append(item[2])

    return fname, torch.stack(imgs, 0), targets


def validate(Pre_data, model, args):




    test_loader = torch.utils.data.DataLoader(
        dataset.listDataset(Pre_data, args.save_path,
                            shuffle=False,
                            transform=transforms.Compose([
                                transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                            std=[0.229, 0.224, 0.225]),

                            ]),
                            args=args, train=False),
        batch_size=1,
    )

    model.eval()

    mae = 0.0
    mse = 0.0
    visi = []
    # f_loc = open("./local_eval/A_localization.txt", "w+")
    for i, (fname, img, kpoint, targets, patch_info) in enumerate(test_loader):

        if len(img.shape) == 5:
            img = img.squeeze(0)
        if len(img.shape) == 3:
            img = img.unsqueeze(0)
        if len(kpoint.shape) == 5:
            kpoint = kpoint.squeeze(0)

        with torch.no_grad():
#             print(img)
            img = img.cuda()
            outputs = model(img)
            
#             print(type(outputs))
#             count, pred_kpoint, f_loc = LMDS_counting(img, i + 1, f_loc, args)
            
       
        
        
        out_logits, out_point = outputs['pred_logits'], outputs['pred_points']
        prob = out_logits.sigmoid()
        prob = prob.view(1, -1, 2)
        out_logits = out_logits.view(1, -1, 2)
        topk_values, topk_indexes = torch.topk(prob.view(out_logits.shape[0], -1),
                                               kpoint.shape[0] * args.num_queries, dim=1)
        topk_points = topk_indexes // out_logits.shape[2]
#         print(topk_points.unsqueeze(-1).repeat(1, 1, 2))
#         print(out_point)
#         out_point = torch.gather(out_point, 1, topk_points.unsqueeze(-1).repeat(1, 1, 2))
#         out_point = out_point * 256

        value_points = out_point
        #torch.cat([topk_values.unsqueeze(2), out_point], 2)
        
        # kpoint_list = []

#         f_loc = open("local_eval/A_localization.txt", "w+")
#         for i in range(len(value_points)):
#
#             out_value = value_points[i].squeeze(0)[:, 0].data.cpu().numpy()
#             out_point = value_points[i].squeeze(0)[:, 1:3].data.cpu().numpy().tolist()
#             k = np.zeros((256, 256))

        # '''get coordinate'''
        # for j in range(len(out_point)):
        #     if out_value[j] < 0.25:
        #         break
        #     x = int(out_point[j][0])
        #     y = int(out_point[j][1])
        #     k[x, y] = 1
        #
        # kpoint_list.append(k)
        #
        # pred_kpoint = torch.from_numpy(np.array(kpoint_list)).unsqueeze(0)
#         pred_kpoint = pred_kpoint.view(num_h, num_w, crop_size, crop_size).permute(0, 2, 1, 3).contiguous().view(num_h, crop_size,
#                                                                                                        width).view(height,
#                                                                                                                    width).cpu().numpy()
        
        
        
        
        
        
        
        
        count = 0
        gt_count = torch.sum(kpoint).item()
        for k in range(topk_values.shape[0]):
            sub_count = topk_values[k, :]
            sub_count[sub_count < args.threshold] = 0
            sub_count[sub_count > 0] = 1
            sub_count = torch.sum(sub_count).item()
            count += sub_count
        # f_loc.write('{} {} '.format(i+1, count))
        # point_map = generate_point_map(pred_kpoint, f_loc, rate=1)
#         print(str(i+1))
#         print(count)
        mae += abs(count - gt_count)
        mse += abs(count - gt_count) * abs(count - gt_count)

#         if i % 30 == 0:
        print('{fname} Gt {gt:.2f} Pred {pred}'.format(fname=fname[0], gt=gt_count, pred=count))

    mae = mae / len(test_loader)
    mse = math.sqrt(mse / len(test_loader))

    print('mae', mae, 'mse', mse)
    return mae, mse, visi


def LMDS_counting(input, w_fname, f_loc, args):
    input_max = torch.max(input).item()

    ''' find local maxima'''
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
#         cv2.circle(point_map, (w, h), 2, (0, 0, 0), -1)

    for data in coord_list:
        f_loc.write('{} {} '.format(math.floor(data[0]), math.floor(data[1])))
        print("math.floor(data[0])"+str(math.floor(data[0])))
    f_loc.write('\n')

    return point_map
    
          
          


if __name__ == '__main__':
    tuner_params = nni.get_next_parameter()
    params = vars(merge_parameter(return_args, tuner_params))

    main(args)
