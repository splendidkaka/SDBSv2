import argparse
import numpy as np
parser = argparse.ArgumentParser('Deformable DETR Detector', add_help=False)
parser.add_argument('--lr', default=1e-4, type=float)
parser.add_argument('--lr_backbone_names', default=["backbone.0"], type=str, nargs='+')
parser.add_argument('--lr_backbone', default=1e-4, type=float)
parser.add_argument('--lr_linear_proj_names', default=['reference_points', 'sampling_offsets'], type=str, nargs='+')
parser.add_argument('--lr_linear_proj_mult', default=0.1, type=float)
parser.add_argument('--batch_size', default=8, type=int)
parser.add_argument('--weight_decay', default=5 * 1e-4, type=float)
parser.add_argument('--epochs', default=50, type=int)
parser.add_argument('--lr_drop', default=40, type=int)
parser.add_argument('--lr_drop_epochs', default=None, type=int, nargs='+')
parser.add_argument('--clip_max_norm', default=0.1, type=float,
                    help='gradient clipping max norm')

# video demo
parser.add_argument('--video_path', type=str, default='./video_demo/1.mp4',
                    help='input video path ')
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
parser.add_argument('--bbox_loss_coef', default=5, type=float)
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
parser.add_argument('--workers', type=int, default=2,
                    help='load data workers')
parser.add_argument('--photo', default="",
                    help='vis photo')
args = parser.parse_args()
return_args = parser.parse_args()
