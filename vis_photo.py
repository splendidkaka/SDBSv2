from __future__ import division

import math
import os
from PIL import Image
import warnings
from collections import OrderedDict
from config import return_args, args
from scipy.ndimage.filters import gaussian_filter
from torchvision import transforms
from utils import setup_seed
import nni
from nni.utils import merge_parameter
import util.misc as utils
import torch
import numpy as np
import cv2
import torch.nn as nn
from models import build_model

img_transform = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
tensor_transform = transforms.ToTensor()

# create the pre-processing transform
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


warnings.filterwarnings('ignore')
'''fixed random seed '''
setup_seed(args.seed)


def main(args):

    utils.init_distributed_mode(return_args)
    model, criterion, postprocessors = build_model(return_args)
    model = model.cuda()
    device = torch.device('cuda')


if args['pre']:
    print("=> loading checkpoint '{}'".format(args['pre']))
    checkpoint = torch.load(args.pre,map_location='cuda:0')
    model.load_state_dict(checkpoint['state_dict'], strict=False)
    args['start_epoch'] = checkpoint['epoch']
    args['best_pred'] = checkpoint['best_prec1']


    # set your image path here
    img_path = args.photo
    # load the images
    width = 1024
    height = 768


    img_raw2= Image.open(img_path).convert('RGB')
    # round the size
    # width, height = img_raw.size
    # new_width = width // 128 * 128
    # new_height = height // 128 * 128
    # img_raw = img_raw.resize((new_width, new_height), Image.ANTIALIAS)
    # pre-proccessing

    img_raw = cv2.imread(img_path)
    # img_raw = img_raw.resize((width, width), Image.ANTIALIAS)

    imgc = img_raw.copy()

    # new_width = width // 128 * 128
    # new_height = height // 128 * 128
    # img_raw = img_raw.resize((new_width, new_height), Image.ANTIALIAS)
    image = tensor_transform(img_raw)
    image = img_transform(image)

    width, height = image.shape[2], image.shape[1]
    num_w = int(width / 256)
    num_h = int(height / 256)

    image = image.view(3, num_h, 256, width).view(3, num_h, 256, num_w, 256)
    image = image.permute(0, 1, 3, 2, 4).contiguous().view(3, num_w * num_h, 256, 256).permute(1, 0, 2, 3)

    # imgc=image.copy()
    # samples = torch.Tensor(image).unsqueeze(0)
    # samples = samples.to(device)

    # run inference
    with torch.no_grad():
        image = image.to(device)
        outputs = model(image)

        # outputs_scores = torch.nn.functional.softmax(outputs['pred_logits'], -1)[:, :, 1][0]

        out_logits= outputs['pred_logits']
        out_point = outputs['pred_points']
        prob = out_logits.sigmoid()

        # outputs_points = outputs['pred_points'][0]

        topk_values, topk_indexes = torch.topk(prob.view(out_logits.shape[0], -1), 500, dim=1)

        topk_points = topk_indexes // out_logits.shape[2]
        out_point = torch.gather(out_point, 1, topk_points.unsqueeze(-1).repeat(1, 1, 2))
        out_point = out_point * 256

        value_points = torch.cat([topk_values.unsqueeze(2), out_point], 2)
        crop_size = 256
        coord_list= []
        kpoint_map, density_map,  count,coord_list = show_map(value_points, width, height, crop_size, num_h, num_w)



        # cv2.putText(res, "Count:" + str(count), (80, 80), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 5)
        #
        # out.write(res)

        print('count:', count)

        for p in coord_list:

            # n = np.array(image)

            img_to_draw = cv2.cvtColor(np.array(img_raw), cv2.COLOR_RGB2BGR)


            size = 2
            img_to_draw = cv2.circle(img_to_draw, (int(p[0]), int(p[1])), size, (0, 0, 255), -1)

        cv2.imwrite(os.path.join("logs", 'pred{}.jpg'.format(count)), img_to_draw)


    threshold = 0.5








    # filter the predictions
    # points = outputs_points[outputs_scores > threshold].detach().cpu().numpy().tolist()
    # predict_cnt = int((outputs_scores > threshold).sum())

    outputs_scores = torch.nn.functional.softmax(outputs['pred_logits'], -1)[:, :, 1][0]

    outputs_points = outputs['pred_points'][0]
    # draw the predictions



    # save the visualized image

    print(outputs_points)


def show_map(out_pointes, width, height, crop_size, num_h, num_w):
    kpoint_list = []
    confidence_list = []
    f_loc = open("local_eval/A_localization.txt", "w+")
    for i in range(len(out_pointes)):

        out_value = out_pointes[i].squeeze(0)[:, 0].data.cpu().numpy()
        out_point = out_pointes[i].squeeze(0)[:, 1:3].data.cpu().numpy().tolist()
        k = np.zeros((crop_size, crop_size))
        c_map = np.zeros((crop_size, crop_size))

        '''get coordinate'''
        for j in range(len(out_point)):
            if out_value[j] < 0.25:
                break
            x = int(out_point[j][0])
            y = int(out_point[j][1])
            k[x, y] = 1

        kpoint_list.append(k)
        confidence_list.append(c_map)

    kpoint = torch.from_numpy(np.array(kpoint_list)).unsqueeze(0)
    kpoint = kpoint.view(num_h, num_w, crop_size, crop_size).permute(0, 2, 1, 3).contiguous().view(num_h, crop_size,
                                                                                                   width).view(height,
                                                                                                               width).cpu().numpy()
    density_map = gaussian_filter(kpoint.copy(), 6)
    density_map = density_map / np.max(density_map) * 255
    density_map = density_map.astype(np.uint8)
    density_map = cv2.applyColorMap(density_map, 2)

    '''obtain the coordinate '''
    pred_coor = np.nonzero(kpoint)
    count = len(pred_coor[0])
    coord_list = []

    point_map = np.zeros((int(kpoint.shape[0]), int(kpoint.shape[1]), 3), dtype="uint8") + 255  # 22
    for i in range(count):
        w = int(pred_coor[1][i])
        h = int(pred_coor[0][i])

        coord_list.append([w, h])
        # cv2.circle(point_map, (w, h), 3, (0, 0, 0), -1)
        # cv2.circle(frame, (w, h), 3, (0, 255, 50), -1)

    # for data in coord_list:
    #     f_loc.write('{} {} '.format(math.floor(data[0]), math.floor(data[1])))
    # f_loc.write('\n')

    return point_map, density_map, count,coord_list

if __name__ == '__main__':
    tuner_params = nni.get_next_parameter()

    params = vars(merge_parameter(return_args, tuner_params))

    main(params)
