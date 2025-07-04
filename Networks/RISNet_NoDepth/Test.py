import argparse
import os

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import time, datetime
from model.RISNet import RISNet
from utils.dataloader import test_dataset


parser = argparse.ArgumentParser()
parser.add_argument('--testsize', type=int, default=448, help='testing size default 704')
parser.add_argument('--pth_path', type=str, default='./Networks/RISNet/ckpt45_RISNet.pth', help='path to load your model checkpoint')
parser.add_argument('--test_path', type=str, default='./Datasets/', help='path to test dataset')
opt = parser.parse_args()

for _data_name in ['ACOD-12K']:
    data_path = opt.test_path + '{}/Test/'.format(_data_name)
    save_path = './results/{}/'.format(_data_name)
    model = RISNet()
    model.load_state_dict(torch.load(opt.pth_path, map_location="cuda:0"))
    model.cuda()
    model.eval()

    os.makedirs(save_path, exist_ok=True)
    image_root = '{}/Imgs/'.format(data_path)
    gt_root = '{}/GT/'.format(data_path)
    print('root', image_root, gt_root)
    test_loader = test_dataset(image_root, gt_root, opt.testsize)
    print('****', test_loader.size)
    start = time.time()
    for i in range(test_loader.size):
        image, gt, name = test_loader.load_data()
        print('***name', name)
        gt = np.asarray(gt, np.float32)
        gt /= (gt.max() + 1e-8)
        image = image.cuda()
        x_in = image

        P1, P2 = model(x_in)
        res = F.upsample(P1[-1]+P2, size=gt.shape, mode='bilinear', align_corners=False)
        res = res.sigmoid().data.cpu().numpy().squeeze()
        res = (res - res.min()) / (res.max() - res.min() + 1e-8)
        res = (res * 255).astype(np.uint8)
        print('> {} - {}'.format(_data_name, name))
        cv2.imwrite(save_path+name, res)
    
    end = time.time()
    print("Total Testing Time: {}".format(str(datetime.timedelta(seconds=int(end - start)))))
