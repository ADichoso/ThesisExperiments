import time
import torch
import torchvision
import torch.nn.functional as F
import numpy as np
import os, argparse
import imageio
from scipy import misc
# import cv2
from model.B_TwoStage_P66_20_SC_emb_att import SAM_ResNet
# from model.PFNet import PFNet
from utils.data_val import test_dataset
import cv2

parser = argparse.ArgumentParser()
parser.add_argument('--testsize', type=int, default=704, help='testing size')
parser.add_argument('--pth_path', type=str, default='./Checkpoints/PreyNet_w_PVT/Net_epoch_best.pth')
opt = parser.parse_args()

for _data_name in ['ACOD-12K']:
    # 'CHAMELEON', 'CAMO', 'COD10K', 'NC4K'
    data_path = './Datasets/{}/Test'.format(_data_name)
    save_path = './Results/PreyNet_w_PVT/{}/'.format(_data_name)
    model = SAM_ResNet()
    # Set cuda device to cpu if no gpu is available
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
        model.load_state_dict(torch.load(opt.pth_path))  # as-is
    else:
        device = torch.device('cpu')
        model.load_state_dict(torch.load(opt.pth_path, map_location=device))  # fix

    model = model.to(device)
    model.eval()

    os.makedirs(save_path, exist_ok=True)
    image_root = '{}/Imgs/'.format(data_path)
    gt_root = '{}/GT/'.format(data_path)
    test_loader = test_dataset(image_root, gt_root, opt.testsize)

    inference_times = [-1 for i in range(test_loader.size)]
    for i in range(test_loader.size):
        image, gt, name, _ = test_loader.load_data()
        gt = np.asarray(gt, np.float32)
        gt /= (gt.max() + 1e-8)
        image = image.to(device)

        start_time = time.time()
        res2, _, _, _, _, _, _, _, _ = model(image)
        end_time = time.time()

        res = res2
        res = F.upsample(res, size=(gt.shape[1], gt.shape[2]), mode='bilinear', align_corners=False)
        res = res.sigmoid().data.cpu().numpy().squeeze()
        res = (res - res.min()) / (res.max() - res.min() + 1e-8)

        #Get inference time (seconds / 1 frame)
        inference_times[i] = end_time - start_time
        print('> {} - {}: {} fps'.format(_data_name, name, 1 / inference_times[i]))

        res_uint8 = (res * 255).astype(np.uint8)
        imageio.imwrite(save_path+name, res_uint8)

    test_fps = [1 / time for time in inference_times]

    print("AVERAGE FPS:", sum(test_fps) / len(test_fps))