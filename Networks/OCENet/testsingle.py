import torch
import torch.nn.functional as F
import numpy as np
import os, argparse
from scipy import misc
import cv2
from model.ResNet_models import Generator, FCDiscriminator
from data import test_dataset
from PIL import ImageFile
from collections import OrderedDict
ImageFile.LOAD_TRUNCATED_IMAGES = True
import time

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams.update({'figure.figsize':(7,5), 'figure.dpi':100})

parser = argparse.ArgumentParser()
parser.add_argument('--testsize', type=int, default=480, help='testing size')
parser.add_argument('--langevin_step_num_des', type=int, default=10, help='number of langevin steps for ebm')
parser.add_argument('-langevin_step_size_des', type=float, default=0.026,help='step size of EBM langevin')
parser.add_argument('--energy_form', default='identity', help='tanh | sigmoid | identity | softplus')
opt = parser.parse_args()
# Set device to CPU
device = torch.device('cpu')

dataset_path = './Datasets/'

# Create the COD-Network and load the weights
COD_Net = Generator(channel=32).to(device)
COD_state = torch.load('./Checkpoints/OCENet/COD_Model_50.pth', map_location=device)
COD_Net.load_state_dict(COD_state)
COD_Net.eval()

# Create the OCE-Network and load the weights
OCE_Net = FCDiscriminator().to(device)
OCE_state = torch.load('./Checkpoints/OCENet/OCE_Model_50.pth', map_location=device)
OCE_Net.load_state_dict(OCE_state)
OCE_Net.eval()

# 4 COD test datasets
test_datasets = ['ACOD-12K']

# Iterate over the test datasets
for dataset in test_datasets:
    save_path = './Results/OCENet/' + dataset + '/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    confi_path = './Results/OCENet/confidences/' + dataset + '/'
    if not os.path.exists(confi_path):
        os.makedirs(confi_path)

    image_root = dataset_path + dataset + '/Test/Imgs/'
    test_loader = test_dataset(image_root, opt.testsize)

    for i in range(test_loader.size):
        image, HH, WW, name = test_loader.load_data()
        image = image.to(device)

        # Compute COD prediction
        _, COD_pred = COD_Net.forward(image)

        # Compute confidence map prediction
        OCE_input = torch.cat((COD_pred, image), dim=1)
        confi_pred = torch.sigmoid(OCE_Net.forward(OCE_input))
        confi = F.interpolate(confi_pred, size=[WW, HH], mode='bilinear', align_corners=False)
        confi = confi.sigmoid().data.cpu().numpy().squeeze()
        confi = (confi - confi.min()) / (confi.max() - confi.min())
        confi *= 255.0
        confi = confi.astype(np.uint8)
        confi = cv2.applyColorMap(confi, cv2.COLORMAP_JET)

        res = COD_pred
        res = F.upsample(res, size=[WW,HH], mode='bilinear', align_corners=False)
        res = res.sigmoid().data.cpu().numpy().squeeze()
        res *= 255.0
        res = res.astype(np.uint8)

        # Save COD prediction and confidence map prediction
        cv2.imwrite(save_path + name, res)
        cv2.imwrite(confi_path + name, confi)
