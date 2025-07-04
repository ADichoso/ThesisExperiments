
"""
 @Time    : 2021/7/6 09:46
 @Author  : Haiyang Mei
 @E-mail  : mhy666@mail.dlut.edu.cn
 
 @Project : CVPR2021_PFNet
 @File    : config.py
 @Function: Configuration
 
"""
import os

backbone_path = './PFNet/backbone/resnet/resnet50-19c8e357.pth'

datasets_root = './'

cod_training_root = os.path.join(datasets_root, 'train')
acod_training_root = os.path.join(datasets_root, 'ACOD-12K/Train')

chameleon_path = os.path.join(datasets_root, 'test/CHAMELEON')
camo_path = os.path.join(datasets_root, 'test/CAMO')
cod10k_path = os.path.join(datasets_root, 'test/COD10K')
nc4k_path = os.path.join(datasets_root, 'test/NC4K')
