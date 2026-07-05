import torch
import FSPNet_model
import dataset
import os
from torch.utils.data import DataLoader
import numpy as np
import torch.nn.functional as F
from imageio import imwrite
import time
if __name__ =='__main__':
    batch_size = 1
    net = FSPNet_model.Model(None, img_size=384).cuda()
    Dirs=[#"./Datasets/ACOD-12K/Test/",
          #"./Datasets/Papple_RGB-D-Size/Test/",
          "./Datasets/Sweet_Pepper/Test/"]


    for dataset_name in ['Sweet_Pepper']: #'ACOD-12K', 'PApple_RGB-D-Size',
        result_save_root="./Results/FSPNet/" + dataset_name + "/"
        # pretrained_dict = torch.load("./ckpt/"+m)['model']
        ckpt_root="./Checkpoints/FSPNet/" + dataset_name + "/"
        ckpt_file="fspnet_epoch98_mainloss0.00804.pth"
        checkpoint = torch.load(ckpt_root + ckpt_file)
        state_dict = checkpoint["model"]

        net.load_state_dict(state_dict)
        net.eval()
        for i in range(len(Dirs)):
            Dir = Dirs[i]
            if not os.path.exists(result_save_root):
                os.mkdir(result_save_root)
            if not os.path.exists(os.path.join(result_save_root, Dir.split("/")[-1])):
                os.mkdir(os.path.join(result_save_root, Dir.split("/")[-1]))
            Dataset = dataset.TestDataset(Dir, 384)
            dataloader = DataLoader(Dataset, batch_size=batch_size, num_workers=batch_size*2)
            count=0
            for data in dataloader:
                count+=1
                img, label = data['img'].cuda(), data['label'].cuda()
                name = data['name'][0].split("/")[-1]
                with torch.no_grad():
                    out = net(img)[3]
                    # out = net(img)
                B,C,H,W = label.size()
                o = F.interpolate(out, (H,W), mode='bilinear', align_corners=True).detach().cpu().numpy()[0,0]
                o =(o-o.min())/(o.max()-o.min()+1e-8)
                o = (o*255).astype(np.uint8)
                imwrite(result_save_root+Dir.split("/")[-1]+"/"+name, o)
    
    print("Test finished!")


