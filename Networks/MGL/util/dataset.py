import os
import os.path
import cv2
import numpy as np
import torch

from torch.utils.data import Dataset


def _find_existing_dir(root, names):
    for name in names:
        path = os.path.join(root, name)
        if os.path.isdir(path):
            return path
    return None


def _resolve_dataset_root(data_root=None):
    if os.path.isdir(data_root):
        image_root = _find_existing_dir(data_root, ['Imgs', 'Image', 'images', 'Images'])
        gt_root = _find_existing_dir(data_root, ['GT', 'Gt', 'gt', 'Mask', 'Masks', 'masks'])
        if image_root is not None and gt_root is not None:
            return image_root, gt_root

    raise RuntimeError(
        "Could not find an MGL dataset folder. Expected a folder containing "
        "'Imgs/' (or 'Image/') and 'GT/' under data_root or data_list.\n"
    )


# RETURN A TUPLE (Image Path, GT Path)
def make_dataset(split='train', data_root=None):
    image_root, gt_root = _resolve_dataset_root(data_root)
    
    images = [image_root + f for f in os.listdir(image_root) if f.endswith('.jpg') or f.endswith('.png')]
    gts = [gt_root + f for f in os.listdir(gt_root) if f.endswith('.png')]

    image_label_list = []

    i = 0
    for i in range(len(images)):
        image_label_list.append((images[i], gts[i]))

    return image_label_list


class SemData(Dataset):
    def __init__(self, split='train', dataset=None, data_root=None, transform=None):
        print(data_root)
        self.split = split
        self.data_list = make_dataset(split, data_root)
        self.transform = transform
        self.name = dataset
        self.kernel = np.ones((5, 5), np.uint8)

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        image_path, label_path = self.data_list[index]
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)  # BGR 3 channel ndarray wiht shape H * W * 3
        if image is None:
            print(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # convert cv2 read image from BGR order to RGB order
        image = np.float32(image)
        label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)  # GRAY 1 channel ndarray with shape H * W

        if image.shape[0] != label.shape[0] or image.shape[1] != label.shape[1]:
            raise (RuntimeError("Image & label shape mismatch: " + image_path + " " + label_path + "\n"))

        edge = cv2.Canny(label, 50, 200) #extract edge from region mask

        #if image.shape[0] != edge.shape[0] or image.shape[1] != edge.shape[1]:
        #    raise (RuntimeError("Image & edge shape mismatch: " + image_path + " " + edge_path + "\n"))

        edge = cv2.dilate(edge, self.kernel, iterations=1)

        '''
        image_name = image_path.split('/')[-1].split('.')[0]
        debug_edge_path = '/raid/workspace/loc_toy/code/semseg/dataset/cam/COD_train/D_Edge/' + image_name + '.png'
        cv2.imwrite(debug_edge_path, edge)
        '''

        if self.transform is not None:
            image, label, edge = self.transform(image, label, edge)

        return  image, label, edge #, image
