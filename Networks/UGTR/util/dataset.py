import os
import os.path
import cv2
import numpy as np
import torch

from torch.utils.data import Dataset


IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm']


def is_image_file(filename):
    filename_lower = filename.lower()
    return any(filename_lower.endswith(extension) for extension in IMG_EXTENSIONS)


def make_dataset(split='train', data_root=None, data_list=None):
    assert split in ['train', 'val', 'test']
    if not os.path.isfile(data_list):
        raise (RuntimeError("Image list file do not exist: " + data_list + "\n"))
    image_label_list = []
    list_read = open(data_list).readlines()
    print("Totally {} samples in {} set.".format(len(list_read), split))
    print("Starting Checking image&label pair {} list...".format(split))
    for line in list_read:
        line = line.strip()
        line_split = line.split(' ')
        if split == 'test':
            if len(line_split) != 1:
                raise (RuntimeError("Image list file read line error : " + line + "\n"))
            image_name = os.path.join(data_root, line_split[0])
            label_name = image_name  # just set place holder for label_name, not for use
            edge_name = image_name
        else:
            if len(line_split) != 3:
                raise (RuntimeError("Image list file read line error : " + line + "\n"))
            image_name = os.path.join(data_root, line_split[0])
            label_name = os.path.join(data_root, line_split[1])
            edge_name = os.path.join(data_root, line_split[2])
        '''
        following check costs some time
        if is_image_file(image_name) and is_image_file(label_name) and os.path.isfile(image_name) and os.path.isfile(label_name):
            item = (image_name, label_name)
            image_label_list.append(item)
        else:
            raise (RuntimeError("Image list file line error : " + line + "\n"))
        '''
        item = (image_name, label_name, edge_name)
        image_label_list.append(item)
    print("Checking image&label pair {} list done!".format(split))
    return image_label_list


class SemData(Dataset):
    def __init__(self, split='train', data_root=None, data_list=None, transform=None):
        print(data_root)
        self.split = split
        self.data_list = make_dataset(split, data_root, data_list)
        self.transform = transform
        self.name = 'ACOD-12K'
        self.kernel = np.ones((5, 5), np.uint8)

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        image_path, label_path, edge_path = self.data_list[index]
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)  # BGR 3 channel ndarray wiht shape H * W * 3
        if image is None:
            print(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # convert cv2 read image from BGR order to RGB order
        image = np.float32(image)
        label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)  # GRAY 1 channel ndarray with shape H * W

        if image.shape[0] != label.shape[0] or image.shape[1] != label.shape[1]:
            raise (RuntimeError("Image & label shape mismatch: " + image_path + " " + label_path + "\n"))

        edge = cv2.imread(edge_path, cv2.IMREAD_GRAYSCALE)
        if edge is None:
            label = cv2.resize(label, (473, 473), interpolation=cv2.INTER_NEAREST)
            edge = cv2.Canny(label, 50, 200) #extract edge from region mask
            cv2.imwrite(edge_path, edge)

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

