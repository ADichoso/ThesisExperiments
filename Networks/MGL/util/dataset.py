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


def _find_existing_dir(root, names):
    for name in names:
        path = os.path.join(root, name)
        if os.path.isdir(path):
            return path
    return None


def _list_images(root):
    return sorted(
        os.path.join(root, f)
        for f in os.listdir(root)
        if is_image_file(f)
    )

def _stem(path):
    return os.path.splitext(os.path.basename(path))[0]


def _resolve_dataset_root(split, data_root=None, data_list=None):
    split_dir_names = [split, split.capitalize()]
    if split == 'val':
        split_dir_names.extend(['Val', 'Valid', 'Validation', 'Test'])

    candidates = []
    for root in [data_list, data_root]:
        if root is None:
            continue
        if os.path.isfile(root):
            raise RuntimeError("MGL now builds datasets from folders, not .lst files: " + root + "\n")
        candidates.append(root)
        for split_dir_name in split_dir_names:
            candidates.append(os.path.join(root, split_dir_name))

    for root in candidates:
        if os.path.isdir(root):
            image_root = _find_existing_dir(root, ['Imgs', 'Image', 'images', 'Images'])
            gt_root = _find_existing_dir(root, ['GT', 'Gt', 'gt', 'Mask', 'Masks', 'masks'])
            if image_root is not None and (gt_root is not None or split == 'test'):
                return image_root, gt_root

    raise RuntimeError(
        "Could not find an MGL dataset folder. Expected a folder containing "
        "'Imgs/' (or 'Image/') and 'GT/' under data_root or data_list.\n"
    )


def make_dataset(split='train', data_root=None, data_list=None):
    assert split in ['train', 'val', 'test']
    image_root, gt_root = _resolve_dataset_root(split, data_root, data_list)
    images = _list_images(image_root)
    if not images:
        raise RuntimeError("No image files found in: " + image_root + "\n")

    image_label_list = []
    if gt_root is None:
        for image_name in images:
            image_label_list.append((image_name, image_name, None))
        print("Totally {} samples in {} set.".format(len(image_label_list), split))
        return image_label_list

    gts = _list_images(gt_root)
    gt_by_stem = {_stem(path): path for path in gts}

    for image_name in images:
        name = _stem(image_name)
        if name not in gt_by_stem:
            raise RuntimeError("Missing GT for image: " + image_name + "\n")
        label_name = gt_by_stem[name]
        image_label_list.append((image_name, label_name, None))

    print("Totally {} samples in {} set.".format(len(image_label_list), split))
    print("Image root: {}".format(image_root))
    print("GT root: {}".format(gt_root))
    return image_label_list


class SemData(Dataset):
    def __init__(self, split='train', data_root=None, data_list=None, transform=None):
        print(data_root)
        self.split = split
        self.data_list = make_dataset(split, data_root, data_list)
        self.transform = transform
        self.name = 'COD10K'
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
