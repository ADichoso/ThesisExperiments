import os
from PIL import Image
import torch.utils.data as data
import torchvision.transforms as transforms


class PolypDataset(data.Dataset):
    """
    dataloader for polyp segmentation tasks
    """
    def __init__(self, image_root, gt_root, trainsize, zoom_scales=(0.5, 1.0, 1.5)):
        self.trainsize = trainsize
        self.zoom_scales = tuple(zoom_scales)
        self.images = [image_root + f for f in os.listdir(image_root) if f.endswith('.jpg') or f.endswith('.png')]
        self.gts = [gt_root + f for f in os.listdir(gt_root) if f.endswith('.png')]
        self.images = sorted(self.images)
        self.gts = sorted(self.gts)
        self.filter_files()
        self.size = len(self.images) * len(self.zoom_scales)
        self.img_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])])
        self.gt_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor()])

    def __getitem__(self, index):
        base_index = index // len(self.zoom_scales)
        scale = self.zoom_scales[index % len(self.zoom_scales)]
        image = self.rgb_loader(self.images[base_index])
        gt = self.binary_loader(self.gts[base_index])
        image, gt = self.apply_zoom(image, gt, scale)
        image = self.img_transform(image)
        gt = self.gt_transform(gt)
        return image, gt

    def filter_files(self):
        assert len(self.images) == len(self.gts)
        images = []
        gts = []
        for img_path, gt_path in zip(self.images, self.gts):
            img = Image.open(img_path)
            gt = Image.open(gt_path)
            if img.size == gt.size:
                images.append(img_path)
                gts.append(gt_path)
        self.images = images
        self.gts = gts

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            # return img.convert('1')
            return img.convert('L')

    def resize(self, img, gt):
        assert img.size == gt.size
        w, h = img.size
        if h < self.trainsize or w < self.trainsize:
            h = max(h, self.trainsize)
            w = max(w, self.trainsize)
            return img.resize((w, h), Image.BILINEAR), gt.resize((w, h), Image.NEAREST)
        else:
            return img, gt

    def apply_zoom(self, img, gt, scale):
        if scale == 1.0:
            return img, gt

        width, height = img.size
        scaled_width = max(1, int(round(width * scale)))
        scaled_height = max(1, int(round(height * scale)))

        scaled_img = img.resize((scaled_width, scaled_height), Image.BILINEAR)
        scaled_gt = gt.resize((scaled_width, scaled_height), Image.NEAREST)

        if scale > 1.0:
            left = max(0, (scaled_width - width) // 2)
            top = max(0, (scaled_height - height) // 2)
            right = left + width
            bottom = top + height
            return scaled_img.crop((left, top, right, bottom)), scaled_gt.crop((left, top, right, bottom))

        canvas_img = Image.new('RGB', (width, height), (0, 0, 0))
        canvas_gt = Image.new('L', (width, height), 0)
        paste_left = (width - scaled_width) // 2
        paste_top = (height - scaled_height) // 2
        canvas_img.paste(scaled_img, (paste_left, paste_top))
        canvas_gt.paste(scaled_gt, (paste_left, paste_top))
        return canvas_img, canvas_gt

    def __len__(self):
        return self.size


def get_loader(
    image_root,
    gt_root,
    batchsize,
    trainsize,
    shuffle=True,
    num_workers=4,
    pin_memory=True,
    zoom_scales=(0.5, 1.0, 1.5),
):

    dataset = PolypDataset(image_root, gt_root, trainsize, zoom_scales=zoom_scales)
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batchsize,
                                  shuffle=shuffle,
                                  num_workers=num_workers,
                                  pin_memory=pin_memory)
    return data_loader


class test_dataset:
    def __init__(self, image_root, gt_root, testsize):
        self.testsize = testsize
        self.images = [image_root + f for f in os.listdir(image_root) if f.endswith('.jpg') or f.endswith('.png')]
        self.gts = [gt_root + f for f in os.listdir(gt_root) if f.endswith('.tif') or f.endswith('.png')]
        self.images = sorted(self.images)
        self.gts = sorted(self.gts)
        self.transform = transforms.Compose([
            transforms.Resize((self.testsize, self.testsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])])
        self.gt_transform = transforms.ToTensor()
        self.size = len(self.images)
        self.index = 0

    def load_data(self):
        image = self.rgb_loader(self.images[self.index])
        image = self.transform(image).unsqueeze(0)
        gt = self.binary_loader(self.gts[self.index])
        name = self.images[self.index].split('/')[-1]
        if name.endswith('.jpg'):
            name = name.split('.jpg')[0] + '.png'
        self.index += 1
        return image, gt, name

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')
