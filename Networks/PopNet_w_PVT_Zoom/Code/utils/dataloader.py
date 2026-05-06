from torch.utils import data
import torch
import os
from PIL import Image
class EvalDataset(data.Dataset):
    def __init__(self, img_root, label_root, zoom_scales=(0.5, 1.0, 1.5)):
        self.zoom_scales = tuple(zoom_scales)
        self.image_path = list(map(lambda x: os.path.join(img_root, x), sorted(os.listdir(img_root))))
        self.label_path = list(map(lambda x: os.path.join(label_root, x), sorted(os.listdir(label_root))))

    def __getitem__(self, item):
        pred = Image.open(self.image_path[item]).convert('L')

        gt = Image.open(self.label_path[item]).convert('L')
        # print(self.image_path[item], self.label_path[item])
        if pred.size != gt.size:
            pred = pred.resize(gt.size, Image.BILINEAR)
        return pred, gt

    def __getitem__(self, index):
        base_index = index // len(self.zoom_scales)
        image = Image.open(self.image_path[base_index]).convert('L')
        gt = Image.open(self.label_path[base_index]).convert('L')
        scale = self.zoom_scales[index % len(self.zoom_scales)]
        image, gt = self.apply_zoom(image, gt, scale)
        
        return image, gt
    
    def __len__(self):
        return len(self.image_path)
    
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