import os, glob, re
import random

import numpy as np

import PIL
from PIL import Image

import torch
from torch.utils.data import Dataset

import torchvision
from torchvision import transforms

gopro_default_train_path = ''

def get_image_list(dataset_path):
    image_list = []
    for ext in ('jpg', 'jpeg', 'png', 'tif', 'tiff', 'bmp'):
        image_list += sorted(glob.glob(dataset_path + '/*.' + ext))
    return image_list


def read_image(image_path):
    img = Image.open(image_path)
    return torchvision.transforms.functional.to_tensor(img)


def collect_gopro_image_list(dataset_path):
    image_list = []
    for parent_dir in os.listdir(dataset_path):
        blur_directory = os.path.join(dataset_path, parent_dir, 'blur')
        sharp_directory = os.path.join(dataset_path, parent_dir, 'sharp')
        for image_name in sorted(os.listdir(blur_directory)):
            blur_image_path = os.path.join(blur_directory, image_name)
            sharp_image_path = os.path.join(sharp_directory, image_name)
            if os.path.exists(blur_image_path) and os.path.exists(sharp_image_path):
                image_list.append((blur_image_path, sharp_image_path))
    return image_list


class TrainDataset(Dataset):
    def __init__(self, dataset_path=gopro_default_train_path,
                 patch_size=256, normalize=False):
        self.dataset_path = os.path.expanduser(dataset_path)
        self.image_list = collect_gopro_image_list(self.dataset_path)
        self.patch_size = patch_size
        
    def __len__(self):
        return len(self.image_list)
    
    def __getitem__(self, n):
        blur_image_path, sharp_image_path = self.image_list[n]
        blur_image = read_image(blur_image_path)
        sharp_image = read_image(sharp_image_path)
        height, width = blur_image.shape[-2:]
        origin_y = random.randint(0, height - self.patch_size - 1)
        origin_x = random.randint(0, width - self.patch_size - 1)
        blur_image = blur_image[..., origin_y : origin_y + self.patch_size,
                                origin_x : origin_x + self.patch_size]
        sharp_image = sharp_image[..., origin_y : origin_y + self.patch_size,
                                  origin_x : origin_x + self.patch_size]
        return {'Blurred': blur_image, 'Sharp': sharp_image}