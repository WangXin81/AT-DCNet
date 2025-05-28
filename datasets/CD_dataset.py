"""
变化检测数据集
"""

import os
from PIL import Image
import numpy as np
from torch.utils import data
from datasets.data_utils import CDDataAugmentation

"""
CD data set with pixel-level labels；
├─image
├─image_post
└─label
"""
IMG_FOLDER_NAME = "A"
IMG_POST_FOLDER_NAME = 'B'
ANNOT_FOLDER_NAME = "label"

IGNORE = 255

label_suffix = '.png'  # jpg for gan dataset, others : png


def load_img_name_list(dataset_path):
    img_name_list = os.listdir(dataset_path)
    img_name_list = [item for item in img_name_list if item != 'desktop.ini']
    return img_name_list


def get_img_post_path(root_dir, split, img_name):
    return os.path.join(root_dir, split, IMG_POST_FOLDER_NAME, img_name)


def get_img_path(root_dir, split, img_name):
    return os.path.join(root_dir, split, IMG_FOLDER_NAME, img_name)


def get_label_path(root_dir, split, img_name):
    return os.path.join(root_dir, split, ANNOT_FOLDER_NAME, img_name)


class ImageDataset(data.Dataset):
    """VOCdataloder"""

    def __init__(self, root_dir, split='train', img_size=256, is_train=True, to_tensor=True):
        super(ImageDataset, self).__init__()
        self.root_dir = root_dir
        self.img_size = img_size
        self.split = split
        self.list_path = os.path.join(self.root_dir, self.split, IMG_FOLDER_NAME)
        self.img_name_list = load_img_name_list(self.list_path)

        self.A_size = len(self.img_name_list)
        self.to_tensor = to_tensor
        if is_train:
            self.augm = CDDataAugmentation(
                img_size=self.img_size,
                with_random_hflip=True,
                with_random_vflip=True,
                # with_random_rot=True,
                # with_random_crop= True,
                with_scale_random_crop=True,
                with_random_blur=True,
            )
        else:
            self.augm = CDDataAugmentation(
                img_size=self.img_size
            )

    def __getitem__(self, index):
        name = self.img_name_list[index]
        A_path = get_img_path(self.root_dir, self.split, self.img_name_list[index % self.A_size])
        B_path = get_img_post_path(self.root_dir, self.split, self.img_name_list[index % self.A_size])

        img = np.asarray(Image.open(A_path).convert('RGB'))
        img_B = np.asarray(Image.open(B_path).convert('RGB'))

        [img, img_B], _ = self.augm.transform([img, img_B], [], to_tensor=self.to_tensor)

        return {'A': img, 'B': img_B, 'name': name}

    def __len__(self):
        """Return the total number of images in the dataset."""
        return self.A_size


# 一个自定义数据集类，它加载图像和标签数据，应用转换或扩充，并以适合训练模型的字典格式返回它们。
class CDDataset(ImageDataset):

    def __init__(self, root_dir, img_size, split='train', is_train=True, label_transform=None,
                 to_tensor=True):
        super(CDDataset, self).__init__(root_dir, img_size=img_size, split=split, is_train=is_train,
                                        to_tensor=to_tensor)
        self.label_transform = label_transform

    def __getitem__(self, index):
        name = self.img_name_list[index]
        A_path = get_img_path(self.root_dir, self.split, self.img_name_list[index % self.A_size])
        B_path = get_img_post_path(self.root_dir, self.split, self.img_name_list[index % self.A_size])
        img = np.asarray(Image.open(A_path).convert('RGB'))
        img_B = np.asarray(Image.open(B_path).convert('RGB'))
        L_path = get_label_path(self.root_dir, self.split, self.img_name_list[index % self.A_size])
        label = np.array(Image.open(L_path), dtype=np.uint8)

        if self.label_transform == 'norm':
            label = label // 255

        [img, img_B], [label] = self.augm.transform([img, img_B], [label], to_tensor=self.to_tensor)

        return {'name': name, 'A': img, 'B': img_B, 'L': label}
