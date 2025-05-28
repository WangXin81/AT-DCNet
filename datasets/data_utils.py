import random
import numpy as np
from PIL import Image
from PIL import ImageFilter
import torchvision.transforms.functional as TF
from torchvision import transforms
import torch


class CDDataAugmentation:
    def __init__(
            self,
            img_size,
            with_random_hflip=False,  # 是否应用随机水平翻转
            with_random_vflip=False,  # 是否应用随机垂直翻转
            with_random_rot=False,  # 是否应用随机旋转
            with_random_crop=False,  # 是否应用随机裁剪
            with_scale_random_crop=False,  # 是否先应用缩放再随机裁剪
            with_random_blur=False,  # 是否应用随机模糊
    ):
        self.img_size = img_size
        if self.img_size is None:
            self.img_size_dynamic = True
        else:
            self.img_size_dynamic = False
        self.with_random_hflip = with_random_hflip
        self.with_random_vflip = with_random_vflip
        self.with_random_rot = with_random_rot
        self.with_random_crop = with_random_crop
        self.with_scale_random_crop = with_scale_random_crop
        self.with_random_blur = with_random_blur

    def transform(self, imgs, labels, to_tensor=True):
        imgs = [TF.to_pil_image(img) for img in imgs]
        if self.img_size is None:
            self.img_size = None

        if not self.img_size_dynamic:
            if imgs[0].size != (self.img_size, self.img_size):
                imgs = [TF.resize(img, [self.img_size, self.img_size], interpolation=TF.InterpolationMode.BICUBIC)
                        for img in imgs]
        else:
            self.img_size = imgs[0].size[0]

        labels = [TF.to_pil_image(img) for img in labels]
        if len(labels) != 0:
            if labels[0].size != (self.img_size, self.img_size):
                labels = [TF.resize(img, [self.img_size, self.img_size], interpolation=TF.InterpolationMode.NEAREST)
                          for img in labels]

        random_base = 0.5
        if self.with_random_hflip and random.random() > 0.5:  # 随机水平翻转
            imgs = [TF.hflip(img) for img in imgs]
            labels = [TF.hflip(img) for img in labels]

        if self.with_random_vflip and random.random() > 0.5:  # 随机垂直翻转
            imgs = [TF.vflip(img) for img in imgs]
            labels = [TF.vflip(img) for img in labels]

        if self.with_random_rot and random.random() > random_base:  # 随机旋转
            angles = [90, 180, 270]
            index = random.randint(0, 2)
            angle = angles[index]
            imgs = [TF.rotate(img, angle) for img in imgs]
            labels = [TF.rotate(img, angle) for img in labels]

        if self.with_random_crop and random.random() > random_base:  # 随机裁剪
            i, j, h, w = transforms.RandomResizedCrop(size=self.img_size). \
                get_params(img=imgs[0], scale=[0.8, 1.0], ratio=[1, 1])  # 生成随机的裁剪参数
            imgs = [TF.resized_crop(img, i, j, h, w,
                                    size=[self.img_size, self.img_size],
                                    interpolation=TF.InterpolationMode.BICUBIC)
                    for img in imgs]

            labels = [TF.resized_crop(img, i, j, h, w,
                                      size=[self.img_size, self.img_size],
                                      interpolation=TF.InterpolationMode.NEAREST)
                      for img in labels]

        if self.with_scale_random_crop:  # 先应用缩放再随机裁剪
            # rescale
            scale_range = [1, 1.2]
            target_scale = scale_range[0] + random.random() * (scale_range[1] - scale_range[0])
            imgs = [pil_rescale(img, target_scale, order=3) for img in imgs]
            labels = [pil_rescale(img, target_scale, order=0) for img in labels]
            # crop
            imgsize = imgs[0].size  # h, w
            box = get_random_crop_box(imgsize=imgsize, cropsize=self.img_size)
            imgs = [pil_crop(img, box, cropsize=self.img_size, default_value=0)
                    for img in imgs]
            labels = [pil_crop(img, box, cropsize=self.img_size, default_value=255)
                      for img in labels]

        if self.with_random_blur and random.random() > 0:  # 随机模糊
            radius = random.random()
            # imgs = [TF.to_pil_image(img) for img in imgs if not isinstance(img, Image.Image)]
            imgs = [img.filter(ImageFilter.GaussianBlur(radius=radius))
                    for img in imgs]

        if to_tensor:
            # to tensor
            imgs = [TF.to_tensor(img) for img in imgs]
            labels = [torch.from_numpy(np.array(img, np.uint8)).unsqueeze(dim=0)
                      for img in labels]
            # img = TF.normalize(imgs[0], mean=[0.47538333, 0.47538333, 0.47538333],
            #                     std=[0.17277557772262636, 0.17277557772262636, 0.17277557772262636])
            # img_B = TF.normalize(imgs[1], mean=[0.43854199, 0.43854199, 0.43854199],
            #                       std=[0.13903312632810402, 0.13903312632810402, 0.13903312632810402])
            # imgs = [img, img_B]
            imgs = [TF.normalize(img, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
                    for img in imgs]

        return imgs, labels


# 该函数使用 PIL 从输入图像中裁剪区域，并将裁剪区域作为新Image对象返回
# 如果裁剪区域超出图像边界，则使用可选的填充。
def pil_crop(image, box, cropsize, default_value):
    assert isinstance(image, Image.Image)
    if not isinstance(image, Image.Image):
        image = [TF.to_pil_image(image)]
    img = np.array(image)
    if len(img.shape) == 3:
        cont = np.ones((cropsize, cropsize, img.shape[2]), img.dtype) * default_value
    else:
        cont = np.ones((cropsize, cropsize), img.dtype) * default_value
    cont[box[0]:box[1], box[2]:box[3]] = img[box[4]:box[5], box[6]:box[7]]

    return Image.fromarray(cont)


# 该函数在给定图像大小和所需裁剪大小的情况下为图像生成随机裁剪坐标。
# imgsize表示格式中图像的大小(height, width)，并且cropsize是裁剪的所需大小（假设为正方形）。
def get_random_crop_box(imgsize, cropsize):
    h, w = imgsize
    ch = min(cropsize, h)
    cw = min(cropsize, w)

    w_space = w - cropsize
    h_space = h - cropsize
    # 此代码块处理沿宽度随机生成裁剪坐标。如果为正w_space（即图像宽度大于所需的裁剪宽度），则cont_left（
    # 裁剪中的左坐标）设置为 0，并且img_left（图像中的左坐标）在 0 和 w_space之间随机选择。否则，如果w_space
    # 为非正数，表示图像宽度小于或等于所需的裁剪宽度，cont_left在 0 和 -w_space之间随机选择，并img_left设置为 0。
    if w_space > 0:
        cont_left = 0
        img_left = random.randrange(w_space + 1)
    else:
        cont_left = random.randrange(-w_space + 1)
        img_left = 0
    # 此代码块处理沿高度随机生成裁剪坐标。与上一个块类似
    if h_space > 0:
        cont_top = 0
        img_top = random.randrange(h_space + 1)
    else:
        cont_top = random.randrange(-h_space + 1)
        img_top = 0

    return cont_top, cont_top + ch, cont_left, cont_left + cw, img_top, img_top + ch, img_left, img_left + cw


# 该函数使用 Python 图像库 (PIL) 重新缩放图像。该img参数表示要重新缩放的输入图像。
# 该scale参数确定调整图像大小的比例因子。该order参数指定在调整大小时使用的插值顺序。
def pil_rescale(img, scale, order):
    assert isinstance(img, Image.Image)
    height, width = img.size  # 提取图像高度宽度等信息
    target_size = (int(np.round(height * scale)), int(np.round(width * scale)))  # 计算重新缩放图像的目标大小
    return pil_resize(img, target_size, order)


# 该函数使用 Python 图像库 (PIL) 调整图像大小。该img参数表示要调整大小的输入图像。
# 该size参数以格式指定调整大小的目标大小(width, height)。该order参数确定在调整大小时使用的插值方法。
def pil_resize(img, size, order):
    assert isinstance(img, Image.Image)
    if size[0] == img.size[0] and size[1] == img.size[1]:  # 检查目标大小是否与图像的当前大小 相同。如果相等，则意味着不需要调整大小，因此返回原始图像。
        return img
    # 将order参数映射到相应的 PIL 重采样滤波器。如果order是 3，它将Image.BICUBIC过滤器分配给变量resample，
    # 表示双三次插值。如果order为 0，则将Image.NEAREST过滤器分配给resample，表示最近邻插值。
    if order == 3:
        resample = Image.Resampling.BICUBIC
    elif order == 0:
        resample = Image.Resampling.NEAREST
    else:
        resample = None
    return img.resize(size[::-1], resample)
