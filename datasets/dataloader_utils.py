import numpy as np
import torch
from torch.utils.data import DataLoader
from datasets import data_config
from datasets.CD_dataset import CDDataset
from torchvision import utils
from PIL import Image, ImageDraw


def get_loader(data_name, img_size=256, batch_size=8, split='test', is_train=False, dataset='CDDataset'):
    dataConfig = data_config.DataConfig().get_data_config(data_name)
    root_dir = dataConfig.root_dir
    label_transform = dataConfig.label_transform

    if dataset == 'CDDataset':
        data_set = CDDataset(root_dir=root_dir, split=split,
                             img_size=img_size, is_train=is_train,
                             label_transform=label_transform)
    else:
        raise NotImplementedError(
            'Wrong dataset name %s (choose one from [CDDataset])'
            % dataset)

    shuffle = is_train
    dataloader = DataLoader(data_set, batch_size=batch_size,
                            shuffle=shuffle, num_workers=4)

    return dataloader


def get_loaders(args):
    data_name = args.data_name
    dataConfig = data_config.DataConfig().get_data_config(data_name)
    root_dir = dataConfig.root_dir
    label_transform = dataConfig.label_transform
    split = args.split
    split_val = 'val'
    if hasattr(args, 'split_val'):
        split_val = args.split_val
    if args.dataset == 'CDDataset':
        training_set = CDDataset(root_dir=root_dir, split=split,
                                 img_size=args.img_size, is_train=True,
                                 label_transform=label_transform)
        val_set = CDDataset(root_dir=root_dir, split=split_val,
                            img_size=args.img_size, is_train=False,
                            label_transform=label_transform)
    else:
        raise NotImplementedError(
            'Wrong dataset name %s (choose one from [CDDataset,])'
            % args.dataset)

    datasets = {'train': training_set, 'val': val_set}
    dataloaders = {x: DataLoader(datasets[x], batch_size=args.batch_size,
                                 shuffle=True, num_workers=args.num_workers)
                   for x in ['train', 'val']}

    return dataloaders


def make_numpy_grid(tensor_data, pad_value=0, padding=0):
    tensor_data = tensor_data.detach()
    vis = utils.make_grid(tensor_data, pad_value=pad_value, padding=padding)
    vis = np.array(vis.cpu()).transpose((1, 2, 0))
    if vis.shape[2] == 1:
        vis = np.stack([vis, vis, vis], axis=-1)
    return vis


def create_numpy_grid_image(image_data):
    # 将张量数据转换为 NumPy 数组
    image_data = image_data.detach()
    image_array = image_data.cpu().detach().numpy()

    # 获取图像的数量、通道数、高度和宽度
    num_images, _, height, width = image_array.shape

    # 计算网格的大小
    grid_width = width
    grid_height = height

    # 创建新图像
    grid_image = np.ones((num_images * height, width,1), dtype=np.uint8) * 255

    # 将图像数据填充到网格中
    for i in range(num_images):
        start_row = i * height
        end_row = start_row + height
        grid_image[start_row:end_row, :, 0] = image_array[i, 0, :, :]

    # 创建 PIL 图像对象
    pil_image = Image.fromarray(grid_image)

    # 绘制网格线
    draw = ImageDraw.Draw(pil_image)
    for x in range(0, width, grid_width):
        draw.line([(x, 0), (x, height * num_images)], fill="black", width=2)
    for y in range(0, height * num_images, grid_height):
        draw.line([(0, y), (width, y)], fill="black", width=2)

    # pil_image = np.array(pil_image)
    # pil_image = pil_image[:, :, np.newaxis]

    return pil_image


def de_norm(tensor_data):
    return tensor_data * 0.5 + 0.5


def get_device(args):
    # set gpu ids
    str_ids = args.gpu_ids.split(',')
    gpu_ids = []
    for str_id in str_ids:
        Id = int(str_id)
        if Id >= 0:
            gpu_ids.append(Id)
    if len(gpu_ids) > 0:
        torch.cuda.set_device(gpu_ids[0])
