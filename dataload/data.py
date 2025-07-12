#建立起最终的数据集
from os.path import join
from torchvision.transforms import Compose, ToTensor, Normalize, CenterCrop
from dataload.dataset import DatasetFromFolderEval, DatasetFromFolder     #原先的会报错
#from dataset import DatasetFromFolderEval, DatasetFromFolder
# from data_transform import *

def transform():
    return Compose([
        ToTensor(),
        #Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

def get_training_set(data_dir, upscale_factor, patch_size, data_augmentation):  #得到训练的数据集
    hr_dir = join(data_dir, 'GT')   #高分辨率数据集  标签图
    lr_dir = join(data_dir, 'Bicubic')    #这里主要是用这个方法来训练一下
    return DatasetFromFolder(hr_dir, lr_dir, patch_size, upscale_factor, data_augmentation, transform=transform())   #训练的数据集的格式

def get_eval_set(data_dir, upscale_factor):     #得到测试的数据集
    hr_dir = join(data_dir, 'GT')       #标签图
    lr_dir = join(data_dir, 'Bicubic')       #数据图
    return DatasetFromFolderEval(hr_dir, lr_dir, upscale_factor, transform=transform())   #测试的数据集的格式

