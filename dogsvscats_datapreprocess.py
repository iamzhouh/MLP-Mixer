import os
import torch.utils.data as data
from PIL import Image
import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

# 默认输入网络的图片大小
IMAGE_H = 200
IMAGE_W = 200

# 定义一个转换关系，用于将图像数据转换成PyTorch的Tensor形式
data_transform = transforms.Compose([
    transforms.ToTensor()   # 转换成Tensor形式，并且数值归一化到[0.0, 1.0]
])


class DogsVSCatsDataset(data.Dataset):      # 新建一个数据集类，并且需要继承PyTorch中的data.Dataset父类
    def __init__(self, mode, dir):          # 默认构造函数，传入数据集类别（训练或测试），以及数据集路径
        self.mode = mode
        self.list_img = []                  # 新建一个image list，用于存放图片路径，注意是图片路径
        self.list_label = []                # 新建一个label list，用于存放图片对应猫或狗的标签，其中数值0表示猫，1表示狗
        self.data_size = 0                  # 记录数据集大小
        self.transform = data_transform     # 转换关系

        # if self.mode == 'train':            # 训练集模式下，需要提取图片的路径和标签
        #     dir = dir + '/train/'           # 训练集路径在"dir"/train/
        #     for file in os.listdir(dir):    # 遍历dir文件夹
        #         self.list_img.append(dir + file)        # 将图片路径和文件名添加至image list
        #         self.data_size += 1                     # 数据集增1
        #         name = file.split(sep='.')              # 分割文件名，"cat.0.jpg"将分割成"cat","0","jpg"3个元素
        #         # label采用one-hot编码，"1,0"表示猫，"0,1"表示狗，任何情况只有一个位置为"1"，在采用CrossEntropyLoss()计算Loss情况下，label只需要输入"1"的索引，即猫应输入0，狗应输入1
        #         if name[0] == 'cat':
        #             self.list_label.append(0)         # 图片为猫，label为0
        #         else:
        #             self.list_label.append(1)         # 图片为狗，label为1，注意：list_img和list_label中的内容是一一配对的
        # elif self.mode == 'test':           # 测试集模式下，只需要提取图片路径就行
        #     dir = dir + '/test/'            # 测试集路径为"dir"/test/
        #     for file in os.listdir(dir):
        #         self.list_img.append(dir + file)    # 添加图片路径至image list
        #         self.data_size += 1
        #         name = file.split(sep='.')
        #
        #         if name[0] == 'cat':
        #             self.list_label.append(0)
        #         else:
        #             self.list_label.append(1)
        # else:
        #     return print('Undefined Dataset!')
        #

        if self.mode == 'train':            # 训练集模式下，需要提取图片的路径和标签
            dir = dir + '/train/'           # 训练集路径在"dir"/train/
        elif self.mode == 'test':           # 测试集模式下，只需要提取图片路径就行
            dir = dir + '/test/'            # 测试集路径为"dir"/test/
        else:
            return print('Undefined Dataset!')

        for file in os.listdir(dir):
            self.list_img.append(dir + file)    # 添加图片路径至image list
            self.data_size += 1
            name = file.split(sep='.')

            if name[0] == 'cat':
                self.list_label.append(0)
            else:
                self.list_label.append(1)


    def __getitem__(self, item):            # 重载data.Dataset父类方法，获取数据集中数据内容
        if self.mode == 'train':                                        # 训练集模式下需要读取数据集的image和label
            img = Image.open(self.list_img[item])                       # 打开图片
            img = img.resize((IMAGE_H, IMAGE_W))                        # 将图片resize成统一大小
            img = np.array(img)[:, :, :3]                               # 数据转换成numpy数组形式
            label = self.list_label[item]                               # 获取image对应的label
            return self.transform(img), torch.LongTensor([label])       # 将image和label转换成PyTorch形式并返回
        elif self.mode == 'test':                                       # 测试集只需读取image
            img = Image.open(self.list_img[item])
            img = img.resize((IMAGE_H, IMAGE_W))
            img = np.array(img)[:, :, :3]
            label = self.list_label[item]
            return self.transform(img), torch.LongTensor([label])       # 只返回image
        else:
            print('None')

    def __len__(self):
        return self.data_size               # 返回数据集大小



# if __name__ == "__main__":
#     datafile = DogsVSCatsDataset('test', "D:\\迅雷下载\\dogs-vs-cats")
#     dataloader = DataLoader(datafile, batch_size=16, shuffle=True)
#     for images, labels in dataloader:
#         print(images)
#         print(labels)
#         print("="*20)