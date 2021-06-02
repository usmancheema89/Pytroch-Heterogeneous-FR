from __future__ import print_function, division
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torchvision.transforms.functional as TF

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

# plt.ion()   # interactive mode


##
    # def show_landmarks(image, label):
    #     """Show image with landmarks"""
    #     plt.imshow(image)
    #     print(label)
    #     plt.pause(0.001)  # pause a bit so that plots are updated

    # def show_landmarks_batch(sample_batched):
    #     """Show image with landmarks for a batch of samples."""
    #     images_batch, landmarks_batch = \
    #             sample_batched['image'], sample_batched['label']
    #     batch_size = len(images_batch)
    #     im_size = images_batch.size(2)
    #     grid_border_size = 2

    #     grid = utils.make_grid(images_batch)
    #     plt.imshow(grid.numpy().transpose((1, 2, 0)))

    #     for i in range(batch_size):
    #         plt.title('Batch from dataloader')

    # class NormalizetoTensor(object):
    #     def __init__(self, factor):
    #         """
    #         Args:
    #             fac (float): factor to scale the values
    #             data_t (string): data type to change into
    #         """
    #         self.factor = factor
        
    #     def __call__(self, sample):
    #         if isinstance(sample, dict):
    #             image, label = sample['image'], sample['label']
    #             image = np.divide(image,self.factor)
    #             # swap color axis because
    #             # numpy image: H x W x C
    #             # torch image: C X H X W
    #             image = image.transpose((2, 0, 1))
    #             return {'image': torch.from_numpy(image), 'label': torch.tensor(label)}
    #         else:
    #             sample = np.divide(sample,self.factor).transpose((2, 0, 1))
    #             return torch.from_numpy(sample)
        
class FaceLandmarksDataset(Dataset):
    """Face dataset."""

    def __init__(self, info_dic, transform=None):
        """
        Args:
            csv_file (dictionary): Path to the data folder, data_dir, DB, modality.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.images, self.labels = self.__get_data__(info_dic)
        self.mean = np.divide( np.mean(self.images,axis=(0,1,2)), 255)
        self.std = np.divide( np.std(self.images,axis=(0,1,2)),  255)
        self.no_classes = len(np.unique(self.labels)) + 2
        # print(np.unique(self.labels))
        self.transform = transform
        self.normalize = transforms.Compose([transforms.Normalize(mean = self.mean, std = self.std)])

    def __len__(self):
        return len(self.labels)
    
    def __get_data__(self, info_dic):
        labels_p = os.path.join(info_dic['data_dir'], 
                            info_dic['DB']+' Data',
                            info_dic['DB']+ info_dic['subset'] +'Labels.npy')
        labels = np.load(labels_p)
        # print('Loaded Labels: ', labels_p)
        if len(info_dic['modality']) == 1:
            images_p = os.path.join(info_dic['data_dir'], 
                            info_dic['DB']+' Data',
                            info_dic['DB']+info_dic['subset']+ info_dic['modality'][0] + ' Images.npy')
            # print('Loaded single modal data: ', images_p)
            images = np.load(images_p).astype('uint8')
        else:
            images_p_1 = os.path.join(info_dic['data_dir'], 
                            info_dic['DB']+' Data',
                            info_dic['DB']+info_dic['subset']+ info_dic['modality'][0] + ' Images.npy')
            # print('Loaded data_1: ', images_p_1)
            images_1 = np.load(images_p_1).astype('uint8')
            images_p_2 = os.path.join(info_dic['data_dir'], 
                            info_dic['DB']+' Data',
                            info_dic['DB']+info_dic['subset']+ info_dic['modality'][1] + ' Images.npy')
            # print('Loaded data_2: ', images_p_2)
            images_2 = np.load(images_p_2).astype('uint8')
            images = np.concatenate((images_1, images_2), axis=0)
            labels = np.concatenate((labels, labels), axis=0)  
        
        return images, labels

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        image = self.images[idx]
        label = self.labels[idx]

        if self.transform:
            image = self.transform(TF.to_pil_image(image))
            image = self.normalize(image)

        # sample = {'image': image, 'label': torch.tensor(label,dtype=torch.long)}
        # sample = {'image': image, 'label': label}
        return (image, label)#sample

    def get_classes(self):
        return self.no_classes
    


def create_data_loader(info_dic, test = False):
    if test == False:
        transform = transforms.Compose([
                transforms.RandomResizedCrop(245),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor()
                ])
    else:
        transform = transforms.Compose([
                transforms.ToTensor()
                ])
    dataset = FaceLandmarksDataset(info_dic, transform)

    dataloader = DataLoader(dataset, batch_size = info_dic['batch_s'],
                        shuffle = True, num_workers = 0)

    return dataloader, dataset.get_classes()


# for i_batch, sample_batched in enumerate(dataloader):
#     print(i_batch, sample_batched['image'].size(),
#           sample_batched['label'].size())

#     # observe 4th batch and stop.
#     if i_batch == 3:
#         plt.figure()
#         show_landmarks_batch(sample_batched)
#         plt.axis('off')
#         plt.ioff()
#         plt.show()
#         break