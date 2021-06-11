from __future__ import print_function, division
import os, torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchvision.transforms.functional as TF

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

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

class SiameseDataset(Dataset):
    def __init__(self, info_dic, transform=None):
        self.images, self.labels = self.__get_data__(info_dic)
        self.mean = np.divide( np.mean(self.images,axis=(0,1,2)), 255)
        self.std = np.divide( np.std(self.images,axis=(0,1,2)),  255)
        self.no_classes = len(np.unique(self.labels)) + 2
        self.transform = transform
        self.normalize = transforms.Compose([transforms.Normalize(mean = self.mean, std = self.std)])
        self.indexes = self.create_indexes() 
    
    def create_indexes(self):
        indexes = []
        no_images = len(self.labels)
        for i in range(0,no_images):
            for j in range(i,no_images):
                indexes.append([i,j])
        indexes = np.array(indexes,np.int16)
        np.random.shuffle(indexes)

        return indexes

    def __len__(self):
        return len(self.indexes)

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

    def fetch_item(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        image = self.images[int(idx)]
        label = self.labels[int(idx)]

        if self.transform:
            image = self.transform(TF.to_pil_image(image))
            image = self.normalize(image)

        return image, label

    def __getitem__(self, idx):
        image = [None] *2
        label = [None] *2
        image[0], label[0] = self.fetch_item( self.indexes[idx,0])
        image[1], label[1] = self.fetch_item( self.indexes[idx,1])
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

def get_siamese_data_loader(info_dic, test = False):
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
    dataset = SiameseDataset(info_dic, transform)

    dataloader = DataLoader(dataset, batch_size = info_dic['batch_s'],
                        shuffle = True, num_workers = 0)

    return dataloader, dataset.get_classes()


def create_info_dic(model, loss, db, modality, Cmnt):
    # model, loss, db, modality
    info_dic = dict()
    info_dic['epochs'] = 300
    info_dic['batch_s'] = 112
    info_dic['data_dir'] = r'E:\Work\Cross Modality FR 2\Numpy Data'
    info_dic['subset'] = ' Train '
    
    info_dic['model'] = model
    info_dic['loss'] = loss

    info_dic['DB'] = db
    info_dic['modality'] = modality
    mods = '-'.join(modality)
    info_dic['run_name'] = f'{db}_{mods}_{model}_{loss}_{Cmnt}'
    
    return info_dic

# info_dic = create_info_dic('SiamCMDCIND','Triplet','SMMD',['Vis','IRR'], 'DataLoaderTest')

# data_loader, classes =  get_siamese_data_loader(info_dic)

# for i, data in enumerate(data_loader, 0):
#     data, lbls = data
#     print(lbls[0].data)
#     print(lbls[1].data)