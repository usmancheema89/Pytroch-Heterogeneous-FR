import json, cv2, os
import numpy as np
from shutil import copyfile

from skimage.transform import resize
from skimage.util import img_as_ubyte
import matplotlib.pyplot as plt

train_vis_images = []
train_the_images = []
train_labels = []
test_vis_images = []
test_the_images = []
test_labels = []

def read_file(pairs_file):

    for key in pairs_file:
        subject = key.split('_')[0]
        # if int(key.split('_')[-1]) <= 3:
        vis_img = get_img(pairs_file[key][1])
        the_img = get_img(pairs_file[key][2])
        if (vis_img is not None) and (the_img is not None):
            label = pairs_file[key][0]
            train_vis_images.append(vis_img)
            train_the_images.append(the_img)
            train_labels.append(label)

def get_label(path):
    label = int(path.split('\\')[4])
    return label

def get_img(img_path):

    img = cv2.imread(img_path,cv2.IMREAD_UNCHANGED) #uint8
    # cv2.imshow('img',img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    if img is None:
        print('image not found: ', img_path)
        return None
    else:
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = crop_img(img)
        img = cv2.resize(img,(224,224), interpolation = cv2.INTER_AREA)
        if len(img.shape) < 3:
            img = cv2.cvtColor(img,cv2.COLOR_GRAY2RGB)
        # cv2.imshow('img',img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        img = np.array(img,np.float64)


    
    return img

def crop_img(img):

    half = (img.shape[0]-img.shape[1])/2
    (h,w) = img.shape[:2]
    topx, topy = int(half), int(0)
    botx, boty = int(h-half), int(w)
    img_crop = img[topx:botx, topy:boty]

    return img_crop

def resize_img(img):
    if img.shape[0] <= img.shape[1]:
        if img.shape[0] < 256:
            diff = 256 - img.shape[0]
            mag = (diff/img.shape[0]) + 1
            shape1 = int(img.shape[1] * mag)
            dim = (shape1,256)
            img = cv2.resize(img,dim, interpolation = cv2.INTER_AREA)
    elif img.shape[1] < img.shape[0]:
        if img.shape[1] < 256:
            diff = 256 - img.shape[1]
            mag = (diff/img.shap[1]) + 1
            shape0 = int(img.shape[0] * mag)
            dim = (256, shape0)
            img = cv2.resize(img,dim, interpolation = cv2.INTER_AREA)



    return img

def read_Json(file_path):
    with open(file_path) as data:
        files_dic = json.load(data)                     # files_dic key = subjectname_imagename_i.bmp, dict[key][0] = file path, dict[key][1] = subject id 

    return files_dic

def write_numpy(data, name):

    data_array = np.array(data)
    np.save(name,data_array)

data_type = 'Test' 
pairs_file_name = f'SURF {data_type} Img Pairs.txt'
pairs_file = read_Json(pairs_file_name)
read_file(pairs_file)

write_numpy(train_labels,f'CASIA {data_type} Labels.npy')
write_numpy(train_vis_images,f'CASIA {data_type} Vis Images.npy')
write_numpy(train_the_images,f'CASIA {data_type} IRR Images.npy')
# write_numpy(test_labels,'USTC CMFR Test Labels.npy')
# write_numpy(test_vis_images,'USTC CMFR Test Vis Images.npy')
# write_numpy(test_the_images,'USTC CMFR Test The Images.npy')