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
        subject = key.split('d')[0]
        if 'train' not in key:
            vis_img = get_img(pairs_file[key][1])
            the_img = get_img(pairs_file[key][2])
            if (vis_img is not None) and (the_img is not None):
                label = get_label(key)
                if label < 50:
                    train_vis_images.append(vis_img)
                    train_the_images.append(the_img)
                    train_labels.append(label)
                else:
                    test_vis_images.append(vis_img)
                    test_the_images.append(the_img)
                    test_labels.append(label)


def get_label(path):
    raw = int(path.split('d')[0])
    if raw < 4200:
        label = 0
    else:
        label = raw - 4199 
    return label

def get_img(img_path):

    img = cv2.imread(img_path,cv2.IMREAD_UNCHANGED) #uint8 


    if 'ir' in img_path:
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                if img[i][j] == 0:
                    img[i][j] = img.max()
        img = cv2.cvtColor(img,cv2.COLOR_GRAY2RGB)
        img = cv2.convertScaleAbs(cv2.normalize(img,img,0,255,cv2.NORM_MINMAX) )

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
        # cv2.imshow('img',img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()


    return img

def crop_img(img):

    if img.shape[1] > img.shape[0]:
        diff = img.shape[1] - img.shape[0]
        (h,w) = img.shape[:2]
        topx, topy = int(0), int(diff/2)
        botx, boty = int(w), int(w-diff/2)
        img_crop = img[topx:botx, topy:boty]

    else:
        diff = img.shape[0] - img.shape[1] 
        (h,w) = img.shape[:2]
        topx, topy = int(diff/2), int(0)
        botx, boty = int(h-diff/2), int(w)
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


pairs_file_name = 'UNDX Img Pairs.txt'
pairs_file = read_Json(pairs_file_name)
read_file(pairs_file)

# x = np.unique(np.array(train_labels))
# print(x)

write_numpy(train_labels,'UNDX CMFR Train Labels.npy')
write_numpy(train_vis_images,'UNDX CMFR Train Vis Images.npy')
write_numpy(train_the_images,'UNDX CMFR Train The Images.npy')
write_numpy(test_labels,'UNDX CMFR Test Labels.npy')
write_numpy(test_vis_images,'UNDX CMFR Test Vis Images.npy')
write_numpy(test_the_images,'UNDX CMFR Test The Images.npy')