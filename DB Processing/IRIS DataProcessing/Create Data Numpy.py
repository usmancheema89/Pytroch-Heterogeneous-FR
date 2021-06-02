import json
import cv2
import numpy as np
from skimage.transform import resize
from skimage.util import img_as_ubyte
import matplotlib.pyplot as plt

def read_file(pairs_file):

    for key in pairs_file:
        if pairs_file[key][0] >= 20:                   # only the first 20 subjects are used for creating the training dataset
            # print(pairs_file[key][0])
            vis_path = pairs_file[key][1]
            the_path = pairs_file[key][2]
            vis_img = get_img(vis_path)
            the_img = get_img(the_path)

            label.append(pairs_file[key][0])
            vis_imgs.append(vis_img)
            the_imgs.append(the_img)

def get_img(img_path):

    img = cv2.imread(img_path,cv2.IMREAD_UNCHANGED)
    # plt.imshow(img)
    # plt.show()
    img = resize_img(img)
    img = crop_img(img)
    img = cv2.resize(img,(224,224), interpolation = cv2.INTER_AREA)
    # img = img_as_ubyte( resize(img, (img.shape[0] // 2, img.shape[1] // 2),preserve_range = False, anti_aliasing=True) )
    # plt.imshow(img)
    # plt.show()
    return img

def crop_img(img):

    half = 256/2
    (h,w) = img.shape[:2]
    topx, topy = int(h/2-half), int(w/2-half)
    botx, boty = int(h/2+half), int(w/2+half)
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



label = []
vis_imgs = []
the_imgs = []

pairs_file_name = 'IRIS Img Pairs.txt'
pairs_file = read_Json(pairs_file_name)
read_file(pairs_file)
write_numpy(label,'IRIS Test Labels.npy')
write_numpy(vis_imgs,'IRIS Test Vis Images.npy')
write_numpy(the_imgs,'IRIS Test The Images.npy')

