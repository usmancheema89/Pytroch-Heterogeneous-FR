import json, cv2, os
import numpy as np
from shutil import copyfile

from skimage.transform import resize
from skimage.util import img_as_ubyte
import matplotlib.pyplot as plt

def read_file(pairs_file):
    base_path_vis = r'C:\USTC\Vis'
    base_path_the = r'C:\USTC\The'
    for key in pairs_file:
        subject = key.split('_')[0]
        subject_vis_folder = os.path.join(base_path_vis,subject)
        subject_the_folder = os.path.join(base_path_the,subject)
        if not os.path.exists(subject_vis_folder):
            os.mkdir(subject_vis_folder)
        if not os.path.exists(subject_the_folder):
            os.mkdir(subject_the_folder)
        
        if int(key.split('_')[-1]) <= 3:
            if not os.path.exists(os.path.join(subject_vis_folder,key+'.jpeg')):
                # print('Not found:', os.path.join(subject_vis_folder,key+'.jpeg'))
                write_img(pairs_file[key][1],os.path.join(subject_vis_folder,key+'.jpeg'))
            if not os.path.exists(os.path.join(subject_the_folder,key+'.jpeg')):
                # print('Not found:', os.path.join(subject_the_folder,key+'.jpeg'))
                write_img(pairs_file[key][2],os.path.join(subject_the_folder,key+'.jpeg'))



def write_img(img_path,path):

    img = cv2.imread(img_path,cv2.IMREAD_UNCHANGED)
    if img is None:
        print('image not found: ', img_path)
    else:
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = crop_img(img)
        cv2.imwrite(path,img)

    return

def crop_img(img):

    half = img.shape[0]/2
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

pairs_file_name = 'USTC Img Pairs.txt'
pairs_file = read_Json(pairs_file_name)
read_file(pairs_file)

# write_numpy(label,'IRIS CMFR Train Labels.npy')
# write_numpy(vis_imgs,'IRIS CMFR Train Vis Images.npy')
# write_numpy(the_imgs,'IRIS CMFR Train The Images.npy')

