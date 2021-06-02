### reads the data files from the folders and generates two files:
    #  DB_Images.text: path to all files in a dictionary format with subject id
    #  DB_labels_lookup.txt: subject id vs subject name dictionary
    #  use json to read and write
    ###
import os
import json

rootpath = r'D:\CM Vis Th Face Image dbs\SpontaneousDatabase'


sub_folder_paths = []
files_dic = dict()
key_dic = dict()
labels_dic = dict()


def write_Json():

    with open('USTC filepaths.txt','w') as images:
        json.dump(files_dic,images)

def get_key(subject,modality,seq): ##dict for key number
    key = subject +'_'+ modality +'_'+ seq
    if key in key_dic:
        key_dic[key] = key_dic[key] + 1
    else:
        key_dic[key] = 0
    
    return key + '_' + str(key_dic[key])

def add_to_dic(img_path):
    subject = img_path.split('\\')[4]
    modality = img_path.split('\\')[5]
    seq = img_path.split('\\')[6]
    file_name = img_path.split('\\')[7]
    key = get_key(subject,modality,seq)
    files_dic[key] = [subject,modality,seq,file_name,img_path]


def explore_sub_mod_dir(sub_ps):
    for sub_p in sub_ps:
        folder_types = os.listdir(sub_p)
        for folder_name in folder_types:
            img_dir = os.path.join(sub_p,folder_name)
            if os.path.isdir(img_dir):
                images = os.listdir(img_dir)
                for image in images:
                    if ('.bmp' in image) or ('.BMP' in image):
                        img_path = os.path.join(img_dir,image)
                        add_to_dic(img_path)


for direction in ['front','left','right']:
    path = os.path.join(rootpath,direction)
    subjects = os.listdir(path)
    for subject in subjects:
        subject_path = os.path.join(path,subject)
        ir_sub_p = os.path.join(subject_path, 'infrared')
        vi_sub_p = os.path.join(subject_path, 'visible')
        if os.path.exists(ir_sub_p) and os.path.exists(vi_sub_p):
            explore_sub_mod_dir([vi_sub_p,ir_sub_p])

write_Json()