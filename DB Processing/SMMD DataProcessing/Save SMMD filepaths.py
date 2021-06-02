### reads the data files from the folders and generates two files:
    #  DB_Images.text: path to all files in a dictionary format with subject id
    #  DB_labels_lookup.txt: subject id vs subject name dictionary
    #  use json to read and write
    ###
import os
import json

rootpath = r'D:\C DataSet\Data Face Cropped'


sub_folder_paths = []
files_dic = dict()
key_dic = dict()
labels_dic = dict()


def write_Json():

    with open('SMMD filepaths.txt','w') as images:
        json.dump(files_dic,images)

def get_key(subject,file_name): ##dict for key number
    key = subject +'_'+ file_name
    if key in key_dic:
        key_dic[key] = key_dic[key] + 1
        key + '_' + str(key_dic[key])
        print('duplicate')
    else:
        key_dic[key] = 0
        return key
    

def add_to_dic(img_path):
    subject = img_path.split('\\')[4]
    modality = img_path.split('_')[-1].split('.')[0]
    file_name = img_path.split('\\')[-1]
    key = get_key(subject,file_name)
    files_dic[key] = [subject,modality,file_name,img_path]


def explore_dir(sub_dir):
    addons = os.listdir(sub_dir)
    for addon in addons:
        addon_dir = os.path.join(sub_dir,addon)
        images = os.listdir(addon_dir)
        for image in images:
            if ('.jpg' in image):
                img_path = os.path.join(addon_dir,image)
                add_to_dic(img_path)


for gender in ['Female','Male']:
    path = os.path.join(rootpath,gender)
    subjects = os.listdir(path)
    for subject in subjects:
        subject_path = os.path.join(path,subject)
        explore_dir(subject_path)

write_Json()