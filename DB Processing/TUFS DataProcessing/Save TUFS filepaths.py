### reads the data files from the folders and generates two files:
    #  DB_Images.text: path to all files in a dictionary format with subject id
    #  DB_labels_lookup.txt: subject id vs subject name dictionary
    #  use json to read and write
    ###
import os
import json

rootpath = r'D:\CM Vis Th Face Image dbs\Tufs'


sub_folder_paths = []
files_dic = dict()
key_dic = dict()
labels_dic = dict()


def write_Json():

    with open('TUFS filepaths.txt','w') as images:
        json.dump(files_dic,images)

def get_key(subject,file_name): ##dict for key number
    key = subject +'_'+ file_name
    if key in files_dic:
        print('over writing')
    return key 

def add_to_dic(img_path):
    subject = img_path.split('\\')[5]
    modality = img_path.split('_')[1]
    seq = img_path.split('\\')[6]
    file_name = img_path.split('\\')[-1]
    if ('_RGB_A_' in file_name) and (int(file_name[9]) == 1):
        file_name = file_name[:8] + file_name[10:]
        key = get_key(subject,file_name)
        files_dic[key] = [img_path]
    elif '_RGB_A_' not in file_name:
        key = get_key(subject,file_name)
        files_dic[key] = [img_path]


def explore_sub_mod_dir(sub_ps):
    for sub_p in sub_ps:
        images = os.listdir(sub_p)
        for image in images:
            img_path = os.path.join(sub_p,image)
            if '.jpg' in img_path:
                if ('_A_' in image) and (int(image.split('_')[-1][0]) in [3, 4, 5]):
                    add_to_dic(img_path)
                elif ('_E_'in image):
                    add_to_dic(img_path)
                # else:
                    # print(img_path)

pres = 'TD_'
mods = ['RGB']
posts = ['_A','_E']
for mod in mods:
    for post in posts:
        folder = os.path.join(rootpath,pres+mod+post)
        sets = os.listdir(folder)
        for sub_set in sets:
            sets_folder = os.path.join(folder,sub_set)
            if os.path.isdir(sets_folder):
                subs = os.listdir(sets_folder)
                for subject in subs:
                    sub_rgb_dir = os.path.join(sets_folder,subject)
                    sub_the_dir = sub_rgb_dir.replace('TD_RGB','TD_IR')
                    if os.path.exists(sub_the_dir) and os.path.exists(sub_rgb_dir):
                        explore_sub_mod_dir([sub_the_dir, sub_rgb_dir])
                    else:
                        print(sub_the_dir, ' Does not exist')


write_Json()