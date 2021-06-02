

    ### reads the data files from the folders and generates two files:
    #  IRIS_Images.text: path to all files in a dictionary format with subject id
    #  IRIS_labels_lookup.txt: subject id vs subject name dictionary
    #  use json to read and write
    ###

import os
import json

rootpath = r'D:\CM Vis Th Face Image dbs\IRIS'
 
sub_folder_paths = []
files_dic = dict()
labels_dic = dict()

def find_Label(key):

    sub_name = key.split('_')[0]
    label_no = labels_dic[sub_name]

    return label_no


def read_Image_Names(exp_ill_path):
    # exp_ill_path = root.../brad/brad/Expression
    #     
    i = 0
    path_folder_names = os.listdir(exp_ill_path)
    for path_folder_name in path_folder_names: #exp1, exp2, exp3
        folder_path = os.path.join(exp_ill_path,path_folder_name)
        all_files = os.listdir(folder_path)
        for file in all_files:
            if file.endswith(".bmp"):
                file_path = os.path.join(folder_path,file) # join to create path to image file
                sub_id = file_path.split('\\')  # split the path to get individual folder names
                key = sub_id[3] + '_' + file    # join subject name, '_', file name as dic key
                if key in files_dic:            # if the key already exist, put an incremental number at the end of it
                    i+=1
                key = key.split('.')[0] + '_' + str(i) + '.bmp'
                files_dic[key] = [file_path]    # add file path to dic
                label_no = find_Label(key)      # get subject label (subjectname_imagename)
                files_dic[key].append(label_no) 
                #now we have a dict with key = subjectname_imagename_i.bmp
                # dict[0] = file path
                # dict[1] = label number (subject id)


def write_Json():

    with open('IRIS_label_lookup.txt','w') as lookup:
        json.dump(labels_dic,lookup)
    
    with open('IRIS_Images.txt','w') as images:
        json.dump(files_dic,images)



sub_folder_names = os.listdir(rootpath) #subject named folder
sub_label = 0
for sub_folder_name in sub_folder_names:
    sub_folder_paths.append(os.path.join(rootpath, sub_folder_name,sub_folder_name))
    labels_dic[sub_folder_name] = sub_label
    sub_label += 1

for sub_folder_path in sub_folder_paths:
    exp_path = os.path.join(sub_folder_path,'Expression')
    read_Image_Names(exp_path)
    ill_path = os.path.join(sub_folder_path,'Illumination')
    read_Image_Names(ill_path)


write_Json()

