### reads the data files from the folders and generates two files:
    #  DB_Images.text: path to all files in a dictionary format with subject id
    #  DB_labels_lookup.txt: subject id vs subject name dictionary
    #  use json to read and write
    ###
import os
import json


root_path = r'D:\CM Vis Th Face Image dbs\CASIA NIR-VIS 2.0\NIR-VIS-2.0'

def write_Json(datatype):

    with open('SURF ' + datatype + ' filepaths.txt','w') as images:
        json.dump(files_dic,images)

def get_key(seq,modality,subject,file_name): ##dict for key number
    # key = subject +'_'+ modality +'_'+ seq
    key = seq + '_' + modality + '_' + str(subject) + '_' + str(int(file_name.split('.')[0]))
    # if key in files_dic:
    #     print('ERROR KEY EXISTS')
    # else:
    return key 

def add_to_dic(img_path):
    subject = int(img_path.split('\\')[-2])
    modality = img_path.split('\\')[-3]
    seq = img_path.split('\\')[-4]
    file_name = img_path.split('\\')[-1]
    key = get_key(seq,modality,subject,file_name)
    files_dic[key] = [subject,modality,seq,file_name,img_path]


def read_txt(way):    
    file = open(way,'r')
    Lines = file.readlines()
    return Lines

def train_data(vis_txt,nir_txt):
    
    
    files_list = dict()
    files_list['vis'] = read_txt(vis_txt)
    files_list['nir'] = read_txt(nir_txt)


    for i in range(len(files_list['vis'])):
        vis_path = os.path.join(root_path, files_list['vis'][i].split('\n')[0])
        nir_path = os.path.join(root_path, files_list['nir'][i].split('\n')[0])
        if os.path.exists(vis_path) and os.path.exists(nir_path):
            add_to_dic(vis_path)
            add_to_dic(nir_path)
        else:
            print(vis_path)
            print(nir_path)

def test_data(vis_txt,nir_txt):


    for i in range(10):
        vis_p = vis_txt + str(i+1) + '.txt'
        nir_p = nir_txt + str(i+1) + '.txt'

        files_list = dict()
        files_list['vis'] = read_txt(vis_p)
        files_list['nir'] = read_txt(nir_p)


        for i in range(len(files_list['vis'])):
            vis_path = os.path.join(root_path, files_list['vis'][i].split('\n')[0])
            nir_path = os.path.join(root_path, files_list['nir'][i].split('\n')[0])
            if os.path.exists(vis_path) and os.path.exists(nir_path):
                add_to_dic(vis_path)
                add_to_dic(nir_path)
            else:
                print(vis_path)
                print(nir_path)
    

sub_folder_paths = []
files_dic = dict()

vis_txt = r'D:\CM Vis Th Face Image dbs\CASIA NIR-VIS 2.0\HFB-Supplemental-Protocol\HFB_protocols\vis_train_dev.txt'
nir_txt = r'D:\CM Vis Th Face Image dbs\CASIA NIR-VIS 2.0\HFB-Supplemental-Protocol\HFB_protocols\nir_train_dev.txt'
train_data(vis_txt,nir_txt)
vis_txt = r'D:\CM Vis Th Face Image dbs\CASIA NIR-VIS 2.0\HFB-Supplemental-Protocol\HFB_protocols\vis_gallery_dev.txt'
nir_txt = r'D:\CM Vis Th Face Image dbs\CASIA NIR-VIS 2.0\HFB-Supplemental-Protocol\HFB_protocols\nir_probe_dev.txt'
train_data(vis_txt,nir_txt)
write_Json('Train')

sub_folder_paths = []
files_dic = dict()

vis_txt = r'D:\CM Vis Th Face Image dbs\CASIA NIR-VIS 2.0\HFB-Supplemental-Protocol\HFB_protocols\vis_train_'
nir_txt = r'D:\CM Vis Th Face Image dbs\CASIA NIR-VIS 2.0\HFB-Supplemental-Protocol\HFB_protocols\nir_train_'
test_data(vis_txt,nir_txt)
vis_txt = r'D:\CM Vis Th Face Image dbs\CASIA NIR-VIS 2.0\HFB-Supplemental-Protocol\HFB_protocols\vis_gallery_'
nir_txt = r'D:\CM Vis Th Face Image dbs\CASIA NIR-VIS 2.0\HFB-Supplemental-Protocol\HFB_protocols\nir_probe_'
test_data(vis_txt,nir_txt)
write_Json('Test')
