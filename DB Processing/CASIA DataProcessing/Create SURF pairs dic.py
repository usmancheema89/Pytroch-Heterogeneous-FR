    ### reads IRIS_Images.text and saves image pairs in another dictionary with labels:
    #  output file format
    #  Json Dict key = img no, [label, visible image path, thermal image path]
    #  use json to read and write
    ###
import json



def create_pairs(files_dic):                        # need a dictionary for thermal and visible image pairs, paths and subject id
    pairs_dic = dict()

    for vi_key in files_dic:
        if 'VIS' in vi_key:
            th_key = vi_key.replace('VIS','NIR')
            if th_key in files_dic:
                subject = files_dic[vi_key][0]
                print(subject)
                vis_path = files_dic[vi_key][4]
                the_path = files_dic[th_key][4]
                pairs_dic[vi_key.replace('_VIS','')] = [subject, vis_path, the_path]

    return pairs_dic

def write_Jason(data, file_name):

    with open(file_name,'w') as io_file:
        json.dump(data,io_file)

def read_Json(file_path):
    with open(file_path) as data:
        files_dic = json.load(data)                     # files_dic key = subjectname_imagename_i.bmp, dict[key][0] = file path, dict[key][1] = subject id 

    return files_dic




for t in ['Train', 'Test']:
    file_path = 'SURF ' + t + ' filepaths.txt' 
    files_dic = read_Json(file_path)
    pairs_dic = create_pairs(files_dic)
    write_Jason(pairs_dic,'SURF ' + t + ' Img Pairs.txt')