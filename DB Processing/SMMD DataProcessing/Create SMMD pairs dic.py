    ### reads IRIS_Images.text and saves image pairs in another dictionary with labels:
    #  output file format
    #  Json Dict key = img no, [label, visible image path, thermal image path]
    #  use json to read and write
    ###
import json



def create_pairs(files_dic):                        # need a dictionary for thermal and visible image pairs, paths and subject id
    pairs_dic = dict()

    for vi_key in files_dic:
        if '_v.jpg' in vi_key:
            th_key = vi_key.replace('_v','_t')
            ir_key = vi_key.replace('_v','_d')
            vir_key = vi_key.replace('_v','_l')
            if (th_key in files_dic) and (ir_key in files_dic) and (vir_key in files_dic):
                folder = files_dic[vi_key][0]
                vis_path = files_dic[vi_key][3]
                the_path = files_dic[th_key][3]
                ir_path = files_dic[ir_key][3]
                vir_path = files_dic[vir_key][3]
                pairs_dic[vi_key.replace('_v.jpg','')] = [folder, vis_path, the_path, ir_path, vir_path]

    return pairs_dic

def write_Jason(data, file_name):

    with open(file_name,'w') as io_file:
        json.dump(data,io_file)

def read_Json(file_path):
    with open(file_path) as data:
        files_dic = json.load(data)                     # files_dic key = subjectname_imagename_i.bmp, dict[key][0] = file path, dict[key][1] = subject id 

    return files_dic




file_path = 'SMMD filepaths.txt' 
files_dic = read_Json(file_path)
pairs_dic = create_pairs(files_dic)
write_Jason(pairs_dic,"SMMD Img Pairs.txt")