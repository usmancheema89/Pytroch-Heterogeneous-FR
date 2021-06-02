    ### reads IRIS_Images.text and saves image pairs in another dictionary with labels:
    #  output file format
    #  Json Dict key = img no, [label, visible image path, thermal image path]
    #  use json to read and write
    ###
import json



def create_pairs(files_dic):                        # need a dictionary for thermal and visible image pairs, paths and subject id
    img_no = 0                                      # total no of images, used a key for img_pair_dic
    img_pairs_dic = dict()
    for v_key in files_dic:
        if '_V-' in v_key:                          # check only visible images    
            ther_key = v_key.replace('_V-','_L-')
            
            if ther_key in files_dic:
                vis_img_path = files_dic[v_key][0]
                the_img_path = files_dic[ther_key][0]
                label = files_dic[v_key][1]
                img_pairs_dic[img_no] = [label, vis_img_path, the_img_path]
                img_no += 1
                # detect_Face(vis_img_path)         # face detection of thermal images doesnt work well !
                # detect_Face(the_img_path)

    return img_pairs_dic

def write_Jason(data, file_name):

    with open(file_name,'w') as io_file:
        json.dump(data,io_file)

def read_Json(file_path):
    with open(file_path) as data:
        files_dic = json.load(data)                     # files_dic key = subjectname_imagename_i.bmp, dict[key][0] = file path, dict[key][1] = subject id 

    return files_dic




file_path = 'IRIS_Images.txt' 
files_dic = read_Json(file_path)
img_pairs_dic = create_pairs(files_dic)
write_Jason(img_pairs_dic,"IRIS Img Pairs.txt")