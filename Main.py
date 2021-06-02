import DataLoader as DataLoader
from densenetpytorch import My_SE_DenseNet, My_Modules
from Train import train, train_CMD_Dense
import csv

def create_info_dic(model, loss, db, modality, Cmnt):
    # model, loss, db, modality
    info_dic = dict()
    info_dic['epochs'] = 400
    info_dic['batch_s'] = 192
    info_dic['data_dir'] = r'E:\Work\Cross Modality FR 2\Numpy Data'
    info_dic['subset'] = ' Train '
    
    info_dic['model'] = model
    info_dic['loss'] = loss


    info_dic['DB'] = db
    info_dic['modality'] = modality
    mods = '-'.join(modality)
    info_dic['run_name'] = f'{db}_{mods}_{model}_{loss}_{Cmnt}'
    
    return info_dic


def train_network(info_dic):

    data_loader, classes= DataLoader.create_data_loader(info_dic)

    if info_dic['model'] == 'BaseDenseSe': 
        model = My_SE_DenseNet.my_se_densenet121_g32(256)
        net = train(data_loader, model, info_dic)
    elif info_dic['model'] == 'DenseCMD':
        model = My_SE_DenseNet.my_se_densenet121_g32(256)
        cmd_net = My_Modules.CMD(256)
        net = train_CMD_Dense(data_loader, [model, cmd_net], info_dic)

    del  data_loader, model, net


with open('TrainDic.csv',newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',')
        for row in spamreader:
            num, model, loss, db, modality, run, Cmnt = row
            modalities = list(modality.split(','))
            if run == '1':
                info_dic = create_info_dic(model, loss, db, modalities, Cmnt)
                print(info_dic['run_name'] )
                train_network(info_dic)
