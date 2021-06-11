import DataLoader as DataLoader
from densenetpytorch import My_SE_DenseNet, My_Modules, Siamese_SE_Dense, Siamese_Modules
from Train import train, train_CMD_Dense, train_CMDCIND_Dense, train_CMDCIND_Siam
import csv

def create_info_dic(model, loss, db, modality, Cmnt):
    # model, loss, db, modality
    info_dic = dict()
    info_dic['epochs'] = 300
    info_dic['batch_s'] = 112
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
    elif info_dic['model'] == 'DenseCMDCIND':
        model = My_SE_DenseNet.my_se_densenet121_g32(256, cind = True)
        cmd_net = My_Modules.CMD(256)
        cind_net = My_Modules.CIND()
        net = train_CMDCIND_Dense(data_loader, [model, cind_net, cmd_net], info_dic)
    elif info_dic['model'] == 'SiamCMDCIND':
        data_loader, classes =  DataLoader.get_siamese_data_loader(info_dic)
        model = Siamese_SE_Dense.get_siam_dense(256, cind = True)
        cmd_net = Siamese_Modules.CMD(256)
        cind_net = Siamese_Modules.CIND()
        train_CMDCIND_Siam(data_loader, [model, cind_net, cmd_net], info_dic)

    del  data_loader, model, cmd_net, cind_net


with open('TrainDic.csv',newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',')
        for row in spamreader:
            num, model, loss, db, modality, run, Cmnt = row
            modalities = list(modality.split(','))
            if run == '1':
                info_dic = create_info_dic(model, loss, db, modalities, Cmnt)
                print(info_dic['run_name'] )
                train_network(info_dic)
