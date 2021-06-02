import torch, csv, DataLoader

from torch.utils import data
from torchvision import transforms
from densenetpytorch import My_SE_DenseNet, My_Modules
from pytorch_metric_learning import testers
import numpy as np
import sklearn.metrics as skm
import sklearn as sk

def test_triplet(GPUnet, info_dic):
    GPUnet.eval()
    # data_loader, classes = DataLoader.create_data_loader(info_dic, test=True)
    t = testers.GlobalEmbeddingSpaceTester(normalize_embeddings=True, batch_size=128, dataloader_num_workers=2 )
    transform = transforms.Compose([
                transforms.ToTensor()
                ])
    dataset = DataLoader.FaceLandmarksDataset(info_dic, transform)
    dataset_dict = {info_dic['subset']: dataset}

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if torch.cuda.device_count() > 1:
    #     # GPUnet = torch.nn.DataParallel(net)
        GPUnet.to(device)


    all_accuracies = t.test(dataset_dict, 299, GPUnet)
    acc = all_accuracies[info_dic['subset']]['precision_at_1_level0']*100
    # print(info_dic['subset'], 'Accuracy: ', acc)
    
    return acc

def test_cmd(cmdnet, info_dic, backbone):
    cmdnet.eval()
    transform = transforms.Compose([
                transforms.ToTensor()
                ])
    dataset = DataLoader.FaceLandmarksDataset(info_dic, transform)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.nn.DataParallel(backbone).to(device)
    torch.nn.DataParallel(cmdnet).to(device)

    disc_pred = []
    labels = []
    data_len = dataset.__len__()
    for i in range(data_len):
        img1, lbl1 = dataset.__getitem__(i)
        for j in range(i+1, data_len):
            img2, lbl2 = dataset.__getitem__(j)
            stack = torch.stack((img1,img2),0).to(device)
            x = backbone(stack).to(device)
            x = torch.cat((x.view(1,x.size()[0],x.size()[1]).tile((x.size()[0],1,1)),
                            x.tile(1,1,x.size()[0]).reshape(x.size()[0], x.size()[0],x.size()[1])
                            ),2).reshape(-1,x.size()[1]*2)
            pred = cmdnet(x)
            pred = pred.detach().cpu().numpy()
            pred = np.average(pred[1:3])
            disc_pred.append(pred)
            if lbl1 == lbl2: labels.append(1)
            else: labels.append(0)

    disc_pred, labels = np.array(disc_pred), np.array(labels)
    disc_pred = sk.preprocessing.minmax_scale(disc_pred, feature_range=(0,1))
    disc_pred = np.where(disc_pred > 0.5, 1, 0)
    cmd_acc = skm.accuracy_score(labels,np.round(disc_pred, decimals=0))

    return cmd_acc * 100

def test_model(info_dic):
    if info_dic['model'] == 'BaseDenseSe': 
        net = My_SE_DenseNet.my_se_densenet121_g32(256)
        net = torch.nn.DataParallel(net).to(torch.device("cuda"))
        saved = torch.load(r'E:/Work/Cross Modality FR 2/Pytorch DenseSE/Models/' + info_dic['run_name'] + '.pth')
        net.load_state_dict( saved['trunk'])

        print('Loaded model: ', info_dic['run_name'] + '.pth')
        net.eval()
        return test_triplet(net, info_dic)
    elif info_dic['model'] == 'DenseCMD': ## Change to save CMD and Test on CMD
        net = My_SE_DenseNet.my_se_densenet121_g32(256)
        net = torch.nn.DataParallel(net).to(torch.device("cuda")) #torch.nn.DataParallel(net)
        saved = torch.load(r'E:/Work/Cross Modality FR 2/Pytorch DenseSE/Models/' + info_dic['run_name'] + '.pth')
        net.load_state_dict(saved['trunk'])
        print('Loaded model: ', info_dic['run_name'] + '.pth')
        net.eval()
        trip_acc = test_triplet(net, info_dic)
        cmd = My_Modules.CMD(256)
        cmd = torch.nn.DataParallel(cmd).to(torch.device("cuda"))
        cmd.load_state_dict(saved['cmd'])
        cmd.eval()
        cmd_acc = test_cmd(cmd,info_dic, net)
        return [trip_acc, cmd_acc]

def create_info_dic(model, loss, db, modality, Cmnt):
    info_dic = dict()
    info_dic['epochs'] = 2
    info_dic['subset'] = ' Test '
    info_dic['data_dir'] = r'E:\Work\Cross Modality FR 2\Numpy Data'
    info_dic['model'] = model
    info_dic['loss'] = loss
    info_dic['DB'] = db
    info_dic['modality'] = modality
    mods = '-'.join(modality)
    info_dic['run_name'] = f'{db}_{mods}_{model}_{loss}_{Cmnt}'
    return info_dic

def write_acc(acc, info_dic): 
    f = open('Results.csv','a')
    with f:
        writer = csv.writer(f, lineterminator='\n')
        row = [info_dic['run_name'], info_dic['model'], info_dic['loss'], 
                info_dic['DB'], '-'.join(info_dic['modality']),info_dic['subset'], acc]
        writer.writerow(row)
    print (info_dic['run_name'],' ', info_dic['subset'],' ', acc)
    
    return

if __name__ == '__main__':
    with open('TestDic.csv',newline='') as csvfile:
            spamreader = csv.reader(csvfile, delimiter=',')
            for row in spamreader:
                num, model, loss, db, modality, run, Cmnt = row
                modalities = list(modality.split(','))
                if run == '1':
                    # print('Testing: ', num, model, loss, db, modalities, Cmnt )
                    info_dic = create_info_dic(model, loss, db, modalities, Cmnt)
                    for data_m in [' Train ',' Test ']:
                        info_dic['subset'] = data_m
                        acc = test_model(info_dic)
                        write_acc(acc, info_dic)