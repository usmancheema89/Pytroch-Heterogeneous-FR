import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from pytorch_metric_learning.distances import CosineSimilarity
from pytorch_metric_learning.reducers import ThresholdReducer
from pytorch_metric_learning.regularizers import RegularFaceRegularizer
from pytorch_metric_learning import losses, miners
from densenetpytorch import My_Modules

def triplet_margin_loss():
    loss_func = losses.TripletMarginLoss(margin = 1.0, distance = CosineSimilarity(),
                embedding_regularizer = RegularFaceRegularizer())
    miner = miners.MultiSimilarityMiner()
    return loss_func, miner
# def create_dic(names, models):
#     model_dic = dict()
#     for i in range(len(names)):
#         model_dic[names[i]] = models[i]
# return model_dic

def train_CMD_Dense(dataloader, net, info_dic):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.device_count() > 1:
        GPUnet = nn.DataParallel(net[0]).to(device)
        cmdnet = nn.DataParallel(net[1]).to(device)

    criterion, miner = triplet_margin_loss()
    cmd_criterion = My_Modules.CMDLoss
    parameters = list(GPUnet.parameters()) + list(cmdnet.parameters()) 
    optimizer = optim.Adam(parameters, lr=0.03)
    tb_writer = SummaryWriter(log_dir=r'./Logs/'+info_dic['run_name'])
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.7)
                    # ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=6, threshold=0.01, 
                    # threshold_mode='rel', cooldown=0, min_lr=0.0000001, eps=1e-08, verbose=True)

    epochs = info_dic['epochs']
    for epoch in range(epochs):  # loop over the dataset multiple times        
        cm_trip_loss = 0.0
        cm_cmd_loss = 0.0
        batches = 0
        least_loss = 100

        for i, data in enumerate(dataloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            input_tag, label_tag = data
            inputs, labels = input_tag.to(device), label_tag.to(device) # to device kerlo input tag ko
                        
            outputs = GPUnet(inputs)
            x = outputs.to(device)
            x = torch.cat((
            x.view(1,x.size()[0],x.size()[1]).tile((x.size()[0],1,1)),
            x.tile(1,1,x.size()[0]).reshape(x.size()[0], x.size()[0],x.size()[1])
            ),2).reshape(-1,x.size()[1]*2)

            cmd_outputs = cmdnet(x)
            cmd_outputs = cmd_outputs.to(device)
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            hard_pairs = miner(outputs, labels)
            loss = criterion(outputs, labels, hard_pairs)
            cmd_loss = cmd_criterion(cmd_outputs,labels)
            total_loss = loss + cmd_loss
            total_loss.backward()
            optimizer.step()
            
            # print statistics
            cm_trip_loss += loss.item()
            cm_cmd_loss += cmd_loss.item()
            batches  += 1
        lr_scheduler.step(loss)
        print('Epoch: %d, trp_loss: %.3f, cmd_loss: %.3f' %(epoch + 1, cm_trip_loss / batches, cm_cmd_loss / batches))
        tb_writer.add_scalar('Trip_Loss/train', cm_trip_loss / batches, epoch+1)
        tb_writer.add_scalar('CMD_Loss/train', cm_cmd_loss / batches, epoch+1)
        
        if least_loss >= ((cm_trip_loss + cm_cmd_loss) / batches):
            least_loss = (cm_trip_loss + cm_cmd_loss) / batches
            PATH = r'./Models/Best_' + info_dic['run_name'] + '.pth'
            torch.save({'trunk': GPUnet.state_dict(), 'cmd': cmdnet.state_dict()}, PATH)

        batches = 0

    print('Finished Training')
    
    PATH = r'./Models/' + info_dic['run_name'] + '.pth'
    torch.save({'trunk': GPUnet.state_dict(), 'cmd': cmdnet.state_dict()}, PATH)


    return [GPUnet, cmdnet]    

def train(dataloader, net, info_dic):

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if torch.cuda.device_count() > 1:
        # print("Let's use", torch.cuda.device_count(), "GPUs!")
        # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
        GPUnet = nn.DataParallel(net)
        GPUnet.to(device)
    
    # criterion = nn.CrossEntropyLoss()
    criterion, miner = triplet_margin_loss()
    optimizer = optim.Adam(GPUnet.parameters(), lr=0.03)
    tb_writer = SummaryWriter(log_dir=r'./Logs/'+info_dic['run_name'])
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.6)
                                    # ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=6, threshold=0.01, 
                    # threshold_mode='rel', cooldown=0, min_lr=0.0000001, eps=1e-08, verbose=True)

    epochs = info_dic['epochs']
    for epoch in range(epochs):  # loop over the dataset multiple times        
        running_loss = 0.0
        batches = 0
        correct_pred = 0
        total_pred = 0
        for i, data in enumerate(dataloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            input_tag, label_tag = data
            inputs, labels = input_tag, label_tag
            # inputs, labels = data[input_tag].to(device), data[label_tag]#.to(device)
                        
            outputs = GPUnet(inputs.to(device))

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            np_label = labels.numpy()
            hard_pairs = miner(outputs, labels)
            loss = criterion(outputs, labels, hard_pairs)
            loss.backward()
            optimizer.step()
            # predict
            # _, predictions = torch.max(outputs, 1)
            # # collect the correct predictions for each class
            # for label, prediction in zip(labels, predictions):
            #     if label == prediction:
            #         correct_pred += 1
            #     total_pred += 1
            
            # print statistics
            running_loss += loss.item()
            batches  += 1
        lr_scheduler.step(loss)
        accuracy = 1 #correct_pred/total_pred * 100
        print('Epoch: %d, loss: %.3f, Acc: %.3f' %(epoch + 1, running_loss / batches, accuracy))
        tb_writer.add_scalar('Loss/train', running_loss / batches, epoch+1)
        tb_writer.add_scalar('Accuracy/train', accuracy, epoch+1)
        running_loss = 0.0
        batches = 0
        correct_pred = 0
        total_pred = 0    

    print('Finished Training')
    
    PATH = r'./Models/' + info_dic['run_name'] + '.pth'
    torch.save(GPUnet.state_dict(), PATH)

    return GPUnet