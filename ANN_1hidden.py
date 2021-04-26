import pandas as pd
import numpy as np
import torch 
import torch.nn as nn
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt
import random

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
lr = 0.001
bottleneck = 794    #epilegoume tis times sumfwna me ta erwthmata (10, 397, 794)

#apo thn timh tou label ftiaxnoume ena one hot dianusma - gia to mse
def batch2onehots(labels):
    one_hot = torch.zeros(labels.size()[0], 10)
    for i in range(len(labels)):
        num = labels[i]
        one_hot[i,int(num.detach().cpu().numpy())] = 1
        
    return one_hot

#cross validation
def n_fold(data,labels,n):
    data_len = len(data)
    eval_len = data_len / 5 #to 1/5 einai gia eval sto cv
    eval_len = int(eval_len)
    eval_data = data[n*eval_len: (n+1) * eval_len, :]
    eval_labels = labels[n*eval_len: (n+1) * eval_len]
    #pairnw gia train o,ti yparxei prin kai meta to eval - 4/5
    train_data = np.concatenate((data[(n-1)*eval_len: n*eval_len, :], data[(n+1)*eval_len:,:]),axis=0) 
    train_labels = np.concatenate((labels[(n-1)*eval_len: n*eval_len] ,labels[(n+1)*eval_len:]),axis=0)
    
    return eval_data, eval_labels, train_data, train_labels    


#arxitektonikh nn
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(784, bottleneck)       #eisodos
        self.fc2 = nn.Linear(bottleneck, 10)

        self.relu = nn.LeakyReLU(0.2)                #sun. energopoihshs krufou epipedou
        self.softmax = nn.Softmax(dim=1)     #sun. energopoihshs epipedou eksodou
        
    def forward(self, input):
        x = self.relu(self.fc1(input))   # hidden layer
        return self.softmax(self.fc2(x)) # output layer


#dhmiourgia custom dataset synarthshs gia to dataloader
class myDataset(torch.utils.data.Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        data = self.data[idx]        
        labels = self.labels[idx]
        sample = {'data' : data, 'labels' : labels}
        return sample

#diavazo ta dedomena
train = pd.read_csv('./mnist_train.csv')
test = pd.read_csv('./mnist_test.csv')

#apo pandas se numpy
x = train.to_numpy()

#ksexwrizw labels apo dedomena
labels = x[:,0]
data = x[:,1:]

#normalization [0, 1]
data = data/255

#katanomh dedomenwn
occ = {}
for i in range (10):
    occ[str(i)]=str(np.count_nonzero(labels == i)/600 )+"%"
    

fold_avg_accuracy = 0  
fold_avg_loss = 0

for fold in range(5):
    print("Fold:",fold)
    train_fold_plot = []
    eval_fold_plot = []

    plt.figure()

    eval_data, eval_labels, train_data, train_labels = n_fold(data, labels, fold) 
    eval_dataset = myDataset(eval_data, eval_labels)
    train_dataset = myDataset(train_data, train_labels)
    
    eval_dataloader = DataLoader(eval_dataset, batch_size=64, num_workers= 12)
    train_dataloader = DataLoader(train_dataset, batch_size=64, num_workers= 12)

    net = Net().double().to(device)
    optimizer = torch.optim.Adam(net.parameters(),lr = lr)

    loss_function1 = nn.MSELoss() 

    for epoch in range(20):
        epoch_loss = 0
        for j, data1 in enumerate(train_dataloader):
            dedomena = data1['data'].double().to(device)
            etiketes = data1['labels'].double().to(device)
            
            goal = batch2onehots(etiketes).double().to(device)
            output = net(dedomena)
            
            loss_mse = loss_function1(output,goal)
            
            batch_ce = torch.sum(goal*torch.log(output),dim=1)
            loss_ce = -torch.mean(batch_ce)
            
            #epilegoume th metrikh pou theloume 
            #epoch_loss += loss_ce.item()
            epoch_loss += loss_mse.item()
            
            #epilegoume me poia metrikh theloume na ekpaideusoume
            #loss_ce.backward()
            loss_mse.backward()
            
            optimizer.step()
            net.zero_grad()
            
        correct = 0
        total = 0

        #eval
        with torch.no_grad():
            eval_loss = 0
            for eval_ind, data1 in enumerate(eval_dataloader):
                dedomena = data1['data'].double().to(device)
                etiketes = data1['labels'].double().to(device)
                goal = batch2onehots(etiketes).double().to(device)
                output = net(dedomena)

                loss_mse = loss_function1(output,goal)
                batch_ce = torch.sum(goal*torch.log(output),dim=1)
                loss_ce = -torch.mean(batch_ce)

                #antistoixa me to train
                #eval_loss += loss_ce.item()
                eval_loss += loss_mse.item()                

                #ypologizoyme to eval accuracy
                for idx, i in enumerate(output):
                   
                    if torch.argmax(i) == etiketes[idx]:
                        correct += 1
                    total += 1
                    
        avg_epochLoss =  epoch_loss/len(train_dataloader)
        avg_evalLoss = eval_loss/len(eval_dataloader)

        train_fold_plot.append(avg_epochLoss)       
        eval_fold_plot.append(avg_evalLoss)       

        print('epoch: %d Train loss = %.4f, Eval loss = %.4f, Eval accuracy = %.3f' % (epoch, avg_epochLoss, avg_evalLoss, (correct/total)))

        plt.plot(train_fold_plot)
        plt.plot(eval_fold_plot)
        plt.title('Fold:'+str(fold))
        plt.legend(['Test Loss','Train Loss'])
    
    fold_avg_accuracy+=(correct/total)
    fold_avg_loss += avg_evalLoss

fold_avg_accuracy = fold_avg_accuracy/5            
fold_avg_loss = fold_avg_loss/5