import pandas as pd
import numpy as np
import random
import torch 
import torch.nn as nn
from torch.utils.data import DataLoader 
import random
from matplotlib import pyplot as plt

########## Functions ##########

def initializePopulation(size, genes):
    population = []
    for i in range(size):
        chromosome = np.random.randint(2, size=genes)
        population.append(chromosome)
    return population


########## ANN ##########
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(784, 397)       #eisodos
        self.fc2 = nn.Linear(397, 204)
        self.fc3 = nn.Linear(204, 10)

        self.relu = nn.LeakyReLU(0.2)        #sun. energopoihshs krufou epipedou
        self.softmax = nn.Softmax(dim=1)     #sun. energopoihshs epipedou eksodou
        
    def forward(self, input):
        x = self.relu(self.fc1(input))   # hidden layer
        x = self.relu(self.fc2(x))       # hidden layer
        return self.softmax(self.fc3(x)) # output layer
    
# Custom Dataset function
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
    
#read testdata
testData = pd.read_csv('./mnist_test.csv')
x = testData.to_numpy()
labels = x[:,0]
data = x[:,1:]
data = data/255

test_dataset = myDataset(data, labels)
test_dataloader = DataLoader(test_dataset, batch_size=64, num_workers= 0)

#arxikopoihsh diktyoy me ta varh apo to apothikeumeno tou merous A
net = Net().double()
net.load_state_dict(torch.load('./bestANN.pth'))
net.eval()

chromosomes = 200
genes = 784
num_inputs = 450
mean_best_fit = 0
mean_epoch = 0
mean_plot = []

for loop in range(10):
    #gia to termatismo
    buffer = [0 for i in range(10)]
    temp_plot = []
    population = initializePopulation(chromosomes, genes)
    for epoch in range(100):
        test_total = np.zeros(chromosomes)
        test_correct = np.zeros(chromosomes)
        avg_test_accuracy = np.zeros(chromosomes) 
    
        for j, data1 in enumerate(test_dataloader):
    
            dedomena = data1['data'].double()
            etiketes = data1['labels'].double()
                    
            for k, chromosome in enumerate(population):
                chromosome_output = net(dedomena * chromosome)       
            
                for idx, i in enumerate(chromosome_output):
                   
                    if torch.argmax(i) == etiketes[idx]:
                        test_correct[k]+= 1
                    test_total[k] += 1
            
        #gia poinh sto accuracy an exei megalo arithmo eisodwn
        if np.count_nonzero(chromosome == 1) > num_inputs:
            avg_test_accuracy=90*(test_correct/test_total)
        else:
            avg_test_accuracy=100*(test_correct/test_total)
            
        temp_plot.append(avg_test_accuracy.max())
        if avg_test_accuracy.max() == buffer[0]:
            break

        buffer.append(avg_test_accuracy.max())
        buffer.pop(0)

        population = list(np.array(population)[np.argsort(avg_test_accuracy)])
        
        print('Loop: '+str(loop)+' Epoch: '+str(epoch)+' Best accuracy: '+str(avg_test_accuracy.max()))
        
        #selection
        p_chromosome = avg_test_accuracy / avg_test_accuracy.sum()
        q_chromosome = [p_chromosome[0:i].sum() for i in range(1,chromosomes+1)]
        r = [random.uniform(0, 1) for i in range(chromosomes)]
        
        temp_pop = []
        for i in range(chromosomes):
            if r[i] <= q_chromosome[i]:
                temp_pop.append(population[i])
                
        population = []
        _ = [population.append(i) for i in temp_pop]
    
        while len(population) < chromosomes:
            population.extend(population)
            
        population = population[0:chromosomes]
        
        #crossover
        pc = 0.9          
        mating_pool = []
        rest_pop = []
        r = [random.uniform(0, 1) for i in range(chromosomes)]
        for i in range(chromosomes):
            if r[i] < pc:
                mating_pool.append(population[i])
            else:
                rest_pop.append(population[i])
    
        for i in range(0,len(mating_pool),2):
            if i+1 == len(mating_pool): continue
        
            k = random.randrange(1,genes)
            
            a = np.concatenate((mating_pool[i][0:k],mating_pool[i+1][k:]),axis = 0)
            b = np.concatenate((mating_pool[i+1][0:k],mating_pool[i][k:]),axis = 0)   
            
            mating_pool[i] = a        
            mating_pool[i+1] = b
            
        population = []    
        
        _ = [population.append(i) for i in mating_pool]
        population.extend(rest_pop)
        
        #mutation
        pm = 0.01
        for i in range(chromosomes):
            for j in range(genes):
                if np.random.uniform(0,1)< pm:
                    population[i][j] = population[i][j]^1
    mean_plot.append(temp_plot)
    mean_epoch += epoch
    mean_best_fit += avg_test_accuracy.max()    

mean_best_fit /= 10
mean_epoch /= 10

#ftiaxnoume ta plots
plt.figure()
plt.title('Mean Best Accuracy Plot for case 9')
plt.ylabel('Mean Accuracy')
plt.xlabel('Generations')

#gia ton ypologismo tou plot kathws den einai idios o arithmos geniwn
max_len = 0
for i in range(10):
    max_len = max([max_len, len(mean_plot[i])])

plot_array = np.zeros((10,max_len))

for i in range(10):
    plot_array[i,0:len(mean_plot[i])] = np.array(mean_plot[i])

plt.plot(plot_array.sum(0)/np.count_nonzero(plot_array,axis=0))

