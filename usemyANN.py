import myANN
import torch
import random
import numpy
import matplotlib.pyplot as plt

trainXfile = open("trainfile.txt","r",encoding='utf-8')
trainYfile = open("trainY.txt","r",encoding='utf-8')

trainX = torch.zeros(50000,4,dtype=float)
trainY = torch.zeros(50000,4,dtype=float)

for i in range(50000):
    x = trainXfile.readline()
    y = trainYfile.readline()
    x = x.strip('\n')
    X = x.split(' ')
    for j in range(4):
        trainX[i,j] = float(X[j])
    y = y.strip('\n')
    Y = y.split(' ')
    for j in range(4):
        trainY[i, j] = int(Y[j])

ANN = myANN.ANN([4,512,512,4])
ANN.dataInput(trainX,trainY)
ANN.setParams(500,1000,0.01)
list = ANN.train()


x = numpy.arange(0,len(list))
y = numpy.array(list)
plt.plot(x,y)
plt.show()

# ANN.predictone(torch.tensor([[10,10,10,10]],dtype=float))

list = []
for i in range(1000):
    x1 = random.random() * 10
    x2 = random.random() * 10
    x3 = random.random() * 10
    x4 = random.random() * 10
    x = torch.tensor([[x1,x2,x3,x4]],dtype=float)
    f1 = x1 + x2 + x3 + x4
    f2 = x1 - x2 + x3 - x4
    r = ANN.predictone(x)
    print(r ,end='')
    if f1 < 12.5 or f1 > 25:
        print(" ", 0, end='')
        if r == 0 :
            print('right')
            list.append(1)
        else:
            print('wrong')
            list.append(0)
    elif f2 < -2.5:
        print(" ", 1, end='')
        if r == 1:
            print('right')
            list.append(1)
        else:
            print('wrong')
            list.append(0)
    elif f2 > 5:
        print(" ", 2, end='')
        if r == 2:
            print('right')
            list.append(1)
        else:
            print('wrong')
            list.append(0)
    else:
        print(" ", 3, end='')
        if r == 3:
            print('right')
            list.append(1)
        else:
            print('wrong')
            list.append(0)

x = numpy.arange(0,len(list))
y = numpy.array(list)
plt.plot(x,y)
plt.show()

print('precision: ', y.sum() / len(list))