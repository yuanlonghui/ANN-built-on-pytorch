import torch
import pickle

class ANN:
    def __init__(self,seq : list or torch.Tensor):
        '''
        构建神经网络的基础形状
        :param seq: 神经网络去除偏置项的形状，最后输出层没有偏置项，其他分别代表着每一层去除偏置项的神经元个数
        '''
        if type(seq) == torch.Tensor:
            layers = seq.size()
        elif type(seq) == list :
            layers = len(seq)
        self.layers_tensor = []
        i = 1
        while i < layers:
            self.layers_tensor.append(torch.randn(seq[i-1]+1,seq[i],dtype=float,requires_grad=True))
            i = i + 1

    def dataInput(self, _dataset: torch.Tensor, _labels: torch.Tensor):
        '''
        数据输入
        :param _dataset: 训练集数据，也就是feature
        :param _labels: 对应的lable
        :return: 输入数据不符合设置的神经网络形状，则输入失败
        '''
        if _dataset.size()[1] != self.layers_tensor[0].size()[0] - 1 :
            print('error feature size with %d'%(self.layers_tensor[0].size()[0] - 1 - _dataset.size()[1]))
            return
        if _labels.size()[1] != self.layers_tensor[-1].size()[1] :
            print('error labels size with %d'%(self.layers_tensor[-1].size[1] - _lables.size()[1]))
            return
        self.dateset = _dataset
        self.labels = _labels

    def setParams(self,batchsize:int, round: int, learning_rate: float):
        '''
        参数设置，采用小批量梯度下降法
        :param batchsize: 批量大小
        :param round: 训练轮数
        :param learning_rate: 学习率
        :return:
        '''
        self.batchsize = batchsize
        self.round = round
        self.learning_rate = learning_rate

    def train(self):
        '''
        这个函数完成训练
        :return: 一个list，记录训练过程的代价函数值
        '''
        list = []
        tsize = self.dateset.size()[0]
        Biasterm = torch.ones(self.batchsize,1,dtype=float)
        r = 0
        while r < self.round:
            for j in range(0 , tsize , self.batchsize):
                print('round: {}, batch: {}, '.format(r,j/self.batchsize),end='')
                if j + self.batchsize <= tsize:
                    k = j + self.batchsize
                    batch_size = self.batchsize
                else:
                    k = tsize
                    batch_size = tsize - j
                x = self.dateset[j:k,:]
                y = self.labels[j:k,:]
                for i in range(len(self.layers_tensor)):
                    x = torch.cat((Biasterm,x),dim=1)
                    x = torch.mm(x,self.layers_tensor[i])
                    x = x.sigmoid()
                c = - (y * x.log() + (1 - y) * (1 - x).log()) / y.size()[1]
                cost = c.sum() / self.batchsize
                list.append(cost.item())
                print("cost: {}".format(cost.item()))
                cost.backward()
                for i in range(len(self.layers_tensor)):
                    self.layers_tensor[i] = self.layers_tensor[i] - self.learning_rate * self.layers_tensor[i].grad
                    self.layers_tensor[i].detach_()
                    self.layers_tensor[i].requires_grad = True
            r = r + 1
        return list

    def predictone(self, x: torch.Tensor):
        '''
        输入一个特征，进行预测
        :param x: 输入特征
        :return: 该特征最有可能的类别
        '''
        if x.size()[0] != 1:
            print('the input is not one data')
            return -1
        Biasterm = torch.ones(1, 1, dtype=float)
        for iterm in self.layers_tensor:
            x = torch.cat((Biasterm, x), dim=1)
            x = torch.mm(x, iterm)
            x = x.sigmoid()
        max = 0
        j = 0
        while j < x.size()[1]:
            if x[0,j] >= x[0,max]:
                max = j
            j = j + 1
        return max

    def predictall(self, x: torch.Tensor):
        '''
        输入一组特征，预测他们的类别
        :param x: 一组特征
        :return: 一个list，记录每个特征的类别
        '''
        list = []
        for i in range(0,x.size()[0]):
            list.append(self.predictone(x[i:i+1,:]))
        return list

    def outParams(self):
        '''
        这个神经网络的参数记录到文件中
        :return: None
        '''
        f = open("layers.pkl","wb")
        pickle.dump(self.layers_tensor,f)
        f.close()
        f = open("params.pkl","wb")
        pickle.dump(self.batchsize,f)
        pickle.dump(self.round, f)
        pickle.dump(self.learning_rate, f)
        f.close()

    def enterParams(self):
        '''
        从文件中读出已经训练好的网络的参数
        :return: None
        '''
        f = open("layers.pkl", "wb")
        self.layers_tensor = pickle.load(f)
        f.close()
        f = open("params.pkl", "wb")
        self.batchsize = pickle.load(f)
        self.round  = pickle.load(f)
        self.learning_rate = pickle.load(f)
        f.close()

    def show(self):
        '''
        显示参数
        :return: None
        '''
        print(self.layers_tensor)