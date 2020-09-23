"""
task: MEM do the 2 classical task
model :MEM
dataset:sonar_dataset
最大熵不利用处理特征是连续数值的任务，最好是特征空间比较小的

下面给出几个已实现的MEM模型
OpenNLP : http://incubator.apache.org/opennlp/ 

Malouf: http://tadm.sourceforge.net/ 

Tsujii: http://www-tsujii.is.s.u-tokyo.ac.jp/~tsuruoka/maxent/

"""
from collections import defaultdict
import math
import random
from mpmath import *
import time
import numpy as np
import matplotlib.pyplot as plt


def initparams():
    return 0


class MEM(object):
    """
    IIS中的参数更新采用w一个一个的更新的方式
    """

    def __init__(self):
        self.feats = defaultdict(int)  # 特征函数组
        self.labels = set()  # 标签值域
        self.ep = []  # 特征函数f(x,y)模拟产生的p(x,y) = p(y|x) * p'(x)
        self.ep_ = []  # 特征函数f(x,y)关于经验分布P'(x,y)的期望值， 通过计算feature中频率产生
        self.M = None  # IIS特有的变量f#(x,y)= f1(x,y) + f2(x,y) + ... + fn(x,y)
        self.w = []  # 参数个数对应特征个数
        self.last_w = []  # 记录前一个参数的列表
        self.train_dataset = []  # 切分好的训练集
        self.test_dataset = []  # 切分好的测试集
        self.size = None  # 数据集的长度

    def load_data(self):
        """
        加载数据
        :return:
        """
        for idx, field in enumerate(self.train_dataset):
            if len(field) < 2:
                continue
            label = field[-1]
            self.labels.add(label)
            for j, f in enumerate(field[:-1]):
                self.feats[(label, str(j) + ',' + f)] += 1

    def dataset_split(self, name, rate, seed):
        """
        获取全部数据集，打乱
        划分数据集，按照rate的比例进行划分
        :param name:
        :param rate:
        :param seed:
        :return:
        """
        random.seed(seed)
        dataset = []
        for line in open(name):
            field = line.strip().split(',')
            dataset.append(field)
            random.shuffle(dataset)
        for field in dataset:
            if random.random() < rate:
                self.train_dataset.append(field)
            else:
                self.test_dataset.append(field)

    def initparams(self):
        """
        数据的初始化处理
        :return:
        """
        self.size = len(self.train_dataset)
        self.M = max([len(record)-1] for record in self.train_dataset)
        self.M = self.M[0]
        self.ep_ = [0.0] * len(self.feats)
        for idx, feat in enumerate(self.feats):
            #  计算特征关于经验分布P(x,y)的期望
            self.ep_[idx] = float(self.feats[feat]) / float(self.size)
            #  赋予特征以编号
            self.feats[feat] = idx
        self.w = [0.0] * len(self.feats)
        self.last_w = self.w

    def probwgt(self, features, label):
        """
        用来计算针对不同标签产生的概率
        :param features:
        :param label:
        :return:
        """
        rate = 0.0
        for idx, f in enumerate(features):
            # print(self.feats[label, idx, f])
            if (label, str(idx) + ',' + f) in self.feats:
                rate += self.w[self.feats[(label, str(idx) + ',' + f)]]
        return np.exp(rate)

    def Ep(self):
        """
        计算特征函数f(x,y)关于模型P(y|x)和经验分布P'(x)的期望值
        通过罗列每个循环的x，然后计算当前x对应的特征概率
        :return:
        """
        ep = [0.0] * len(self.feats)
        for record in self.train_dataset:
            features = record[:-1]
            # 计算 p(y|x)
            prob = self.calprob(features)
            for i, f in enumerate(features):
                for w, l in prob:
                    if (l, str(i) + ',' + f) in self.feats:
                        # get the feat id
                        idx = self.feats[(l, str(i) + ',' + f)]
                        # sum(1/N *f(y,x) * p(y|x)) p(x) = 1/N
                        ep[idx] += w * (1.0 / self.size)
        return ep

    def _convergence(self, last_w, w):
        """
        测试算法是否收敛, 通过计算差值是否小于0.01
        :param last_w:
        :param w:
        :return:
        """
        for w1, w2 in zip(last_w, w):
            if abs(w1 - w2) >= 0.005:
                return False
        return True

    def train(self, max_iter=10):
        """
        训练的主函数
        :param max_iter:
        :return:
        """
        self.initparams()
        for i in range(max_iter):
            print('iter %d ....' % (i + 1))
            self.ep = self.Ep()
            self.last_w = self.w[:]
            for idx, w in enumerate(self.w):
                delta = 1.0 / self.M * math.log(self.ep_[idx] / self.ep[idx])
                self.w[idx] += delta
            if self._convergence(self.last_w, self.w):
                break

    def calprob(self, features):
        """
        返回特征的概率
        :param features:
        :return:
        """
        rate = [(self.probwgt(features, l), l) for l in self.labels]
        Z = sum([w for w, l in rate])
        probability = [(w / Z, l) for w, l in rate]
        return probability

    def predict(self):
        """
        预测标签
        :param input:
        :return:
        """
        rate = 0
        for field in self.test_dataset:
            features = field[:-1]
            prob = self.calprob(features)
            prob.sort(reverse=True)
            _, label = prob[0]
            if label == field[-1]:
                rate +=1
        return rate / len(self.test_dataset)


def main():
    maxent = MEM()
    x =0.7
    res_p = 0
    split_rate = 0
    seed = 5
    for i in range(3):
        x += 0.1
        maxent.dataset_split('data/sonar_dataset.txt', x, seed)
        maxent.load_data()
        maxent.train(100)
        result = maxent.predict()
        if result > res_p:
            res_p = result
            split_rate = x
    print("seed:{seed}, split_rate:{split_rate},正确率: {res_p}".format(seed=seed, split_rate=split_rate, res_p=res_p))

if __name__ == '__main__':
    main()
