"""
    algorithm:实现软间隔化的核函数SVM二分类算法
    reference:http://github.com/ajtulloch/svmpy
    Note:上面参考地址给出了基于凸优化库cvxopt的svm实现版本，代码写的很棒，但是对于cvxopt库了解不多，后面如果有机会希望重写一下这个算法，相信会加深理解的。
    采用slearn实现svm
    dataset：lonosphere_dataset.txt
    author：crazicoco
"""
from sklearn.svm import SVC
# sklearn.preprocessing 提供三种数据标准化的处理方式，StandardScaler, MinMaxScaler, RobusScaler。防止因为异常点影响结果
from sklearn.preprocessing import StandardScaler
# 网格搜索，在参数列表中穷举搜索找到最优的结果，为了减少偶然性，会结合交叉验证的方法进行
from sklearn.model_selection import GridSearchCV, train_test_split
import random
import numpy as np


def load_data(address):
    features = []
    labels = []
    for line in open(address):
        field = line.strip().split(',')
        feature = field[:-1]
        label = field[-1]
        features.append(feature)
        labels.append(label)
    scaler = StandardScaler()
    # 标准化
    features = scaler.fit_transform(features)
    return features, labels


def split_data(features, labels):
    random.seed(10)
    train_features = []
    train_labels = []
    test_features = []
    test_labels = []
    for i, (feature, label) in enumerate(zip(features, labels)):
        digit = random.random()
        if digit < 0.9:
            train_features.append(feature)
            train_labels.append(label)
        else:
            test_features.append(feature)
            test_labels.append(label)
    return train_features, train_labels, test_features, test_labels


def main():
    name = "data/lonosphere_dataset.txt"
    features, label = load_data(address=name)
    train_features, train_labels, test_features, test_labels = split_data(features=features, labels=label)
    model = SVC(kernel='linear')
    # 参考参数给出是等比数列的形式
    c_range = np.logspace(-5, 15, 11, base=2)
    gamma_range = np.logspace(-9, 3, 13, base=2)
    param_grid = [{'kernel': ['linear'], 'C':c_range, 'gamma':gamma_range}]
    grid = GridSearchCV(model, param_grid=param_grid, cv=3, n_jobs=-1)
    clf = grid.fit(train_features, train_labels)
    score = grid.score(test_features, test_labels)
    print('精度为%s' % score)


if __name__ == "__main__":
    main()