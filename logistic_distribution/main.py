"""
任务：逻辑斯蒂回归做二分类问题
迭代器：采用随机梯度下降法来进行迭代计算
数据:数据集采用sonar_dataset数据集
损失函数：逻辑斯蒂回归模型，采用最大化似然函数来作为损失函数
预测：二分类只需要求某个选项的概率。
但是这里注意采用的是正选项还是负选项的概率计算公式，
如果是负选项，即使是>0.5也是预测也是0

creater: crazicoco
time: 2020/9/17
参考：https://github.com/ice-tong/ML_demo/tree/master/Logistic
"""
from autograd import numpy as np
from autograd import grad
import random
import time
import matplotlib.pyplot as plt


class Logistic_distribution:
    """
    a class for Linear RegressionModels.
    """

    def __init__(self, max_iter=20000, alpha=0.001, C=0.01, epsilon=1e-4):
        self.max_iter = max_iter
        self.alpha = alpha  # learning rate
        self.C = C  # normalization
        self.epsilon = epsilon  # threshold value for stopping for
        self.Weights = None

    def _cost(self, weights, x, y):
        """
        when calculate the cost, the L2 norm will be added into the cost
        :param weights:
        :param x:
        :param y:
        :return:
        """
        eta = x @ weights
        l2_term = 0.5 * self.C * np.sum(weights ** 2)
        cost = l2_term - np.sum(y * eta - np.log(1 + np.exp(eta)))
        # cost = - np.sum(y * eta - np.log(1 + np.exp(eta)))
        return cost

    def _stocGradAscent(self, x_, y_, batch_size=10):
        weights_history = np.zeros([100, self.Weights.shape[0], 1])  # 只保留前100的参数
        self.sgd_cost_history = []
        self.sgd_iters = []
        self.sgd_timers = []
        t = time.time()
        grad_cost = grad(self._cost)
        for i in range(self.max_iter):
            random_index = np.random.randint(0, x_.shape[0] - batch_size)
            x = x_[random_index: random_index + batch_size]
            y = y_[random_index: random_index + batch_size]
            diff = np.mean(abs(self.Weights - weights_history[i % 100, :]))
            if diff < self.epsilon:
                break
            weights_history[i % 100] = self.Weights
            self.Weights = self.Weights - self.alpha * grad_cost(self.Weights, x, y)
            cost = (self.N / batch_size) * self._cost(self.Weights, x, y)
            if i % 100 == 0:
                print("L(w, w0) values now:", cost)
            self.sgd_cost_history.append(cost)
            self.sgd_iters.append(i)
            self.sgd_timers.append(time.time() - t)
        return self.Weights

    def fit(self, x, y, solver="SGD"):
        x_ = np.insert(x, 0, values=np.ones(x.shape[0]), axis=1)  # 在特征空间的最前面一列添加一列1
        y_ = np.reshape(y, [-1, 1])  #
        self.N = x_.shape[0]  # 保存数据长度
        self.Weights = np.ones([x_.shape[1], 1])  # 初始化参数列表
        if solver == 'SGD':
            self._stocGradAscent(x_, y_)

    def predict(self, x):
        x_ = np.insert(x, 0, values=np.ones(x.shape[0]), axis=1)
        y_pred = np.zeros(x_.shape[0])
        # h = 1.0 / (1 + np.exp((x_ @ self.Weights)))
        h = np.exp(x_ @ self.Weights)*1.0 / (1 + np.exp((x_ @ self.Weights)))
        for i in range(h.shape[0]):
            if h[i] >= 0.5:
                y_pred[i] = 1
            else:
                y_pred[i] = 0
        return y_pred


def load_data(name):
    """
    加载数据
    Note: when you calculate the weight by the autograd, you need to translate the label to be digit
    :param name:文件地址
    :return: 数据和标签
    """
    inputs = []
    outputs = []
    with open(name, "r") as f:
        result = f.readline()
        while result:
            result = result.split(",")
            input = result[:-1]
            input = [float(i) for i in input]
            output = result[-1].split("\n")
            inputs.append(input)
            outputs.append(output)
            result = f.readline()
    for i in range(len(outputs)):
        outputs[i] = outputs[i][0]
        if outputs[i] == 'R':
            outputs[i] = 1  # R
        else:
            outputs[i] = 0  # M
    inputs = np.array(inputs)
    outputs = np.array(outputs)
    return inputs, outputs


def train_split(feature_data, label_data):
    """
    follow set aside method to split the train_set
    the rate is 7:3
    :param feature_data: 传进来的特征数据
    :param label_data:  传进来的标签数据
    :return: 返回训练数据集，测试数据集
    """
    train_features = []
    train_labels = []
    test_features = []
    test_labels = []
    for feature, label in zip(feature_data, label_data):
        if random.random() < 0.7:
            train_features.append(feature)
            train_labels.append(label)
        else:
            test_labels.append(label)
            test_features.append(feature)
    train_features = np.array(train_features)
    train_labels = np.array(train_labels)
    test_features = np.array(test_features)
    test_labels = np.array(test_labels)
    return train_features, train_labels, test_features, test_labels


def evaluate(y_test, y_pred):
    error_rate = (np.sum(np.equal(y_test, y_pred).astype(np.float)) / y_test.size)
    return error_rate


def main():
    name = 'data/sonar_dataset.txt'
    name_test = 'data/test.txt'
    load_data(name=name)
    feature_data, label_data = load_data(name)
    test_feature, test_label = load_data(name_test)
    (train_features, train_labels, valid_features, valid_labels) = train_split(feature_data, label_data)
    logistic = Logistic_distribution()
    logistic.fit(train_features, train_labels, solver='SGD')
    pred = logistic.predict(valid_features)
    error_rate_1 = evaluate(pred, valid_labels)
    pred = logistic.predict(test_feature)
    error_rate = evaluate(pred, test_label)
    print("valid_rate:", error_rate_1)
    print("test_rate", error_rate)
    logistic.fit(train_features, train_labels, solver="")
    plt.plot(logistic.sgd_iters, logistic.sgd_cost_history, label="SGD")
    plt.legend()
    plt.savefig("cost-iter.png")
    plt.show()
    plt.close()

    plt.plot(logistic.sgd_timers, logistic.sgd_cost_history, label="SGD")
    plt.legend()
    plt.savefig("cost-timer.png")
    plt.show()


if __name__ == "__main__":
    main()
