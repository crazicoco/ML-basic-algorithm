from sklearn import linear_model
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt

"""
    误差更低
"""


def load_file(root):
    with open(root, "r") as f:
        head = f.readline()
        Year, Salary = head.split(",")
        data_str = f.readlines()

    data_year = []
    data_salary = []
    for idx, str in enumerate(data_str):
        str_year, str_salary = str.split(",")
        x_sample = float(str_year)
        data_year.append(x_sample)
        str_salary, _ = str_salary.split("\n")
        data_salary.append(float(str_salary))
    assert len(data_year) == len(data_salary)
    return data_year, data_salary


# 显示当前的数据和迭代后损失关系
def plot_X_Y_1(X, y):
    plt.scatter(X, y)
    plt.show()


def plot_X_Y_2(X, y, theta):
    plt.scatter(X, y)
    x = np.linspace(1, 10, 12)
    g = theta[0] * x + theta[1]
    plt.plot(x, g)
    plt.show()


def split_train_test(X, Y):
    print("划分数据集为80%作为训练数据集，20%作为测试数据集")
    X_train = X[:int(len(X) * 0.8)]
    Y_train = Y[:int(len(X) * 0.8)]
    X_test = X[int(len(X) * 0.8):]
    Y_test = Y[int(len(X) * 0.8):]
    return X_train, Y_train, X_test, Y_test


def main():
    print("加载数据中......")
    data_dir = "data\\Salary_Data.csv"
    data_year, data_salary = load_file(data_dir)
    # 本数据y对应薪水不适宜归一化
    # print("对数据进行归一化")
    print("扩充x一列1")
    X = list([year, 1] for year in data_year)
    Y = data_salary
    X_train, Y_train, X_test, Y_test = split_train_test(X, Y)
    theta = np.ones(2)
    model = linear_model.LinearRegression()
    model.fit(X_train, Y_train)
    theta = [model.coef_[0], model.intercept_]
    plot_X_Y_2(data_year, data_salary, theta)
    sum = 0

    for s, r in zip(Y_test, model.predict(X_test)):
        sum += s - r
    print(sum)


if __name__ == "__main__":
    main()