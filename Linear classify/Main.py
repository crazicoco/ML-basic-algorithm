import numpy as np
import random
import math
class LDA():
    """
    the LDA experiment
    """
    def __init__(self, root):
        self.root = root
        self.singa_cla = []
        self.data = []
        self.label_name = []
    def load_data(self):
        """
        load data from root sent them to self.data and self.label
        :param root:
        :return:
        """
        with open(self.root, "r") as f:
            for line in f.readlines():
                data = line.split(",")
                label = data[-1].split("\n")[0]
                source = [float(data[i]) for i in range(len(data) - 1)]
                data = dict()
                data['source'] = source
                data['label'] = label
                self.data.append(data)
            return "load success"
    def split_dataset(self):
        """
        split the data pass the 10 zhe
        :return:
        """
        lens = len(self.data)
        type_label = []
        for i in range(lens):
            if self.data[i]['label'] not in type_label:
                type_label.append(self.data[i]['label'])
        self.label_name = type_label
        if len(type_label) == 2:
            label_1 = []
            label_2 = []
            for i in range(lens):
                if self.data[i]['label'] == type_label[0]:
                    label_1.append(self.data[i])
                else:
                    label_2.append(self.data[i])
            chunk_1 = int(len(label_1)/10)
            chunk_2 = int(len(label_2)/10)
            self.data = []
            for i in range(10):
                new_data = []
                data = label_1[i*chunk_1: (i+1)*chunk_1]
                # data = [label_1[j] for j in range(i*chunk_1,(i+1)*chunk_1-1)]
                # print(label_1[i])
                new_data.extend(data)
                # data = [label_2[j] for j in range(i*chunk_2,(i+1)*chunk_2-1)]
                data = label_2[i*chunk_2: (i+1)*chunk_2]
                new_data.extend(data)
                self.data.append(new_data)
            j = 0
            for i in range(chunk_1*10,len(label_1)):
                self.data[j].append(label_1[i])
                j = (j+1) % 10
            for i in range(chunk_2*10, len(label_2)):
                self.data[j].append(label_2[i])
                j = (j+1) % 10
            for i in range(10):
                random.shuffle(self.data[i])
            random.shuffle(self.data)
            return "split work is over if label is 2"
    def train(self, fuc):
        """
        make sure the parameter in the LDA algorithm
        :param data:
        :param label:
        :return:
        """
        train = []
        test = []
        precise = 0
        for i in range(10):
            test = self.data[i]
            for j in range(10):
                if j != i:
                    train.extend(self.data[j])
            if len(self.label_name) == 2:
                class_1 = [src for src in train if src['label'] == self.label_name[0]]
                class_2 = [src for src in train if src['label'] == self.label_name[1]]
                class_1_feat = [src['source'] for src in class_1]
                class_1_tgt = [src['label'] for src in class_1]
                class_2_feat = [src['source'] for src in class_2]
                class_2_tgt = [src['label'] for src in class_2]
                class_1_feat = np.array(class_1_feat)
                class_2_feat = np.array(class_2_feat)
                feat = []
                feat.extend(class_1_feat)
                feat.extend(class_2_feat)
                pai_1 = len(class_1) / len(train)
                pai_2 = len(class_2) / len(train)
                mean_1 = np.mean(class_1_feat, axis=0)
                mean_2 = np.mean(class_2_feat, axis=0)
                #the cov value in the class
                singa_1 = np.cov(class_1_feat, rowvar=False)
                singa_2 = np.cov(class_2_feat, rowvar=False)
                singa = np.cov(feat, rowvar=False)
                self.singa_lda = singa
                self.singa_cla = [singa_1, singa_2]
                self.mean = [mean_1, mean_2]
                self.pai = [pai_1, pai_2]
                if fuc == "LDA":
                    precise += self.predict_LDA(test)
                elif fuc == "QDA":
                    precise += self.predict_QDA(test)

        return precise / 10

    def predict_LDA(self, test):
        """
        give the test result
        :param target:
        :param label:
        :return:
        """
        lens = len(self.label_name)
        test_feat = [src['source'] for src in test]
        test_tgt = [src['label'] for src in test]
        test_feat = np.array(test_feat)
        log_ = math.log(self.pai[0] / self.pai[1])
        singa_lda_ = np.linalg.inv(self.singa_lda)
        mean_add = np.expand_dims(self.mean[0] + self.mean[1], 0)
        mean_sub = np.expand_dims(self.mean[0] - self.mean[1], 1)
        temp = np.matmul(mean_add, singa_lda_)
        temp = -1 * 0.5 * np.matmul(temp, mean_sub)
        logit = log_ + temp
        temp = np.matmul(test_feat, singa_lda_)
        temp = np.matmul(temp, mean_sub)
        logit = logit + temp
        tgt = [1 if i == "R" else -1 for i in test_tgt]
        tgt = np.array(tgt)
        logit = np.squeeze(logit, 1)
        logit = np.multiply(logit, tgt)
        count = 0
        for i in range(len(test)):
            if logit[i] > 0:
                 count += 1
        return count / len(test)

    def predict_QDA(self, test):
        """
        crruent version the precise is 0.44
        :param test:
        :return:
        """
        test_feat = [src['source'] for src in test]
        test_tgt = [src['label'] for src in test]
        test_feat = np.array(test_feat)
        log_pai1_ = math.log(self.pai[0])
        log_pai2_ = math.log(self.pai[1])
        singa_1_ = np.linalg.inv(self.singa_cla[0])
        singa_2_ = np.linalg.inv(self.singa_cla[1])
        # calculate the trace
        trace_1 = [self.singa_cla[0][i][i] for i in range(self.singa_cla[0].shape[0])]
        trace_1 = [math.pow(trace_, 0.5) for trace_ in trace_1]
        trace_value_1 = 1
        for tr in trace_1:
            trace_value_1 *= math.sqrt(tr)
        trace_2 = [self.singa_cla[1][i][i] for i in range(self.singa_cla[1].shape[0])]
        trace_2 = [math.pow(trace_, 0.5) for trace_ in trace_2]
        trace_value_2 = 1
        for tr in trace_2:
            trace_value_2 *= math.sqrt(tr)
        logit_trace = math.log(trace_value_1 / trace_value_2)
        # calculate the mean * singa * mean
        logit_msm_1 = np.matmul(self.mean[0], singa_1_)
        logit_msm_1 = np.matmul(logit_msm_1, self.mean[0])
        logit_msm_2 = np.matmul(self.mean[1], singa_2_)
        logit_msm_2 = np.matmul(logit_msm_2, self.mean[1])
        logit_msm = 0.5 * (logit_msm_2 - logit_msm_1)
        # calculate the x * singa * mean
        logit_xsm_1_ = np.matmul(singa_1_, self.mean[0])
        logit_xsm_2_ = np.matmul(singa_2_, self.mean[1])
        logit_xsm_ = logit_xsm_1_ - logit_xsm_2_
        logit_xsm = np.matmul(test_feat, logit_xsm_)
        logit = logit_trace + logit_xsm + logit_msm
        tgt = [1 if i == "R" else -1 for i in test_tgt]
        tgt = np.array(tgt)
        logit = np.squeeze(logit)
        logit = np.multiply(logit, tgt)
        count = 0
        for i in range(len(test)):
            if logit[i] > 0:
                count += 1
        return count / len(test)


def main():
    train_root = "../dataset/machine-learning-dataset/dataset/sonar_dataset二分类.txt"
    lda = LDA(root=train_root)
    # load data
    lda.load_data()
    lda.split_dataset()
    result = lda.train("QDA")
    print("the precise for the sonar_dataset task is {}".format(result))

if __name__ == "__main__":
    main()
