# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

# script copy from https://github.com/WenDesi/lihang_book_algorithm/blob/master/perceptron/binary_perceptron.py

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
# print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.

import cv2
import random
import time

from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score

class Perceptron(object):
    
    def __init__(self):
        self.learning_step = 0.00001
        self.max_iteration = 5000
        
    def predict_(self, x):
        wx = sum([self.w[j] * x[j] for j in range(len(self.w))])
        return int(wx > 0)
    
    def train(self, features, labels):
        self.w = [0.0] * (len(features[0]) + 1)
        print(len(self.w))
        print(len(features[0]))
        print(len(features))
        correct_count = 0
        time = 0

        while time < self.max_iteration:
            index = random.randint(0, len(labels) - 1)
            print('index', index)
            x = list(features[index])
            x.append(1.0)
            y = 2 * labels[index] - 1
            wx = sum([self.w[j] * x[j] for j in range(len(self.w))])

            if wx * y > 0:
                correct_count += 1
                if correct_count > self.max_iteration:
                    break
                continue

            for i in range(len(self.w)):
                self.w[i] += self.learning_step * (y * x[i])
        
        #print(self.w)

    def predict(self,features):
        labels = []
        for feature in features:
            x = list(feature)
            x.append(1)
            labels.append(self.predict_(x))
        return labels


if __name__ == '__main__':

    print ('Start read data')

    time_1 = time.time()

    raw_data = pd.read_csv('../input/train_binary.csv', header=0)
    data = raw_data.values

    imgs = data[0::, 1::]
    labels = data[::, 0]

    # 选取 2/3 数据作为训练集， 1/3 数据作为测试集
    train_features, test_features, train_labels, test_labels = train_test_split(
        imgs, labels, test_size=0.33, random_state=23323)
    # print train_features.shape
    # print train_features.shape

    time_2 = time.time()
    print ('read data cost ', time_2 - time_1, ' second', '\n')

    print ('Start training')
    p = Perceptron()
    p.train(train_features, train_labels)

    time_3 = time.time()
    print ('training cost ', time_3 - time_2, ' second', '\n')

    print ('Start predicting')
    test_predict = p.predict(test_features)
    time_4 = time.time()
    print ('predicting cost ', time_4 - time_3, ' second', '\n')

    score = accuracy_score(test_labels, test_predict)
    print ("The accruacy socre is ", score)




