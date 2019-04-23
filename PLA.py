import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import time
import random
import math

class Perceptron(object):
    
    def __init__(self):
        self.learning_step = 0.00001
        self.max_iteration = 5000
    
    def train(self, features, labels):
        self.weight = [0.0] * (len(features.columns)+1) # weight init
        
        self.featureNameList = features.columns.to_list()+['x0'] # feature name append x0
        
        # each row to calculate y
        for index , feature in features.iterrows():
            # index = row index of features
            # x = row
            feature = feature.append( pd.Series([1.0], ['x0']))
            y = labels.loc[index,'Survived']
            wx = 0

            for fName , x in feature.iteritems():
                w = self.weight[self.featureNameList.index(fName)]
                if math.isnan(x) : 
                    x = 0
                wx = wx + (x * w)
                
                # print(fName , ' = ' ,x , 'and weight = ', self.weight[self.featureNameList.index(fName)],' val=' ,wx)
            
            # print('wx = ',np.sign(wx),'& y = ',y)
            if wx<=0:
                predict_sign = 0
            else:
                predict_sign = 1

            if predict_sign != y :
                for fName , x in feature.iteritems():
                    self.weight[self.featureNameList.index(fName)] = y * x
                break
            else:
                print('ok !')
            if index == 5 :
                break

        # while time < self.max_iteration:
        #     index = random.randint(0, len(labels) - 1)
        #     print('index', index)
        #     x = list(features[index])
        #     x.append(1.0)
        #     print('x features', x)
        #     y = 2 * labels[index] - 1
        #     wx = sum([self.w[j] * x[j] for j in range(len(self.w))])

        #     if wx * y > 0:
        #         correct_count += 1
        #         if correct_count > self.max_iteration:
        #             break
        #         continue

        #     for i in range(len(self.w)):
        #         self.w[i] += self.learning_step * (y * x[i])
        
        #print(self.w)

    def predict(self,features):
        labels = []
        for feature in features:
            x = list(feature)
            x.append(1)
            labels.append(self.predict_(x))
        return labels

    def predict_(self, x):
        wx = sum([self.w[j] * x[j] for j in range(len(self.w))])
        return int(wx > 0)

if __name__ == '__main__': #模組名稱

    print ('Start read data')
    time_1 = time.time()

    raw_data = pd.read_csv('./data/train.csv', header=0)

    # train_data = raw_data[['Pclass','Name','Sex','Age']]
    train_data = raw_data[['Pclass','Age','SibSp','Parch']]
    label_data = raw_data[['Survived']]

    # time_2 = time.time()
    # print ('read data cost ', time_2 - time_1, ' second', '\n')
        
    print ('Start training')
    p = Perceptron()
    p.train(train_data, label_data)