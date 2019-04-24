import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import time
import random
import math

class Perceptron(object):
    
    def __init__(self):
        self.max_iteration = 5000
    
    def train(self, features, labels):
        self.weight = [0.0] * (len(features.columns)+1) # weight init
        
        self.featureNameList = features.columns.to_list()+['x0'] # feature name append x0
        
        time = 0;
        total_row_count = features.index.size

        while time < self.max_iteration:
            time += 1
            # print('train ',time,' times')
            correct_count = 0
            for index , feature in features.iterrows():
                feature = feature.append( pd.Series([1.0], ['x0']))
                y = labels.loc[index,'Survived']
                wx = 0

                for fName , x in feature.iteritems():
                    w = self.weight[self.featureNameList.index(fName)]
                    if math.isnan(x) : 
                        x = 0
                    wx = wx + (x * w)
                    
                if wx<=0:
                    predict_sign = 0
                else:
                    predict_sign = 1

                if predict_sign != y :
                    # print('Wt = ',self.weight)
                    for fName , x in feature.iteritems():
                        self.weight[self.featureNameList.index(fName)] += y * x
                    # print('Wt+1 = ',self.weight)
                    break;
                else:
                    correct_count += 1
            
            if correct_count == total_row_count:
                print ('all pass')
                break
        
        if time == self.max_iteration:
            print ('no perfect weight')
        else:
            print ('perfect weight')
        print ('final weight = ',self.weight)

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

    # print ('Start read data')
    # time_1 = time.time()

    raw_data = pd.read_csv('./data/train.csv', header=0)
    
    target_data = raw_data[['Survived']]

    raw_data.drop('Survived', 1, inplace=True)

    raw_data['sex_code'] = raw_data['Sex'].map({'female':1,'male':0}).astype('int')

    # print(type(raw_data['Sex']))
    # print(raw_data.head())
    # print(raw_data.info())
    
    train_data = raw_data[['sex_code','Pclass','Age','SibSp','Parch','Fare']]
    print(target_data.head())

    # time_2 = time.time()
    # print ('read data cost ', time_2 - time_1, ' second')
        
    # print ('Start training')
    p = Perceptron()
    p.train(train_data, target_data)

    # time_3 = time.time()
    # print ('training cost ', time_3 - time_2, ' second', '\n')