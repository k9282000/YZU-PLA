import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import time
import random
import math

class Perceptron(object):
    
    def __init__(self):
        self.max_iteration = 5000
        self.targetLabel = 'Survived'
        self.pocketWeight = None
        self.pocketWeight_errNum = 0
    
    def train(self, features, labels):
        self.weight = [0.0] * (len(features.columns)+1) # weight init
        
        self.featureNameList = features.columns.to_list()+['x0'] # feature name append x0
        
        time = 0
        total_row_count = features.index.size
        self.pocketWeight_errNum = features.index.size

        while time < self.max_iteration:
            time += 1
            # print('train ',time,' times')
            correct_count = 0
            for index , feature in features.iterrows():
                feature = feature.append( pd.Series([1.0], ['x0']))
                y = labels.loc[index,self.targetLabel]
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
                    if index < self.pocketWeight_errNum:
                        self.pocketWeight_errNum = index
                        self.pocketWeight = self.weight

                    if y<=0:
                        y = -1

                    for fName , x in feature.iteritems():
                        self.weight[self.featureNameList.index(fName)] += y * x
                    break
                else:
                    correct_count += 1
            
            if correct_count == total_row_count:
                print ('all pass')
                break
        print(time,self.max_iteration)
        if time == self.max_iteration:
            self.weight = self.pocketWeight
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
        wx = sum([self.weight[j] * x[j] for j in range(len(self.weight))])
        return int(wx > 0)

if __name__ == '__main__': #模組名稱

    #----------------EXAMPLE---------------
    # raw_data = pd.read_csv('./data/example.csv', header=0)
    # # print(raw_data)
    # train_data = raw_data[['str']]
    # target_data = raw_data[['isStr']]
    # p = Perceptron()
    # p.targetLabel = 'isStr'
    # p.train(train_data, target_data)
    #--------------------------------------
    
    
    
    # print ('Start read data')
    # time_1 = time.time()

    # get trainning data from csv file
    raw_data = pd.read_csv('./data/train.csv', header=0)
    
    # extract target value
    target_data = raw_data[['Survived']]

    # Sex trans to sex_code 1/0
    raw_data['sex_code'] = raw_data['Sex'].map({'female':1,'male':0}).astype('int')

    # get average of Age
    age_mean = int(raw_data['Age'].mean())

    # replace nan to average in age
    raw_data['Age'] = raw_data['Age'].transform(lambda x: age_mean if math.isnan(x) else x)

    # filter feature
    train_data = raw_data[['sex_code','Pclass','Age','SibSp','Parch','Fare']]
   
    # time_2 = time.time()
    # print ('read data cost ', time_2 - time_1, ' second')
        
    # print ('Start training')
    p = Perceptron()
    p.train(train_data, target_data)

    # time_3 = time.time()
    # print ('training cost ', time_3 - time_2, ' second', '\n')

    test_predict = p.predict(test_features)