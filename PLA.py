import numpy as np  # 線性代數涵式庫
import pandas as pd # 數據分析涵式庫
import math # 數學函數
import os 
from sklearn.metrics import accuracy_score #機器學習涵式庫

class Perceptron(object):
    
    def __init__(self):
        self.max_iteration = 5000          # 最大訓練次數
        self.pocketWeight = None        # pocket weight
        self.pocketWeight_accuracy = 0  # pocket weight 的 準確率
    
    ''' 預測標籤
        參數
            self
            sampleData     樣本資料      Series
        回傳
            label 預測標籤 int
        '''
    def predict_label(self,sampleData):
        wx = 0
        # 累計 feature 權重
        for feature , x in sampleData.iteritems():
            w = self.weight[self.featureNameList.index(feature)]
            # x = 0 if math.isnan(x) else x 
            wx += x * w
        # 預測結果
        label = 1 if np.sign(wx)==1 else 0
        return label

    ''' 訓練函數
        參數
            self
            data     樣本      DataFrame
            labels   樣本標籤  DataFrame
        '''
    def train(self, data, labels):
        # 初始化 weight 為 0
        self.weight = [0.0] * (len(data.columns)+1) 
        # 設定 feature 名稱清單        
        self.featureNameList = ['x0']+data.columns.to_list()
        # 訓練次數
        trainTime = 0
        # 初始化 pocketWeight_errCount，預設為全資料數
        self.pocketWeight_accuracy = 0

        # 訓練迴圈
        while trainTime < self.max_iteration:
            trainTime += 1      # 訓練次數計數
            correct_count = 0   # 本次訓練的正確筆數
            print('train ',trainTime,' times , weight = ',self.weight)
            # feature 迴圈，計算
            for index , row in data.iterrows():
                # 將 feature 套上 x0，固定為1
                row = pd.Series([1.0], ['x0']).append(row)
                
                y = labels.loc[index,'Survived'] # 取出對應的 label 為 y
                label = self.predict_label(row)

                # Error Point 
                if label != y :
                    # 取得本次Error weight 的準確率
                    data_predict = self.predict(data)
                    train_labels = [x[0] for x in labels[['Survived']].values.tolist()]
                    score = accuracy_score(train_labels, data_predict)

                    # 檢測是否替換 pocket weight
                    if score > self.pocketWeight_accuracy:
                        print('change pocket weight')
                        self.pocketWeight_accuracy = score
                        self.pocketWeight = list(self.weight)

                    label_sign = -1 if y==0 else 1
                    
                    # 計算新權重
                    for feature , x in row.iteritems():
                        self.weight[self.featureNameList.index(feature)] += label_sign * x
                    break # Error 結束舊權重訓練
                else:
                    correct_count += 1 # Pass Counter
            
            if correct_count == data.index.size:
                print ('all pass')
                break
        print ('PLA weight = ',self.weight)
        print ('pocket weight = ',self.pocketWeight)
        # 訓練結束原因判斷，是否取用 Pocket Weight
        if trainTime == self.max_iteration:
            self.weight = self.pocketWeight
            print('use  pocket weight')
        
        

    ''' 預測函數
        參數
            self
            data     樣本      DataFrame
        回傳
            labels 預測標籤 list
        '''
    def predict(self,data):
        labels = []
        for row in data.iterrows():
            row = row[1]
            # 將 feature 套上 x0，固定為1
            row = pd.Series([1.0], ['x0']).append(row)
            labels.append(self.predict_label(row))
        return labels
    
def preProcess(data):
    
    # Sex trans to sex_code 1/0
    data['sex_code'] = data['Sex'].map({'female':1,'male':0}).astype('int')

    # replace nan to average in age
    data["Age"] = data["Age"].fillna(data["Age"].median())
    data["Fare"] = data["Fare"].fillna(data["Fare"].median())
    data["Embarked"] = data["Embarked"].fillna('Q')

    data['hasFamily'] =( data['SibSp']+data['Parch']>0 ).map({True:1,False:0})
    # data["Embarked"] = data["Embarked"].apply(lambda x:1 if x =="C" else 0)
    data['Embarked'] = data['Embarked'].map({'C':2,'Q':1,'S':0}).astype('int')


    # filter feature
    return  data[['sex_code','Pclass','Age','hasFamily','Fare','Embarked']]

if __name__ == '__main__': #模組名稱

    # get trainning data from csv file
    raw_data = pd.read_csv('./data/train.csv', header=0)
    
    # extract target value
    target_data = raw_data[['Survived']]

    # data preprocee    
    train_data = preProcess(raw_data)
    # os._exit(0)
    p = Perceptron()
    p.train(train_data, target_data)
    
    test_data = pd.read_csv('./data/test.csv', header=0)
    

    # data preprocee    
    test_data = preProcess(test_data)
    test_predict = p.predict(test_data) 
    print('predict weight = ',p.weight)

    test_labels = pd.read_csv('./data/gender_submission.csv', header=0)
    test_labels = [x[0] for x in test_labels[['Survived']].values.tolist()]
    score = accuracy_score(test_labels, test_predict)

    print(score)
    result = {'PassengerId': range(892,1310), 'Survived': test_predict}
    result = pd.DataFrame(data=result)
    result.to_csv('dewei_predict_20190429_03.csv', sep=',',index=False)
    