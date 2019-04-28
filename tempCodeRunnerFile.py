a = pd.read_csv('./data/test.csv', header=0)
    

    # # data preprocee    
    # test_data = preProcess(test_data)
    # test_predict = p.predict(test_data) 

    # test_labels = pd.read_csv('./data/gender_submission.csv', header=0)
    # test_labels = [x[0] for x in test_labels[['Survived']].values.tolist()]
    # score = accuracy_score(test_labels, test_predict)

    # print(score)
    # result = {'PassengerId': range(892,1310), 'Survived': test_predict}
    # result = pd.DataFrame(data=result)
    # # print(result)
    # # print(range(10))
    # # print(len(test_predict))
    # # print(len(range(892,1310)))
    # result.to_csv('dew