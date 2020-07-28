import joblib
from sklearn.tree import DecisionTreeClassifier
from src.Functions import Discretize,fit_transforms

numOfBins=3
def ID3SKlearn_algorithm(train,test,structure):
    train = Discretize(numOfBins, train, structure)
    test = Discretize(numOfBins, test, structure)

    train=fit_transforms(train)
    train_target= train['class']
    train_feature=train.drop('class', axis='columns')

    test=fit_transforms(test)
    test_target= test['class']
    test_feature=test.drop('class', axis='columns')

    tree=DecisionTreeClassifier(criterion='entropy',max_depth=100).fit(train_feature,train_target)

    prediction = tree.predict(test_feature)

    #save model to file
    filename='ID3SKlearn_model.sav'
    joblib.dump(tree,filename)

    print("ID3SKlearn_algorithm accuracy is: ",tree.score(test_feature,test_target)*100,"%")
