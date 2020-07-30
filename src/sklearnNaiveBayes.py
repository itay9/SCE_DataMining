from src.functions import getColumnTitles, Discretize, fit_transforms
from sklearn.naive_bayes import GaussianNB
from sklearn import preprocessing
numOfBins=3
def sklearnNaiveBayes(test, train,structure):
    model=GaussianNB()
    le=preprocessing.LabelEncoder()

    train = Discretize(numOfBins, train, structure)
    test = Discretize(numOfBins, test, structure)

    target_train=train['class']
    inputs_train=train.drop('class',axis='columns')

    target_test=test['class']
    inputs_test=test.drop('class',axis='columns')

    inputs_train=fit_transforms(inputs_train)
    model.fit(inputs_train,target_train)
    inputs_test=fit_transforms(inputs_test)
    print("sklearnNaiveBayes accuracy:",model.score(inputs_test,target_test),"%")