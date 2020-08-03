from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import preprocessing
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from functions import getColumnTitles, Discretize, valuesType, pArrayByFeature, fit_transforms
from sklearn.feature_extraction import DictVectorizer

from functions import getColumnTitles, Discretize

numOfBins = 3

def numericCol(table, structureTextFile):
    structure = pd.read_csv(structureTextFile, sep=" ", names=['type', 'feature', 'data'])
    column = []
    headers = getColumnTitles(table)
    for i in range(structure.shape[0]):
        if 'NUMERIC' in structure.loc[i]['data']:
            column += [headers[i]]
    return column

def Encode(

        train,Structure):
    # creating labelEncoder
    le = preprocessing.LabelEncoder()
    # Converting string labels into numbers.
    numeri_col = numericCol(train, Structure)
    column = []
    features2 = {}
    for i in range(len(train.columns)):
        weather_encoded = le.fit_transform(train[train.columns[i]])
        features2.update({train.columns[i]: weather_encoded.tolist()})
    df = pd.DataFrame(features2, columns=getColumnTitles(train))
    return df


def loadFile(path):
    # Load Excel File into Pandas DataFrame
    df = pd.read_csv(path)
    return df

def feature(feature, df):
    # Create arrays for the features and the response variable
    y = df[feature]
    x = df.drop(feature, axis=1)
    return x, y

def TestTrainFitPlot(train, test):
    # Setup arrays to store train and test accuracies
    # Split into training and test set
    neighbors = np.arange(1, 20)
    train_accuracy = np.empty(len(neighbors))
    test_accuracy = np.empty(len(neighbors))
    target_train = train['class']
    inputs_train = train.drop('class', axis='columns')
    target_test = test['class']
    inputs_test = test.drop('class', axis='columns')
    inputs_train = fit_transforms(inputs_train)
    inputs_test = fit_transforms(inputs_test)
    knn = KNeighborsClassifier()
    knn.fit(inputs_train, target_train)
    # save model to file
    filename = 'KNN_model.sav'
    joblib.dump(knn, open(filename, 'wb'))

    # Check Accuracy Score
    print('KNN Accuracy: {}'.format(round(knn.score(inputs_test, target_test), 3)))
    # Enum Loop, accuracy results using range on 'n' values for KNN Classifier
    for acc, n in enumerate(neighbors):
        # Try KNeighbors with each of 'n' neighbors
        knn = KNeighborsClassifier(n_neighbors=n)
        knn.fit(inputs_train, target_train)
        # Training Accuracy
        train_accuracy[acc] = knn.score(inputs_train, target_train)
        # Testing Accuracy
        test_accuracy[acc] = knn.score(inputs_test, target_test)
    # set plot style
    plt.style.use('ggplot')
    plt.title('KNN Neighbors')
    # Set X-Axis Label
    plt.xlabel('Neighbors\n(#)')
    # Set Y-Axis Label
    plt.ylabel('Accuracy\n(%)', rotation=0, labelpad=35)
    # Place Testing Accuracy
    plt.plot(neighbors, test_accuracy, label='Testing Accuracy')
    # Place Training Accuracy
    plt.plot(neighbors, train_accuracy, label='Training Accuracy')
    # Append Labels on Testing Accuracy
    for a, b in zip(neighbors, test_accuracy):
        plt.text(a, b, str(round(b, 2)))
    # Add Legend
    plt.legend()
    # Generate Plot
    plt.show()
    # Plotting
    # Set Main Title

def KNNClassifier(test,train, structFile):
    encode = Encode(train,structFile)
    encode_ = Encode(test,structFile)
    x = feature("class",encode)
    y = feature("class",encode_)
    TestTrainFitPlot(train,test)




