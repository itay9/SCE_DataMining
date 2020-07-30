from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import preprocessing
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from src.functions import getColumnTitles, Discretize, valuesType, pArrayByFeature
from sklearn.feature_extraction import DictVectorizer

from src.functions import getColumnTitles, Discretize

numOfBins = 3

def numericCol(table, structureTextFile):
    structure = pd.read_csv(structureTextFile, sep=" ", names=['type', 'feature', 'data'])
    column = []
    headers = getColumnTitles(table)
    for i in range(structure.shape[0]):
        if 'NUMERIC' in structure.loc[i]['data']:
            column += [headers[i]]
    return column

def Encode(train,Structure):
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

def TestTrainFitPlot(X, y):
    # Setup arrays to store train and test accuracies
    # Split into training and test set
    neighbors = np.arange(1, 20)
    train_accuracy = np.empty(len(neighbors))
    test_accuracy = np.empty(len(neighbors))
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)
    # Try KNN with another neighbors
    knn = KNeighborsClassifier()
    # Fit training data
    knn.fit(X_train, y_train)
    # Check Accuracy Score
    print('KNN Accuracy: {}'.format(round(knn.score(X_test, y_test), 3)))
    # Enum Loop, accuracy results using range on 'n' values for KNN Classifier
    for acc, n in enumerate(neighbors):
        # Try KNeighbors with each of 'n' neighbors
        knn = KNeighborsClassifier(n_neighbors=n)
        knn.fit(X_train, y_train)
        # Training Accuracy
        train_accuracy[acc] = knn.score(X_train, y_train)
        # Testing Accuracy
        test_accuracy[acc] = knn.score(X_test, y_test)
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

def KNNClassifier(train2, structFile):
    train = Discretize(numOfBins, train2, structFile)
    encode = Encode(train,structFile)
    x, y = feature("class",encode)
    TestTrainFitPlot(x, y)




