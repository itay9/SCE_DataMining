import pandas as pd
import sys
from NaiveBayes import naiveBayes
from KNNClassifier import KNNClassifier
from id3 import ID3_algorithm
from id3SKlearn import ID3SKlearn_algorithm
from sklearnNaiveBayes import sklearnNaiveBayes


"""#command line interface (CLI)
path=sys.argv[1]
structFile= path+'/Structure.txt'
trainFile=path+'/train.csv'
testFile=path+'/test.csv'
"C:/Users/ChenAzulai/jupyter"
"""
structFile= 'C:/Users/ChenAzulai/jupyter/Structure.txt'
trainFile="C:/Users/ChenAzulai/jupyter/train.csv"
testFile="C:/Users/ChenAzulai/jupyter/test.csv"

train = pd.read_csv(trainFile)
test = pd.read_csv(testFile)
naiveBayes(test,train,structFile)
sklearnNaiveBayes(test,train,structFile)
ID3_algorithm(train,test,structFile)
ID3SKlearn_algorithm(train,test,structFile)
KNNClassifier(train,structFile)
