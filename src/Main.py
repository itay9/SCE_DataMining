import pandas as pd

from src.NaiveBayes import naiveBayes
from src.Id3 import ID3_algorithm
from src.Id3SKlearn import ID3SKlearn_algorithm
from src.sklearnNaiveBayes import sklearnNaiveBayes

structFile= 'C:/Users/ChenAzulai/jupyter/Structure.txt'
trainFile="C:/Users/ChenAzulai/jupyter/train.csv"
testFile="C:/Users/ChenAzulai/jupyter/test.csv"
train = pd.read_csv(trainFile)
test = pd.read_csv(testFile)
naiveBayes(test,train,structFile)
sklearnNaiveBayes(test,train,structFile)
ID3_algorithm(train,test,structFile)
ID3SKlearn_algorithm(train,test,structFile)