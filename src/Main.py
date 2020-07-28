import pandas as pd
<<<<<<< HEAD:src/main.py
from NaiveBayes import naiveBayes
from KNNClassifier import KNNClassifier
from id3 import ID3_algorithm
from id3SKlearn import ID3SKlearn_algorithm
from sklearnNaiveBayes import sklearnNaiveBayes
=======

from src.NaiveBayes import naiveBayes
from src.Id3 import ID3_algorithm
from src.Id3SKlearn import ID3SKlearn_algorithm
from src.sklearnNaiveBayes import sklearnNaiveBayes
>>>>>>> b676549612d4d9f3911b38572185d713f4179f1a:src/Main.py

#structFile= 'C:/Users/ChenAzulai/jupyter/Structure.txt'
#trainFile="C:/Users/ChenAzulai/jupyter/train.csv"
#testFile="C:/Users/ChenAzulai/jupyter/test.csv"
structFile= '/Users/davidteboul/Documents/כריית נתונים /Projet/Structure.txt'
trainFile="/Users/davidteboul/Documents/כריית נתונים /Projet/train.csv"
testFile="/Users/davidteboul/Documents/כריית נתונים /Projet/test.csv"
train = pd.read_csv(trainFile)
test = pd.read_csv(testFile)
naiveBayes(test,train,structFile)
sklearnNaiveBayes(test,train,structFile)
ID3_algorithm(train,test,structFile)
ID3SKlearn_algorithm(train,test,structFile)
KNNClassifier(train,structFile)
