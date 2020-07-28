import pandas as pd

from functions import getColumnTitles, Discretize, valuesType, pArrayByFeature


# column=['campaign','previous','age','balance','day','duration']
numOfBins = 3



def allArraysOfFetures(table, classCol):  # dict, keys is tuple(column,'yes'/'no'),values is list [p(columnValue|class),...]
    thisDict = {}
    for i in getColumnTitles(table):
        if i not in classCol:
            for j in valuesType(table, classCol):
                thisDict[i, j] = pArrayByFeature(table, i, j, classCol)
                # print(pArrayByFeature(train,i,j,classCol))
    return thisDict

#thisDict = allArraysOfFetures(train, 'class')

def naiveBayes(test, train,structure):
    train = Discretize(numOfBins, train, structure)
    test = Discretize(numOfBins, test, structure)
    thisDict=allArraysOfFetures(train, 'class')
    rows = test.shape[0]
    classMatch = 0
    classDismatch = 0

    column = getColumnTitles(test)[:-1]  # clean 'class' column
    for _ in range(rows):

        noPar = 1
        yesPar = 1
        for col in column:
            try:
                index = valuesType(train, col).index(test.iloc[_][col])
                yesPar *= thisDict[(col, 'yes')][index]
                noPar *= thisDict[(col, 'no')][index]
            except:
                continue
        if yesPar > noPar:
            if test.iloc[_]['class'] == 'yes':
                classMatch += 1
            else:
                classDismatch += 1
        else:
            if test.iloc[_]['class'] == 'no':
                classMatch += 1
            else:
                classDismatch += 1
    #print('classMatch:', classMatch)
    #print('classDismatch:', classDismatch)
    print('naiveBayes accuracy:', (classMatch / rows), '%')


#naiveBayes(test,train)