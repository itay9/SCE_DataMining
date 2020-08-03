import joblib

from functions import getColumnTitles,  valuesType, pArrayByFeature

def allArraysOfFetures(table, classCol):
    """

    :param table:
    :param classCol:
    :return: dict, keys is tuple(feature,'yes'/'no'),values is list of probabilities of the values of the key
    """
    thisDict = {}
    for i in getColumnTitles(table):
        if i not in classCol:
            for j in valuesType(table, classCol):
                thisDict[i, j] = pArrayByFeature(table, i, j, classCol)
                # print(pArrayByFeature(train,i,j,classCol))
    return thisDict

def naiveBayes(test, train, structFile):
    """
    print the accuracy of the model by test file
    :param test:
    :param train:
    :param structure:
    """

    thisDict=allArraysOfFetures(train, 'class')
    rows = test.shape[0]
    classMatch = 0
    classDismatch = 0
    # save model to file
    filename = 'naiveBayes_model.sav'
    joblib.dump(thisDict, filename)

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
    print('naiveBayes accuracy:', (classMatch / rows), '%')


