import joblib
import numpy as np
from pyitlib import discrete_random_variable as drv
from numpy import log2 as log

from functions import Discretize

eps = np.finfo(float).eps

numOfBins = 3


# def ig(e_dataset, e_attr):
#    return (e_dataset - e_attr)


def find_entropy(df):
    """

    @param df: dataFrame obj
    @return: entropy value of df

    """
    Class = df.keys()[-1]  # To make the code generic, changing target variable class name
    entropy = 0
    values = df[Class].unique()
    for value in values:
        fraction = df[Class].value_counts()[value] / len(df[Class])
        entropy += -fraction * np.log2(fraction)
    return entropy


def find_entropy_attribute(df, attribute):
    """

    @param df: dataFrame obj
    @param attribute: string of specific attribute
    @return: entropy of attribute
    """
    Class = df.keys()[-1]  # To make the code generic, changing target variable class name
    target_variables = df[Class].unique()  # This gives all 'Yes' and 'No'
    variables = df[attribute].unique()  # This gives different features in that attribute
    entropy2 = 0
    for variable in variables:
        entropy = 0
        for target_variable in target_variables:
            num = len(df[attribute][df[attribute] == variable][df[Class] == target_variable])
            den = len(df[attribute][df[attribute] == variable])
            fraction = num / (den + eps)
            entropy += -fraction * log(fraction + eps)
            fraction2 = den / len(df)
            entropy2 += -fraction2 * entropy
    return abs(entropy2)


def find_winner(df):
    """

    @param df: dataFrame obj
    @return: the max value of my entropy verification

    """
    Entropy_att = []
    IG = []
    for key in df.keys()[:-1]:
        # Entropy_att.append(find_entropy_attribute(df,key))
        IG.append(find_entropy(df) - find_entropy_attribute(df, key))
    return df.keys()[:-1][np.argmax(IG)]


def bestIGattr(data, attributes, toSplit=False):

    """

    :param data:
    :param attributes:
    :param toSplit:
    :return: best choice by gain

    """
    classEntropy = drv.entropy(data['class']).item(0)
    attrsIG = {}
    for attr in attributes:
        attrsIG[attr] = find_entropy(data) - find_entropy_attribute(data, attr)
    maxGain = max(attrsIG.values())
    for attr in attrsIG:
        if attrsIG[attr] == maxGain:
            return attr


def Build_Dict(data):
    """

    :param data:
    :return: attributes tree
    """
    attributes = {}
    for i in data:
        attr = i.split()[1]
        x = i.split()[2]
        if i.split()[2] == 'NUMERIC':
            field = list(range(numOfBins))
        else:
            field = x.replace('{', '').replace('}', '').split(',')
        attributes[attr] = field
    return attributes


def buildTree(classDict, data, attributes, attrList, toSplit = False,numNodes = 100):
    """
    :param classDict:
    :param data:
    :param attributes:
    :param attrList:
    :param toSplit:
    :param numNodes:
    :return: the model,tree as a dict
    """
    if len(data['class'])<=numNodes and len(data['class'])>0:
        return data['class'].mode().iloc[0]
    else:
        if len(attrList) > 0:
            bestOp = bestIGattr(data, attrList, toSplit)
            classDict[bestOp] = {}
            for val in attributes[bestOp]:
                if len(data.loc[data[bestOp] == val]) > 0 and len(attrList) > 0:
                    newAttrsList = attrList.copy()
                    newAttrsList.remove(bestOp)
                    classDict[bestOp][val] = buildTree({}, data.loc[data[bestOp] == val], attributes, newAttrsList)
            return classDict
        else:
            return data['class'].mode().iloc[0]


def fun(tree, test):
    """
    tree -- decision tree dictionary
    test -- testing examples in form of pandas dataframe
    """
    res = []

    for _, e in test.iterrows():
        x = predict(tree, e)
        res.append(x)  # tree->dictionary , e -> subFrame
    return res  # array with expected class values


def predict(tree, subFrame):
    """
    tree -- decision tree dictionary
    subFrame -- a testing example in form of pandas series
    """
    c = tree
    while isinstance(c, dict):
        root = list(c.keys())[0]
        try:
            v = subFrame[root]
        except:
            for i in c[root]:
                print('r:', root)
                print('sb|:', subFrame[root])
                if subFrame[root] in i:
                    v = i
                    break
            print('i', i)
            v = i
        try:
            c = c[root][v]
        except:
            c = c[root][list(c[root].keys())[0]]
    return c


def result(arrayExpected, arrayTest):
    """
    test the model against the given test file
    :param arrayExpected:
    :param arrayTest:
    """
    match = 0;
    fail = 0;
    for _ in range(len(arrayExpected)):
        if arrayExpected[_] != None and arrayTest[_] != None:
            if arrayExpected[_] == arrayTest[_]:
                match += 1
            else:
                fail += 1
    # print('Matched values:', match)
    # print('NON-Matched:', fail)
    print('ID3 Accuracy:', (match / (match + fail)), '%')


def ID3_algorithm(train,test,structFile):
    """
    main program
    :param train:
    :param test:
    :param structFile:
    """

    train = Discretize(numOfBins, train, structFile)
    test = Discretize(numOfBins, test, structFile)
    attributes = Build_Dict(open(structFile))
    attrList = list(attributes.keys())
    attrList.remove('class')
    Decision_tree = buildTree({}, train, attributes, attrList)

    # save model to file
    filename = 'ID3_model.sav'
    joblib.dump(Decision_tree, filename)

    result(fun(Decision_tree, test), list(test['class']))
