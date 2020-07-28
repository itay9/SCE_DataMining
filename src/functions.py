import pandas as pd
from sklearn import preprocessing


def getColumnTitles(table):
    return list(table.columns)

def Discretize(num, table, structureTextFile):
    def numericCol(table, structureTextFile):
        structure = pd.read_csv(structureTextFile, sep=" ", names=['type', 'feature', 'data'])
        column = []
        headers = getColumnTitles(table)
        for i in range(structure.shape[0]):
            if 'NUMERIC' in structure.loc[i]['data']:
                column += [headers[i]]
        return column
    column = numericCol(table, structureTextFile)
    table = table.applymap(lambda s: s.lower() if type(s) == str else s)
    for col in column:
        table[col] = pd.qcut(table[col], num, labels=False, duplicates='drop')
    table = table.fillna(table.mode().iloc[0])
    # table.fillna(-1, inplace=True)
    # table.apply(lambda x: x.astype(str).str.lower())
    #table = table.applymap(lambda s: s.lower() if type(s) == str else s)
    return table


def p_xy(table,column_x,value_x,column_class,value_class):
    length=table[column_x].value_counts()[value_x]
    try:
        p=table.loc[table[column_x]==value_x][column_class].value_counts()[value_class]
    except:
        p=1
    p=round(p/length,3)
    return p

def valuesType(table,column):#return a list with the name of values in column
    columnValues=table[column].unique().tolist()
    if -1 in columnValues:
        columnValues.remove(-1)
    return columnValues

def pArrayByFeature(table,featureCol,classValue,classCol):#calculate p(x|y=value)
    array=[]
    #for i in range (len(valuesType(table,featureCol))):
    for i in valuesType(table,featureCol):
        array+=[p_xy(table,featureCol,i,classCol,classValue)]
    return array

def fit_transforms(table):
    le=preprocessing.LabelEncoder()
    columns = getColumnTitles(table)
    for col in columns:
        try:
            table[col] = le.fit_transform(table[col])
        except:
            continue
    return table
