import pandas as pd



def calc(tp,tn,fp,fn):
    """

    :param tp: match_yes
    :param tn: match_no
    :param fp: fail_yes
    :param fn: fail_no
    :return:
    """
    accuracy = (tp+tn)/(tp+tn+fn+fp)
    recall = tp/(tp+fn)
    precision = tp/(tp+fp)
    fMeasure = 2*precision*recall/(precision+recall)
    print("Evaluation Indicators:")
    print("Accuracy: ",accuracy)
    print("Recall: ",recall)
    print("Precision: ",precision)
    print("fMeasure: ",fMeasure)
    print()

def fixStr(val,name):
    tmp = name.upper()+" = "+str(val)
    return tmp
def buildMatrix(tp,tn,fp,fn):

    data = {'Negative':[fixStr(tn,'tn'),fixStr(fn,'fn')],'Positive':[fixStr(fp,'fp'),fixStr(tp,'tp')]}
    matrix = pd.DataFrame(data)
    print("Confusion matrix:")
    print()
    print(matrix)
    print()

def Eval(tp,tn,fp,fn):
    calc(tp,tn,fp,fn)
    buildMatrix(tp,tn,fp,fn)

