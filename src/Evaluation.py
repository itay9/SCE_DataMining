import pandas as pd


def Eval(tp, tn, fp, fn):
    def fixStr(val, name):
        tmp = name.upper() + " = " + str(val)
        return tmp

    accuracy = (tp + tn) / (tp + tn + fn + fp)
    recall = tp / (tp + fn)
    precision = tp / (tp + fp)
    fMeasure = 2 * precision * recall / (precision + recall)
    print("Evaluation Indicators:")
    print("Accuracy: ", accuracy)
    print("Recall: ", recall)
    print("Precision: ", precision)
    print("fMeasure: ", fMeasure)
    print()
    data = {'Negative': [fixStr(tn, 'tn'), fixStr(fn, 'fn')], 'Positive': [fixStr(fp, 'fp'), fixStr(tp, 'tp')]}
    matrix = pd.DataFrame(data)
    print("Confusion matrix:")
    print()
    print(matrix)
    print()
