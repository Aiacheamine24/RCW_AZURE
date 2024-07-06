# External Imports
from sklearn.metrics import confusion_matrix

def custom_metrics(matrix): 
    TP, FP, FN, TN = matrix.ravel()
    accuracy = (TP + TN) / (TP + FP + FN + TN)
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    specificity = TN / (TN + FP)
    f1 = (2 * precision * recall) / (precision + recall)
    return accuracy, precision, recall, specificity, f1