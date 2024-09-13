#CMSC436 - Project 1

import os

def accuracy(TP, TN, FP, FN):
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    return accuracy

def error(TP, TN, FP, FN):
    error = (FP + FN) / (TP + TN + FP + FN)
    return error

def true_positive_rate(TP, FN):
    tpr = TP / (TP + FN)
    return tpr

def true_negative_rate(TN, FP):
    tnr = TN / (TN + FP)
    return tnr

def false_positive_rate(TN, FP):
    fpr = FP / (FP + TN)
    return fpr

def false_negative_rate(TP, FN):
    fnr = FN / (TP + FN)
    return fnr

def main():
    # Read and convert input to integers
    TP = int(input("Enter True Positive: "))
    TN = int(input("Enter True Negative: "))
    FP = int(input("Enter False Positive: "))
    FN = int(input("Enter False Negative: "))

    # Calculate metrics
    acc = accuracy(TP, TN, FP, FN)
    err = error(TP, TN, FP, FN)
    tpr = true_positive_rate(TP, FN)
    tnr = true_negative_rate(TN, FP)
    fpr = false_positive_rate(TN, FP)
    fnr = false_negative_rate(TP, FN)

    # Print results
    print("Accuracy: {:.2f}".format(acc))
    print("Error: {:.2f}".format(err))
    print("True Positive Rate: {:.2f}".format(tpr))
    print("True Negative Rate: {:.2f}".format(tnr))
    print("False Positive Rate: {:.2f}".format(fpr))
    print("False Negative Rate: {:.2f}".format(fnr))

if __name__ == "__main__":
    main()
