#clear screen
from os import system
system("cls")

#import libraries
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn import metrics
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelBinarizer
import numpy as np

#load data
df = pd.read_csv('citrus.csv')

#define features & lables
X = df[['diameter', 'weight', 'red', 'green', 'blue']]
Y = df['name']

#convert labels to binary numbers
lb = LabelBinarizer()
Y = lb.fit_transform(Y)

#train/test split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

#define classifier
model = GaussianNB()

#fit model
model.fit(X_train, Y_train)

#make predictions
Y_train_pred = model.predict(X_train)
Y_test_pred = model.predict(X_test)

#define function for evaluation metrics
def metric(dataset, X_true, Y_true, Y_pred):

    #verify manual calculations
    print("Accuracy:", round(metrics.accuracy_score(Y_true, Y_pred), 4))
    print("Sensitivity:", round(metrics.recall_score(Y_true, Y_pred, pos_label=1), 4)) #orange = class 1
    print("Specificity:",round(metrics.recall_score(Y_true, Y_pred, pos_label=0), 4)) #grapefruit = class 0
    print("F-1 Score:", round(metrics.f1_score(Y_true, Y_pred, pos_label=1), 4)) #orange = class 1

    #predict probabilities & calculate true & false positive rates
    prob = model.predict_proba(X_true)
    FPR, TPR, _ = metrics.roc_curve(Y_true, prob[:, 1]) #orange = class 1
    auc = round(metrics.auc(FPR, TPR),4)
    print('Area Under the Curve:', auc)

    #log-loss
    print("Log-Loss:", metrics.log_loss(Y_true, prob))
    print(" ")

    #plot ROC
    plt.plot(FPR, TPR, lw=2, label=f'AUC {dataset}= {auc * 100:.2f}%')

#print evaluation metrics for training dataset
metric('Train', X_train, Y_train, Y_train_pred)

#print evaluation metrics for test dataset
metric('Test', X_test, Y_test, Y_test_pred)

#plot ROC for both training & test data
plt.plot([0, 1], [0, 1], color='green', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves for Both Training & Test Datasets')
plt.legend(loc="lower right")
plt.show()

#additional manual calculations & verification just for fun :)
def metric_m(X_true, Y_true, Y_pred):

    #confusion matrix
    conf_matrix = metrics.confusion_matrix(Y_true, Y_pred)

    #expand confusion matrix
    TN, FP, FN, TP = conf_matrix.ravel()

    #manually calculate & verify metrics
    print("Verified Accuracy:", round((TN+TP)/(TN+FP+FN+TP),4))
    print("Verified Sensitivity:", round(TP/(TP+FN),4))
    print("Verified Specifity:", round(TN/(TN+FP),4))
    print("Verified F-1 Score:", round(2*TP/(TP+FP)*TP/(TP+FN)/(TP/(TP+FP)+TP/(TP+FN)),4))

    #log loss
    prob = model.predict_proba(X_true)
    log_loss = 0.00
    N = len(Y_true)
    for i in range(N):
        Y_i = Y_true[i]
        P_i = prob[i, 1]
        log_loss -= (Y_i * np.log(P_i) + (1 - Y_i) * np.log(1 - P_i))

    print("Verified Log-Loss:", log_loss/N)
    print(" ")

#print metrics for training dataset
metric_m(X_train, Y_train, Y_train_pred)

#print metrics for test dataset
metric_m(X_test, Y_test, Y_test_pred)
