#clear screen
from os import system
system("clear")

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

#train/test split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

#define classifier
model = GaussianNB()

#fit model
model.fit(X_train, Y_train)

#make predictions
Y_train_pred = model.predict(X_train)
Y_test_pred = model.predict(X_test)

#evaluation metrics for training dataset to evaluate the performance of the model
print("Evaluation Metrics for Training Dataset:")

#confusion matrix
conf_matrix_train = metrics.confusion_matrix(Y_train, Y_train_pred)
print(conf_matrix_train)

#expand items from confusion matrix
TN, FP, FN, TP = conf_matrix_train.ravel()

#manually calculate & print related metrics
print("Accuracy:", round((TN+TP)/(TN+FP+FN+TP),4))
print("Sensitivity:", round(TP/(TP+FN),4))
print("Specifity:", round(TN/(TN+FP),4))
print("F1 Score:", round(2*TP/(TP+FP)*TP/(TP+FN)/(TP/(TP+FP)+TP/(TP+FN)),4))

#verify manual calculations
print("Verified Accuracy:", round(metrics.accuracy_score(Y_train, Y_train_pred),4))
print("Verified Sensitivity:", round(metrics.recall_score(Y_train, Y_train_pred, pos_label='orange'),4)) #orange = class 1
print("Verified Specificity:",round(metrics.recall_score(Y_train, Y_train_pred, pos_label='grapefruit'),4)) #grapefruit = class 0
print("Verified F1 Score:", round(metrics.f1_score(Y_train, Y_train_pred, pos_label='orange'),4)) #orange = class 1

#convert test labels to binary numbers
lb = LabelBinarizer()
Y_train_binary = lb.fit_transform(Y_train)

#predict probabilities & calculate False & True Positive Rates
prob_train = model.predict_proba(X_train)
FPR_train, TPR_train, _ = metrics.roc_curve(Y_train_binary, prob_train[:, 1]) #orange = class 1
auc_train = round(metrics.auc(FPR_train, TPR_train),4)
print('Train AUC:', auc_train)

#manually calculate log loss
log_loss_train = 0.00
N = len(Y_train_binary)
for i in range(N):
    Y_i = Y_train_binary[i]
    P_i = prob_train[i, 1]
    log_loss_train -= (Y_i * np.log(P_i) + (1 - Y_i) * np.log(1 - P_i))

print("Log-Loss:", log_loss_train/N)

#verify the log-loss
print("Verified Log-Loss:", metrics.log_loss(Y_train_binary, prob_train))
print(" ")

#evaluation metrics for test dataset to evaluate the performance of the model
print("Evaluation Metrics for Test Dataset:")
#confusion matrix
conf_matrix_test = metrics.confusion_matrix(Y_test, Y_test_pred)
print(conf_matrix_test)

#expand items from confusion matrixs
TN, FP, FN, TP = conf_matrix_test.ravel()

#manually calculate & print related metrics
print("Accuracy:", round((TN+TP)/(TN+FP+FN+TP),4))
print("Sensitivity:", round(TP/(TP+FN),4))
print("Specifity:", round(TN/(TN+FP),4))
print("F1 Score:", round(2*TP/(TP+FP)*TP/(TP+FN)/(TP/(TP+FP)+TP/(TP+FN)),4))

#verify manual calculations
print("Verified Accuracy:", round(metrics.accuracy_score(Y_test, Y_test_pred),4))
print("Verified Sensitivity:", round(metrics.recall_score(Y_test, Y_test_pred, pos_label='orange'),4)) #orange = class 1
print("Verified Specificity:", round(metrics.recall_score(Y_test, Y_test_pred, pos_label='grapefruit'),4)) #grapefruit = class 0
print("Verified F1 Score:", round(metrics.f1_score(Y_test, Y_test_pred, pos_label='orange'),4)) #orange = class 1

#convert test labels to binary numbers
Y_test_binary = lb.fit_transform(Y_test)

#predict probabilities & calculate False & True Positive Rates
prob_test = model.predict_proba(X_test)
FPR_test, TPR_test, _ = metrics.roc_curve(Y_test_binary, prob_test[:, 1]) #orange = class 1
auc_test = round(metrics.auc(FPR_test, TPR_test),4)
print('auc_test:', auc_test)

#calculate log loss
log_loss_test = 0.00
N = len(Y_test_binary)
for i in range(N):
    Y_i = Y_test_binary[i]
    P_i = prob_test[i, 1]
    log_loss_test -= (Y_i * np.log(P_i) + (1 - Y_i) * np.log(1 - P_i))

print("Log Loss:", log_loss_test/N)

#verify the log-loss
print("Verified Log-Loss:", metrics.log_loss(Y_test_binary, prob_test))

#plot ROC for both train & test data
plt.figure(figsize=(8,7))
plt.plot(FPR_train, TPR_train, color='blue', lw=2, label=f'AUC_Train = {auc_train*100:.2f}%')
plt.plot(FPR_test, TPR_test, color='red', lw=2, label = f'AUC_Test = {auc_test*100:.2f}%')
plt.plot([0, 1], [0, 1], color='green', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiving Operating Characteristic Curve')
plt.legend(loc="lower right")
plt.show()