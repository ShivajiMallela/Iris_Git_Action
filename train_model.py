import itertools
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
from sklearn.datasets import load_iris

dataset = load_iris()
X = pd.DataFrame(dataset.data, columns=dataset.feature_names)
y = pd.Series(dataset.target)

X.columns = [col.replace(' (cm)', '') for col in X.columns]
X.columns = [col.replace(" ", "_") for col in X.columns]
X['sepal_length_width_ratio'] = X['sepal_length'] / X['sepal_width']
X['petal_length_width_ratio'] = X['petal_length'] / X['petal_width']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

## Logistic Regression

log_reg = LogisticRegression(C=0.0001, max_iter=100, multi_class='multinomial', solver='lbfgs')
log_reg.fit(X_train, y_train)
y_pred_lr = log_reg.predict(X_test)
cm_lr = confusion_matrix(y_test, y_pred_lr)
f1_score_lr = f1_score(y_test, y_pred_lr, average='micro')
prec_lr = precision_score(y_test, y_pred_lr, average='micro')
recall_lr = recall_score(y_test, y_pred_lr, average='micro')

train_accuracy_lr = log_reg.score(X_train, y_train)*100
test_accuracy_lr = log_reg.score(X_test, y_test)*100

## Random Forest Classifier
rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)
rf_clf.fit(X_train, y_train)
y_pred_rf = rf_clf.predict(X_test)
cm_rf = confusion_matrix(y_test, y_pred_rf)
f1_score_rf = f1_score(y_test, y_pred_rf, average='micro')
prec_rf = precision_score(y_test, y_pred_rf, average='micro')
recall_rf = recall_score(y_test, y_pred_rf, average='micro')

train_accuracy_rf = rf_clf.score(X_train, y_train)*100
test_accuracy_rf = rf_clf.score(X_test, y_test)*100

def plot_confusion_matrix(cm, target_names, title='Confusion Matrix', normalize=True):
    
    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    cmap = plt.cm.Blues
    plt.figure(figsize=(12, 8))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:.04f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig('confusion_matrix.png')
    plt.show()

plot_confusion_matrix(cm_lr, target_names=dataset.target_names, title='Logistic Regression Confusion Matrix')

with open('report.txt', 'w') as f:
    f.write("logistic regression train accuracy: {:.2f}%\n".format(train_accuracy_lr))
    f.write("logistic regression test accuracy: {:.2f}%\n".format(test_accuracy_lr))
    f.write("F1 Score: {:.4f}\n".format(f1_score_lr))
    f.write("Precision: {:.4f}\n".format(prec_lr))
    f.write("Recall: {:.4f}\n".format(recall_lr))   

    f.write("\n\nRandom Forest Classifier train accuracy: {:.2f}%\n".format(train_accuracy_rf))
    f.write("Random Forest Classifier test accuracy: {:.2f}%\n".format(test_accuracy_rf))
    f.write("F1 Score: {:.4f}\n".format(f1_score_rf))
    f.write("Precision: {:.4f}\n".format(prec_rf))
    f.write("Recall: {:.4f}\n".format(recall_rf))

