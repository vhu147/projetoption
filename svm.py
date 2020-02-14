import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
import seaborn as sns
import time
from sklearn import metrics

start_time = time.time()

df = pd.read_csv('data.csv')

df = df.replace([np.inf, -np.inf], np.nan).dropna()

features = df.drop(['Timestamp','Label'], axis = 1)
labels = pd.DataFrame(df['Label'])

feature_array = features.values
label_array = labels.values

# X_train,X_test,y_train,y_test = train_test_split(feature_array,label_array,test_size=0.20)
X_train, X_test, y_train, y_test = train_test_split(feature_array, label_array, test_size=0.3, random_state=1)

X_train = preprocessing.normalize(X_train)
X_test = preprocessing.normalize(X_test)

# print(df.info())

neighbours = np.arange(1,25)
train_accuracy =np.empty(len(neighbours))
test_accuracy = np.empty(len(neighbours))

svclassifier1 = SVC(kernel='linear')
svclassifier1.fit(X_train, y_train.ravel())
svclassifier2 = SVC(kernel='poly', degree=8)
svclassifier2.fit(X_train, y_train.ravel())
svclassifier3 = SVC(kernel='rbf')
svclassifier3.fit(X_train, y_train.ravel())
svclassifier4 = SVC(kernel='sigmoid')
svclassifier4.fit(X_train, y_train.ravel())

y_pred2 = svclassifier2.predict(X_test)
y_pred1 = svclassifier1.predict(X_test)
y_pred3 = svclassifier3.predict(X_test)
y_pred4 = svclassifier4.predict(X_test)

print("Accuracy of Linear Kernel : ",metrics.accuracy_score(y_test, y_pred1))
print("Accuracy of Polynomial Kernel : ",metrics.accuracy_score(y_test, y_pred2))
print("Accuracy of Gaussian Kernel : ",metrics.accuracy_score(y_test, y_pred3))
print("Accuracy of Sigmoid Kernel : ",metrics.accuracy_score(y_test, y_pred4))
print("--- %s seconds ---" % (time.time() - start_time))