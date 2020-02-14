import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
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

clf = DecisionTreeClassifier(criterion="entropy", splitter = "random", max_depth = 4)
clf = clf.fit(X_train,y_train)
y_pred = clf.predict(X_test)

print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
print("--- %s seconds ---" % (time.time() - start_time))