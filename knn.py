import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import seaborn as sns
import time

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

for i,k in enumerate(neighbours):
    knn = KNeighborsClassifier(n_neighbors=k,algorithm="kd_tree",n_jobs=-1)

    knn.fit(X_train,y_train.ravel())

    train_accuracy[i] = knn.score(X_train, y_train.ravel())

    test_accuracy[i] = knn.score(X_test, y_test.ravel())

print("--- %s seconds ---" % (time.time() - start_time))

plt.title('k-NN Varying number of neighbors')
plt.plot(neighbours, test_accuracy, label='Testing Accuracy')
plt.plot(neighbours, train_accuracy, label='Training accuracy')
plt.legend()
plt.xlabel('Number of neighbors')
plt.ylabel('Accuracy')
plt.show()