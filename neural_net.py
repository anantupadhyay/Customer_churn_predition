'''
		THIS FILE IMPLEMENTS THE MLPC CLASSIFIER AND STORES THE CLASSIFIER TO BE USED FOR CREATION OF APP
'''
import numpy as np
import pickle
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import MLPClassifier

data = pd.read_csv('Churn_Modelling.csv')
data['Gender'] = data['Gender'].replace(['Male', 'Female'], [1,0])

X = data.iloc[:, 3:13].values
y = data.iloc[:, 13].values
labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])
labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])
onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()
X = X[:, 1:]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 13)

print X_train.shape
print X_test.shape
print X_train

sc = MinMaxScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

mlpc = MLPClassifier(solver = 'adam', alpha = 1e-5, hidden_layer_sizes = (100,))
mlpc.fit(X_train, y_train)
y_mlpc = mlpc.predict(X_test)

#Dumping the classifier for creating the app

with open('clf.pkl', 'wb') as fid:
    pickle.dump(mlpc, fid, 2) 

with open('sc.pkl', 'wb') as fid2:
	pickle.dump(sc, fid2, 2)