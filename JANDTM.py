import sys 
stdoutOrigin=sys.stdout 
sys.stdout = open("log.txt", "w")
import pandas as pd
import numpy as np  
import matplotlib.pyplot as plt  
dataset = pd.read_csv(("~/ml-model/JanData.csv"))
dataset.fillna(value=0, inplace=True)
print("Head:")
print(dataset.head())
print("Shape:")
print(dataset.shape)
X = dataset.drop('ARR_DELAY', axis=1)  
y = dataset['ARR_DELAY'] 
from sklearn.model_selection import train_test_split  
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20) 
from sklearn.tree import DecisionTreeClassifier  
classifier = DecisionTreeClassifier()  
classifier.fit(X_train, y_train) 
y_pred = classifier.predict(X_test) 
df=pd.DataFrame({'Actual':y_test, 'Predicted':y_pred})  
df 
print ("Predictions:")
print(df)
from sklearn import metrics  
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))  
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

sys.stdout.close()
sys.stdout=stdoutOrigin
