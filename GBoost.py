#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Import Environments:

import pandas as pd
import numpy as np


# In[2]:


#Load Data:

train = pd.read_excel (r'C:\Users\nvhuy\Desktop\test_data.xlsx')
test  = pd.read_excel (r'C:\Users\nvhuy\Desktop\test_data.xlsx')


# In[3]:


train.info(), test.info()


# In[4]:


train.isnull().sum()


# In[5]:


train.set_index("CRS_DEP_TIME", inplace=True)
test.set_index("CRS_DEP_TIME", inplace=True)


# In[6]:


train_y = train["ARR_DEL15"]


# In[7]:


#Delete the column ARR_DEL15 from train set:

train.drop(labels="ARR_DEL15", axis=1, inplace=True)


# In[8]:


train.shape, test.shape


# In[9]:


train_test = train.append(test)


# In[10]:


#Delete columns that are not used as features for training and prediction:

columns_to_drop = ["ORIGIN", "DEST"]
train_test.drop(labels=columns_to_drop, axis=1, inplace=True)


# In[11]:


#Replace NaN with zeroes:

train_test.fillna(value=0, inplace=True)


# In[12]:


train_test.isnull().values.any()


# In[13]:


#Generate feature sets (x):

train_x = train_test.values[0:4997]
test_x = train_test.values[4997:]


# In[14]:


train_x.shape, test_x.shape


# In[15]:


#Transform the data:

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
train_x_scale = scaler.fit_transform(train_x)
test_x_scale = scaler.transform(test_x)


# In[16]:


#Split training feature and target sets into training and validation subsets:
from sklearn.model_selection import train_test_split

train_x_sub, validation_x_sub, train_y_sub, validation_y_sub = train_test_split(train_x_scale, train_y, random_state=0)


# In[17]:


#Train with Gradient Boosting Algorithm:

from sklearn.ensemble import GradientBoostingClassifier

learning_rates = [0.05, 0.1, 0.25, 0.5, 0.75, 1]
for learning_rate in learning_rates:
    gboost = GradientBoostingClassifier(n_estimators=20, learning_rate = learning_rate, max_features=2, max_depth = 2, random_state = 0)
    gboost.fit(train_x_sub, train_y_sub)
    print("Learning rate: ", learning_rate)
    print("Accuracy score (training): {0:.3f}".format(gboost.score(train_x_sub, train_y_sub)))
    print("Accuracy score (validation): {0:.3f}".format(gboost.score(validation_x_sub, validation_y_sub)))
    print()


# In[27]:


#Output confusion matrix and classification report of Gradient Boosting algorithm on validation set:

from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc

gboost = GradientBoostingClassifier(n_estimators=20, learning_rate = 0.5, max_features=2, max_depth = 2, random_state = 0)
gboost.fit(train_x_sub, train_y_sub)
predictions = gboost.predict(validation_x_sub)

print("Confusion Matrix:")
print(confusion_matrix(validation_y_sub, predictions))
print()
print("Classification Report:")
print(classification_report(validation_y_sub, predictions))


# In[28]:


import itertools
import matplotlib.pyplot as plt


# In[29]:


def plot_confusion_matrix(cm,
                          normalize = False,
                          title = 'Confusion Matrix',
                          cmap = plt.cm.Blues):
    
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    plt.ylabel('ACTUAL')
    plt.xlabel('PREDICTED')
    plt.tight_layout()
    plt.grid('off')
    x_labels = ['on-time','','delay']
    y_labels = ['','on-time','','','','delay']
    ax = fig.add_subplot(111)
    ax.set_xticklabels([' '] + x_labels)
    ax.set_yticklabels([' '] + y_labels)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    

# Compute confusion matrix
cnf_matrix = confusion_matrix(validation_y_sub, predictions)
np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
plt.figure()
fig = plt.figure()
plot_confusion_matrix(cnf_matrix, title='Confusion matrix, without normalization')



# Plot normalized confusion matrix
plt.figure()
fig = plt.figure()
plot_confusion_matrix(cnf_matrix, normalize=True,
                      title='Normalized confusion matrix')


plt.show()


# In[25]:


# ROC curve and Area-Under-Curve (AUC)

y_scores_gboost = gboost.decision_function(validation_x_sub)
fpr_gboost, tpr_gboost, _ = roc_curve(validation_y_sub, y_scores_gboost)
roc_auc_gboost = auc(fpr_gboost, tpr_gboost)

print("Area under ROC curve = {:0.2f}".format(roc_auc_gboost))


# In[ ]:




