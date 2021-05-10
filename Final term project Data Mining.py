#!/usr/bin/env python
# coding: utf-8

# ### Final Term Project
# 
# ## Human Activity Recognition Using Smartphones
# 

# This project trying to build a model can predict the Activity of a user whether a person is (Laying, Standing , Sitting, Walking, Walking_upstairs, or Walking_downstairs) on the waist.. 
# 
# The information in this dataset is the measurements from wearable sensors ((accelerometer, gyroscope, magnetometer, and GPS)) of the smartphone. The experiments have been carried out with a group of 30 volunteers within an age bracket of 19-48 years.  
#  

# ##### The Plan is
# 
# 1. Read Dataset
# 
# 2. Dataset Cleaning
# 
# 3. Data Preprocessing
# 
# 4. Models
# 
#    a. Logistic Regression
#    
#    b. Support Vector Machine
#    
#    c. K-Nearest Neighbor
#    
#    d. Random Forest 
#    
# 5- Calculate accuracy.
# 

# In[63]:


# Load the libraries 
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import metrics
get_ipython().run_line_magic('matplotlib', 'inline')
import warnings
warnings.filterwarnings("ignore")


# In[64]:


#Load Dataset
train = pd.read_csv("data/train.csv")
test = pd.read_csv("data/test.csv")


# In[65]:


train.head()


# In[66]:


train['Activity'].value_counts()


# ###### Dataset Cleaning
# 1 Outliers   
# 2 Filling null values  
# 3 Check for data imbalance  

# In[67]:


#Handling outliers:
train.describe()
#There is no any possibility of having Outliers. All the values are squeezed between -1 to 1.


# In[68]:


#Checking for missing  NaN/null values 
print("Total Null values in Train: {}\n".format(train.isnull().values.sum()))
print("Total Null values in Test: {} \n".format(test.isnull().values.sum()))


# In[69]:


#Check for imbalanced dataset
plt.figure(figsize = (16,8))
plt.title("Subject with Each Activity")
sns.countplot(hue = 'Activity', x='subject',data = train);
plt.show()


# In[70]:


# There is no any huge amount of gap between them. 
plt.figure(figsize = (12,8))
sns.countplot(x = 'Activity', data = train);


#  Correcting feature names by remove ()

# In[71]:


# Removing ()
columns = train.columns
columns = columns.str.replace('[()]','')
columns = columns.str.replace('[-]','')
columns = columns.str.replace('[,]','')
train.columns = columns
test.columns = columns


# In[72]:


train.columns


# ### Data Preprocessing
# Splitting training and testing

# In[73]:


X_train = train.drop(["subject","Activity"], axis = 1)
y_train = train.Activity


X = test.drop(["subject","Activity"], axis = 1)
y = test.Activity

print('Training data size:', X.shape)
print('Test data size:', X.shape)


# In[74]:


model_score = pd.DataFrame(columns = ("Model","Score"))


# ### Models and Cross Validations
# Logistic Regression
# SVM
# Random Forest
# KNN

# In[75]:


# 1- Logistic regression model:
from sklearn.linear_model import LogisticRegression
from sklearn import model_selection
from sklearn.model_selection import train_test_split 
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
import matplotlib.cm as cm


# In[76]:


# K-FOLD= 10
kfold = model_selection.KFold(n_splits=10, random_state=42)
kfold
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=42)


# In[77]:


logmodel = LogisticRegression()
logmodel.fit(X_train, y_train)
predictions = logmodel.predict(X_test)
#probs_y=logmodel.predict_proba(X_test) 
accuracy_scores1 = accuracy_score(y_test, predictions)*100
print('Logistic Regression accuracy: {}%'.format(accuracy_scores1))


# In[78]:


from sklearn.metrics import confusion_matrix  
y_pred=logmodel.predict(X_test)
cm = confusion_matrix(y_test, y_pred)  
cm


# In[79]:


model_score = model_score.append(pd.DataFrame({'Model':["logistic regression"],'Score':[accuracy_scores1]}))


# The performance of each human daily activity was measured in terms of precision, recall and F-measure

# In[80]:


from sklearn.metrics import classification_report
print(classification_report(y_test, predictions))


# In[81]:


# 2- Support Vector Classifier
from sklearn.svm import SVC
clf2 = SVC().fit(X_train, y_train)
prediction = clf2.predict(X_test)
accuracy_scores2 = accuracy_score(y_test, prediction)*100
print('Support Vector Classifier accuracy: {}%'.format(accuracy_scores2))


# In[82]:


from sklearn.metrics import confusion_matrix  
y_pred=clf2.predict(X_test)
cm = confusion_matrix(y_test, y_pred)  
cm


# In[83]:


model_score = model_score.append(pd.DataFrame({'Model':["LinearSVM"],'Score':[accuracy_scores2]}))


# In[84]:


# 3- Random Forest model
from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier().fit(X_train, y_train)
prediction = clf.predict(X_test)
accuracy_scores3 = accuracy_score(y_test, prediction)*100
print('Random Forest Classifier accuracy: {}%'.format(accuracy_scores3))


# In[85]:


model_score = model_score.append(pd.DataFrame({'Model':["RandomForest"],'Score':[accuracy_scores3]}))


# In[86]:


# 4- K Nearest Neighbors
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier().fit(X_train, y_train)
prediction = knn.predict(X_test)
accuracy_scores4 = accuracy_score(y_test, prediction)*100
print('K Nearest Neighbors Classifier accuracy: {}%'.format(accuracy_scores4))


# In[87]:


from sklearn.metrics import confusion_matrix  
y_pred=knn.predict(X_test)
cm = confusion_matrix(y_test, y_pred)  
cm


# In[88]:


model_score = model_score.append(pd.DataFrame({'Model':["KNN"],'Score':[accuracy_scores4]}))


# Also measured based on the comparison of overall accuracy rate between different classifiers

# #### Compare accuray between models

# In[89]:


model_score.head()


# After applied different machine learning algorithms; found that Logistic Regression performed the best in classifying different activities.

# ![image.png](attachment:image.png)
