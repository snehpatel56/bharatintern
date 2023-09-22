#!/usr/bin/env python
# coding: utf-8

# # Task2 Titanic Classification

# The Titanic Classification project involves predicting whether passengers on the RMS Titanic survived or not, based on factors like age, gender, and ticket class. It's a classic binary classification task in data science, showcasing how machine learning can be applied to historical data to understand and predict outcomes.

# Importing the libraries

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# Data Collection & Processing

# In[2]:


# load the data from csv file to Pandas DataFrame
titanic_data = pd.read_csv(r'D:\Datasets\train.csv')


# In[4]:


# printing the first 5 rows of the dataframe
titanic_data.head()


# In[5]:


# number of rows and Columns
titanic_data.shape


# In[6]:


# getting some informations about the data
titanic_data.info()


# In[7]:


# check the number of missing values in each column
titanic_data.isnull().sum()


# Handling the Missing values

# In[8]:


# drop the "Cabin" column from the dataframe
titanic_data = titanic_data.drop(columns='Cabin', axis=1)
# replacing the missing values in "Age" column with mean value
titanic_data['Age'].fillna(titanic_data['Age'].mean(), inplace=True)


# In[10]:


# finding the mode value of "Embarked" column
print(titanic_data['Embarked'].mode())

# replacing the missing values in "Embarked" column with mode value
titanic_data['Embarked'].fillna(titanic_data['Embarked'].mode()[0], inplace=True)


# In[11]:


# check the number of missing values in each column
titanic_data.isnull().sum()


# In[12]:


# getting some statistical measures about the data
titanic_data.describe()


# In[13]:


# finding the number of people survived and not survived
titanic_data['Survived'].value_counts()


# # Data Visualization

# In[14]:


sns.set()
# making a count plot for "Survived" column
sns.countplot('Survived', data=titanic_data)


# In[15]:


titanic_data['Sex'].value_counts()


# In[16]:


# making a count plot for "Sex" column
sns.countplot('Sex', data=titanic_data)


# In[17]:


# making a count plot for "Pclass" column
sns.countplot('Pclass', data=titanic_data)


# In[18]:


sns.countplot('Pclass', hue='Survived', data=titanic_data)


# In[32]:


# Checking final features and attributes and making heatmap.
print("Final Features considered for Model Fitting: \n",titanic_data.columns )
d_train = titanic_data.copy()
print(d_train.head(3))

print("Correlation matrix: \n")
print(titanic_data.corr())

fig = plt.figure(figsize=(20,8))
print("\nHeat Map: \n")
sns.heatmap(titanic_data.corr(),annot=True)
print(plt.show())


# Encoding the Categorical Columns

# In[19]:


titanic_data['Sex'].value_counts()


# In[20]:


titanic_data['Embarked'].value_counts()


# In[21]:


# converting categorical Columns

titanic_data.replace({'Sex':{'male':0,'female':1}, 'Embarked':{'S':0,'C':1,'Q':2}}, inplace=True)


# Separating features & Target

# In[23]:


X = titanic_data.drop(columns = ['PassengerId','Name','Ticket','Survived'],axis=1)
Y = titanic_data['Survived']
print(X)
print(X)


# Splitting the data into training data & Test data

# In[24]:


X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.2, random_state=2)


# In[25]:


print(X.shape, X_train.shape, X_test.shape)


# Training model

# In[39]:


model = LogisticRegression()
# training the Logistic Regression model with training data
model.fit(X_train, Y_train)
# accuracy on training data
X_train_prediction = model.predict(X_train)
print(X_train_prediction)
training_data_accuracy = accuracy_score(Y_train, X_train_prediction)
print('Accuracy score of training data : ', training_data_accuracy)


# In[42]:


# Splitting the data in training data and testing data.
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

model = LogisticRegression(penalty='l2',solver='liblinear')
model.fit(X_train,Y_train)

# Predicting the model and printing the accuracy of the model.
Y_pred = model.predict(X_test)
print(Y_pred)

print("Accuracy of Logistic Model: ",sum([Y_pred==Y_test][0].values)/len(Y_test))

