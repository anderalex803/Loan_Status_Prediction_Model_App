#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.svm import SVC
import pickle


# In[8]:


import pandas as pd

data  = pd.read_csv("https://raw.githubusercontent.com/dphi-official/Datasets/master/Loan_Data/loan_train.csv" )
data


# In[13]:


data = data.drop(['Loan_ID', 'Unnamed: 0'], axis = 1)


# In[14]:


data['Gender'] = data['Gender'].fillna(data['Gender'].mode()[0])
data['Dependents'] = data['Dependents'].fillna(data['Dependents'].mode()[0])
data['Married'] = data['Married'].fillna(data['Married'].mode()[0])
data['Self_Employed'] = data['Self_Employed'].fillna(data['Self_Employed'].mode()[0])
data['Credit_History'] = data['Credit_History'].fillna(data['Credit_History'].mode()[0])
data['Loan_Amount_Term'] = data['Loan_Amount_Term'].fillna(data['Loan_Amount_Term'].mode()[0])
data['LoanAmount'] = data['LoanAmount'].fillna(data['LoanAmount'].mode()[0])


# In[15]:


from sklearn.preprocessing import LabelEncoder

lb_make = LabelEncoder()
data["Gender"] = lb_make.fit_transform(data["Gender"])
data["Married"] = lb_make.fit_transform(data["Married"])
data["Dependents"] = lb_make.fit_transform(data["Dependents"])
data["Education"] = lb_make.fit_transform(data["Education"])
data["Self_Employed"] = lb_make.fit_transform(data["Self_Employed"])
data["Property_Area"] = lb_make.fit_transform(data["Property_Area"])
data["Loan_Status"] = lb_make.fit_transform(data["Loan_Status"])


# In[16]:


X = data.drop(['Loan_Status'], axis = 1)
Y = data['Loan_Status']


# In[17]:


from sklearn.preprocessing import StandardScaler
scaled_features = StandardScaler().fit_transform(X)
scaled_features = pd.DataFrame(data=scaled_features)
scaled_features.columns= X.columns


# In[18]:


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0, stratify=Y)


# In[19]:


# import SMOTE 
from imblearn.over_sampling import SMOTE

sm = SMOTE(random_state = 25, sampling_strategy = 1.0)   # again we are eqalizing both the classes
# fit the sampling
X_train, Y_train = sm.fit_sample(X_train, Y_train)


# In[21]:


DT_model = DecisionTreeClassifier(criterion="entropy", max_depth = 250, min_samples_split=3 )


# In[29]:


LR_model = LogisticRegression(C=100)


# In[23]:


extra_clf = ExtraTreesClassifier(n_estimators=100, max_depth=50, min_samples_split=6)


# In[25]:


extra_clf.fit(X_train, Y_train)


# In[30]:


LR_model.fit(X_train,Y_train)


# In[31]:


DT_model.fit(X_train,Y_train)


# In[32]:


pickle.dump(extra_clf,open('Extra Tree Modle.pkl','wb'))
pickle.dump(LR_model,open('log_model.pkl','wb'))
pickle.dump(DT_model,open('DT_model.pkl','wb'))


# In[ ]:




