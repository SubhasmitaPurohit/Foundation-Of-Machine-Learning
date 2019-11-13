#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sklearn


# In[3]:


get_ipython().run_line_magic('pwd', '')


# In[5]:


get_ipython().run_line_magic('cd', 'F:\\BA\\Term 5\\ML')


# In[8]:


df=pd.read_csv("page_block.csv")


# In[9]:


df


# In[10]:


#To check the features
df.columns


# ## DATA SET INFORMATION
# The 5473 examples comes from 54 distinct documents. Each observation concerns one block
# ##### height: Height of the block.
# ##### lenght: Length of the block.
# ##### area:  Area of the block (height * lenght);
# ##### eccen: Eccentricity of the block (lenght / height);
# ##### p_black: Percentage of black pixels within the block (blackpix / area);
# ##### p_and: Percentage of black pixels after the application of the Run Length Smoothing Algorithm (RLSA) (blackand / area);
# ##### mean_tr: continuous. | Mean number of white-black transitions (blackpix / wb_trans);
# ##### blackpix:  Total number of black pixels in the original bitmap of the block.
# ##### blackand:  Total number of black pixels in the bitmap of the block after the RLSA.
# ##### wb_trans:  Number of white-black transitions in the original bitmap of the block.

The problem consists in classifying all the blocks of the page layout of a document that has been detected by a segmentation process.
This is an essential step in document analysis in order to separate text from graphic areas.
Indeed, the five classes are: text (1), horizontal line (2), picture (3), vertical line (4) and graphic (5)

#  Perform initial EDA

# In[1]:


#Check the size
df.shape


# In[12]:


#indetail of the features null values and type of the data
df.info()


# In[14]:


#Variance of numerical column
df.var()


# In[15]:


def normalize(df):
    result = df.copy()
    for feature_name in df.columns:
        max_value = df[feature_name].max()
        min_value = df[feature_name].min()
        result[feature_name] = (df[feature_name] - min_value) / (max_value - min_value)
    return result


# In[16]:


normalize(df)


# In[17]:


df['class'].value_counts()


# # Perform classification

# In[20]:


x = df.iloc[:,[0,1,2,3,4,5,6,7,8,9]].values
y = df.iloc[:,10].values


# In[21]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=0)


# In[22]:


from sklearn.preprocessing import StandardScaler
standard_scaler_x= StandardScaler()
x_train=sc_x.fit_transform(x_train)
x_test=sc_x.transform(x_test)


# In[24]:


y_pred=classifier.predict(x_test)


# In[25]:


y_pred


# In[37]:


from sklearn.linear_model import LogisticRegression 
from sklearn.metrics import accuracy_score 
model = LogisticRegression()

model.fit(x, y)


# In[47]:


predicted_classes=model.predict(x_test)


# In[49]:


accuracy = accuracy_score(y_test,predicted_classes)
parameters = model.coef_
print(accuracy)


# After performimng logistic regression we got accuracy as 68%. To improve the accuracy of the algorithm again we have to do neural network.

# In[52]:


from sklearn.metrics import confusion_matrix


# In[54]:


confusionmatrix=confusion_matrix(y_test,y_pred)
confusionmatrix


# In[55]:


from sklearn.neural_network import MLPClassifier


# In[57]:


neural_network_model = MLPClassifier(hidden_layer_sizes=(40,60))


# In[58]:


neural_network_model.fit(x_train,y_train)


# In[59]:


neural_network_predicted_class=neural_network_model.predict(x_test)


# In[60]:


accuracy_score(y_test,neural_network_predicted_class)


# After perfomimg neural network the accuracy of the algorithm increased to 96%

# In[61]:


from sklearn.metrics import classification_report


# In[62]:


print(classification_report(y_test,neural_network_npredicted_class))


# In[64]:


neural_network_confusionmatrix=confusion_matrix(y_test, nnpredicted_class)
neural_network_confusionmatrix


# In[ ]:




