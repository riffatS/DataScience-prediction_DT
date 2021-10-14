#!/usr/bin/env python
# coding: utf-8

# In[12]:


import pandas as pd
import numpy as np  
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split 
from sklearn.tree import DecisionTreeClassifier
# from sklearn.externals.six import StringIO  
from IPython.display import Image  
from sklearn.tree import export_graphviz
import pydotplus
import six
import sys
sys.modules['sklearn.externals.six'] = six
from sklearn import tree
from io import StringIO


# In[2]:



s_data = pd.read_csv('Iris.csv')
print("Data imported successfully")
s_data=s_data.dropna()
s_data.head()

# s_data.shape


# In[3]:


x_data = s_data[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm','PetalWidthCm']]
y_data =s_data['Species']
print(x_data.columns)
# print(y_data)


# In[4]:


dtree=DecisionTreeClassifier()
dtree.fit(x_data,y_data)


# In[6]:


dot_data = StringIO()
export_graphviz(dtree, out_file=dot_data, feature_names=x_data.columns, filled=True, rounded=True,special_characters=True)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
graph

# Image(graph.create_png())


# In[13]:


text_representation = tree.export_text(dtree)
print(text_representation)

