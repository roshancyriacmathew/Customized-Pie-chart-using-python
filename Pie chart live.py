#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.datasets import load_iris


# In[2]:


iris_data = load_iris()


# In[3]:


print(iris_data.DESCR)


# In[4]:


df = pd.DataFrame(iris_data.data)
df.columns = iris_data.feature_names
df['Species'] = iris_data.target
df.head()


# In[5]:


data = df['Species'].value_counts()
plt.pie(data)


# In[9]:


data = df['Species'].value_counts()
data.plot(kind='pie', autopct="%0.1f%%")


# In[27]:


fig = plt.figure(figsize=(10,10))
colors = ("cyan", "yellow","crimson")
wp = {'linewidth':2, 'edgecolor':"black"}
data = df['Species'].value_counts()
explode = (0.1,0.1,0.1)
textprops = {'fontstyle':'italic','fontweight':'heavy'}
classes = ['Iris-setosa','Iris-Versicolor','Iris-Virginica']
data.plot(kind='pie', autopct="%0.1f%%", labels = classes, startangle=45, colors = colors,
         explode = explode, shadow=True, wedgeprops = wp, textprops = textprops)
plt.legend(title="species")
plt.title("Different iris species", loc='center', color='black', fontsize='25', fontweight='bold')
plt.show()


# In[ ]:




