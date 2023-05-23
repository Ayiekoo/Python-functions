#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[8]:


DataFrame = pd.read_csv('C:/Users/Ayieko/Desktop/python/simplilearn/student.csv')
print(DataFrame)


# In[9]:


#### Lets create a multilevel index
DataFrame.set_index(['class', 'mark'])


# In[10]:


DataFrame.index


# In[11]:





# In[ ]:




