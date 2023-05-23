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


# In[15]:


DataFrame.plot(legend=True) ### gives a line graph by default


# In[14]:


DataFrame['mark'].plot(legend=True)


# In[16]:


## multiple graphs
DataFrame.plot(x='class', y='mark')


# In[19]:


## multiple graphs
DataFrame.plot(x='gender', y='mark')


# In[22]:


DataFrame['mark'].plot(kind='box', title='student scores')


# In[23]:


DataFrame['mark'].plot(kind='hist', title='student scores')


# In[ ]:




