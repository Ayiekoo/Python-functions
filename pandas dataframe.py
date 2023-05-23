#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[7]:


df = pd.read_csv('C:/Users/Ayieko/Desktop/python/student.csv')
print(df)


# In[24]:





# In[ ]:





# In[ ]:





# In[9]:


df.head()


# In[10]:


df.tail()


# In[11]:


df.head()


# In[12]:


df.tail(2)


# In[13]:


df.head(2)


# In[14]:


df.iloc[0]


# In[15]:


df.iloc[3]


# In[16]:


df.values


# In[17]:


df = pd.read_csv('C:/Users/Ayieko/Desktop/python/student.csv', chunksize=2)


# In[19]:


for chunk in df:
    print(chunk)


# In[23]:





# In[25]:





# In[ ]:




