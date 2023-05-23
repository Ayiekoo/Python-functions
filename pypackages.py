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


# In[31]:


import pandas as pd
import matplotlib.pyplot as plt


# In[33]:


DataFrame = pd.read_csv('C:/Users/Ayieko/Desktop/python/simplilearn/airports/airports.csv')
print(DataFrame)


# In[34]:


DataFrame.head()


# In[36]:


#### Lets create a multilevel index
DataFrame.set_index(['type', 'latitude_deg'])


# In[56]:


# Create scatter plot
plt.figure(figsize=(10, 6))
plt.scatter(DataFrame['longitude_deg'], DataFrame['latitude_deg'], c='blue', alpha=0.5)
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title('Geographic Locations of Airports')
plt.show()


# In[37]:


#### Lets create a multilevel index
DataFrame.set_index(['type', 'elevation_ft'])


# In[45]:


# Count airport types
count_by_type = DataFrame['type'].value_counts()

# Create bar chart
plt.figure(figsize=(10, 6))
count_by_type.plot(kind='bar')
plt.xlabel('Airport Type')
plt.ylabel('Count')
plt.title('Count of Airport Types')
plt.show()


# In[46]:


# Create histogram
plt.figure(figsize=(10, 6))
plt.hist(DataFrame['elevation_ft'].dropna(), bins=20)
plt.xlabel('Elevation (ft)')
plt.ylabel('Frequency')
plt.title('Distribution of Airport Elevations')
plt.show()


# In[49]:


# Create scatter plot
plt.figure(figsize=(10, 6))
plt.scatter(DataFrame['longitude_deg'], DataFrame['latitude_deg'], c='blue', alpha=0.5)
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title('Geographic Distribution of Airports')
plt.show()


# In[51]:


# Create box plot
plt.figure(figsize=(10, 6))
DataFrame.boxplot(column='elevation_ft', by='type')
plt.xlabel('Airport Type')
plt.ylabel('Elevation (ft)')
plt.title('Comparison of Elevations by Airport Type')
plt.suptitle('')
plt.show()


# In[54]:


import seaborn as sns


# In[55]:


# Pivot table to count airports by continent and scheduled service
count_by_continent_service = DataFrame.pivot_table(index='continent', columns='scheduled_service', aggfunc='size', fill_value=0)

# Create heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(count_by_continent_service, annot=True, fmt='d', cmap='YlGnBu')
plt.xlabel('Scheduled Service')
plt.ylabel('Continent')
plt.title('Count of Airports by Continent and Scheduled Service')
plt.show()


# In[58]:


### data visualization using seaborn

dat = pd.read_csv('C:/Users/Ayieko/Desktop/python/simplilearn/adaniports/ADANIPORTS.csv')
print(dat)


# In[59]:


dat.head()


# In[60]:


dat['H-L'] = dat.Low - dat.Low


# In[61]:


dat['100MA'] = dat['Close'].rolling(100).mean()


# In[62]:


## set the style to darkgrid
sns.set_style('darkgrid')


# In[86]:


####  plot the 3D graph
ax = plt.axes(projection='3d')
ax.scatter(dat.index, dat['H-L'], dat['100MA'])
ax.set_xlabel('Index')
ax.set_ylabel('H-L')
ax.set_zlabel('100MA')

plt.show()



# In[88]:


### OTHER METHODS OF 3D GRAPHS

import numpy as np


# In[89]:


z1 = np.linspace(0, 10, 100)
x1 = np.cos(2 + z1)
y1 = np.sin(2 + z1)


# In[90]:


sns.set_style('whitegrid')
ax = plt.axes(projection='3d')
ax.plot3D(x1,y1,z1)
plt.show()


# In[91]:


plt.plot(x1, y1)
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Plot of x1 and y1')
plt.show()


# In[71]:





# In[72]:


z1 = np.linspace(0, 10, 100)
x1 = np.cos(4 + z1)
y1 = np.sin(4 + z1)

plt.plot(x1, y1)
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Plot of x1 and y1')
plt.show()


# In[94]:


def return_z(x, y):
    z = x ** 2 + y ** 2
    return z


# In[95]:


sns.set_style('whitegrid')


# In[96]:


x1,y1 = np.linspace(-5,5,50), np.linspace(-5,5,50)


# In[ ]:


#### we shall revisit creating 3d plots using seaborn


# In[97]:


x1,y1 = np.meshgrid(x1,y1)
z1 = return_z(x1,y1)


# In[98]:


ax = plt.axes(projection='3d')
ax.plot_surface(x1,y1,z1)

ax = plt.axes(projection = '3d')
plt.show()


# In[85]:


import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

x1 = np.cos(2 + z1)
y1 = np.sin(2 + z1)
z1 = np.linspace(0, 10, 100)

ax = plt.axes(projection='3d')
ax.plot_surface(x1, y1, z1)

plt.show()


# In[ ]:




