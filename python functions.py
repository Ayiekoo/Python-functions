#!/usr/bin/env python
# coding: utf-8

# In[3]:


def area():
    area_rect = 10 * 5
    print('Area of rectangle = ', area_rect)
    return
area()  ####### calls the function


# In[5]:


#### functions parameters and arguments

# create a function area() and pass three parameters shape, length and breath
def area(shape, length, breath):
    area = length * breath
    print(shape, 'area = ', area)
    return

# calling syntax
area('Rectangle', 10, 5) ######### calling functions with arguments
area('Square', 20, 20)   ######## calling the functions arguments


# In[7]:


########## python *args and *kwargs

def total_sales(*args):
    total = 0
    for sale in args:
        total = total + sale
    return total
print('Total sales:', total_sales(4000, 10000, 5000, 15000, 2000))


# In[9]:


###### **kwargs:

def total_sales(**kwargs):
    items = ""
    total = 0
    
    # Display items sold (keys)
    for item in kwargs.keys():
        items = item + item
    print('Items Sold:', items)
    
    
    # Display total sales
    for sale in kwargs.values():
        total = total + sale
    return total
print('Total sales:', total_sales(Tv_set = 4000, Dining_table = 10000, Fridge = 5000, Carpet = 15000))


# In[ ]:




