#!/usr/bin/env python
# coding: utf-8

# In[1]:


"""
The Boolean data type is trueth value, either True or False

The Boolean operators oderdered by priority

not x   >>>> "if x is Flase, then x, else y"
x and y  >>>> "if x is False, then x, else y"
x or y   >>>> "if x is False, then y, else x"

These comparison operators evaluate to True:

1 < 2 and 0 <= 1 and 3 > 2 and 2 >= 2 and 1 == 1 and 1 !=0 
"""

1 < 2 and 0 <= 1 and 3 > 2 and 2 >= 2 and 1 == 1 and 1 !=0 


# In[5]:


### Boolean operations
x, y = True, False
print(x and not y)
print(not x and y or x)


# In[7]:


### if condition evaluates to False

if None or 0 or 0.0 or '' or [] or {} or set():
    # None, 0, 0.0, empty strings, or empty
    # container types are evaluate to Flase
    print("Dead code")


# In[8]:


if None or 0 or 0.0 or '' or [] or {} or set():
    print("Dead code")


# In[ ]:




