#!/usr/bin/env python
# coding: utf-8

# In[1]:


### return

"""
terminates execution of the function and passess the flow
of execution to the caller.
an optional value after the return keyword specifies the function result
"""
def incrementor(x):
    return x + 1
incrementor(4)


# In[2]:


"""
cleaning numerical columns in a dataset
"""

import pandas as pd
import numpy as np


# In[4]:


# create a simple dataframe
data = {'age': ['25', '31', '45', '?', '54', '38', '?', '29', '35', '?']}
df = pd.DataFrame(data)
print(df)


# In[6]:


df = clean_numerical_data(df, 'age')
print(df)


# In[7]:


def clean_numerical_data(df, column):
    df[column] = df[column].replace('?', np.nan)
    df[column] = pd.to_numeric(df[column])
    return df


# In[8]:


#### feature engineering
def create_binary_feature(df, column, threshold):
    new_feature = df[column].apply(lambda x: 1 if x > threshold else 0)
    return new_feature


# In[10]:


### application of the return function in feature engineering
# create a simple dataframe
data = {'Name': ['John', 'Sally', 'Mike', 'Jessica', 'Mohammed', 'Sophie', 'Louis', 'Chen'],
       'Salary': [5000, 7000, 45000, 80000, 35000, 620000, 750000, 30000]}
df = pd.DataFrame(data)
print(df)


# In[15]:


"""
 let's say we want to add a binary feature that indicates whether each person's salary is above $60,000. 
 We can use the create_binary_feature function we defined earlier:
"""

df['High_salary'] = create_binary_feature(df, 'Salary', 60000)
print(df)


# In[ ]:


#### In the new 'high_salary' column, a 1 indicates a salary above $60,000, and a 0 indicates a salary at or below $60,000.


# In[21]:


### let's do another example
# create a sample dataframe

data = {'Name': ['Austin', 'Bolly', 'Alex', 'Halima', 'Pili', 'Maua', 'Aboubark', 'Samson'],
        'Percentage_Scores': [77, 89, 90, 56, 71, 61, 66, 79]}
df = pd.DataFrame(df)
print(Final)


# In[22]:


data = {'Name': ['Austin', 'Bolly', 'Alex', 'Halima', 'Pili', 'Maua', 'Aboubark', 'Samson'],
        'Percentage_Scores': [77, 89, 90, 56, 71, 61, 66, 79]}
df = pd.DataFrame(df)
print(Final)


# In[23]:


df['Pass'] = create_binary_feature(df, 'Percentage_Scores', 75)
print(df)


# In[27]:


### using the return feature in model evaluation

### import the necessary library
from sklearn.metrics import accuracy_score, f1_score

def evaluate_model(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='weighted')
    return accuracy, f1

"""
calculating the accuracy and F1 score of a model's predictions and return these values.
"""


# In[28]:


# these are the true values
y_true = [0, 1, 0, 1, 0, 0, 1, 1, 0, 1, 1, 0, 1, 0]

# and these are the predicted values from our model
y_pred = [0, 1, 0, 1, 0, 1, 1, 0, 1, 1, 1, 0, 1, 0]


# In[29]:


accuracy, f1 = evaluate_model(y_true, y_pred)
print("Accuracy:", accuracy)
print("F1 Score:", f1)


# In[ ]:


"""
Accuracy and F1 Score are two metrics used to evaluate the performance of classification models.

    Accuracy is simply the proportion of predictions that the model got right. 
    In this case, your model's accuracy is approximately 0.786, or 78.6%. 
    This means that your model correctly predicted whether someone has the disease,
    or not about 78.6% of the time for this particular set of data.
    

    F1 Score is the harmonic mean of precision and recall. 
    Precision is the proportion of true positive predictions (i.e., the model correctly predicted the positive class) 
    out of all positive predictions the model made. 
    Recall, on the other hand, is the proportion of true positive predictions out of all actual positive instances. 
    The F1 Score is used when you want to balance these two values - 
    it's particularly useful when the costs of false positives and false negatives are very different 
    (which is often the case in medical scenarios, for instance).
    
    

    An F1 score of 0.785 (or 78.5%) is quite high, suggesting that the model has a good balance of precision and recall. 
    But whether this is "good enough" depends on the specifics of the problem being solved - in some scenarios, 
    even a small number of false positives or negatives could be very costly.

Always remember that the interpretation of these metrics depends heavily on the specific problem and domain. 
For example, in some medical or safety-critical applications, even 99% accuracy might not be considered sufficient. 
On the other hand, in less critical applications, an accuracy of 78.6% could be considered quite good.
"""


# In[31]:


### data normalization


### import the needed package

from sklearn.preprocessing import MinMaxScaler

def normalize_data(df, columns):
    scaler = MinMaxScaler()
    df[columns] = scaler.fit_transform(df[columns])
    return df

# Usage:
# df = normalize_data(df, ['age', 'income'])


# In[32]:


# create a sample dataframe
data = {'Age': [25, 31, 45, 38, 29, 35, 60],
       'Income': [50000, 70000, 80000, 49000, 91000, 77000, 83000]}
df = pd.DataFrame(data)

print("Before Normalization:\n", df)


# In[34]:


### normalize the data
df = normalize_data(df, ['Age', 'Income'])
print("\nNormalize Data:\n", df)


# In[ ]:


"""
The normalize_data function scales the 'age' and 'income' columns so that all values fall between 0 and 1. 
This can be very helpful in many machine learning algorithms, 
which can behave poorly if the input variables have very different scales. 

After normalization, all features will have the same scale. 
This can lead to a better and more stable convergence of the algorithm.
"""

