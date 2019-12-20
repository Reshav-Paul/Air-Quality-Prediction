#!/usr/bin/env python
# coding: utf-8

# In[1]:


import polyEngine


# In[12]:


filename = "Data\merged_data_DminusOne.csv"
col_list = ['Day No.','Air Temperature', 'Pressure Station Level','Dew Point Temperature',
            'Relative Humidity', 'Horizontal Visibility','D-1 PM10'];
target_y = ['PM10']
test_size = 0.2
degree = 2
engine = polyEngine.polyEngine(filename, col_list, target_y, test_size, degree)
poly_regressor = engine.runEngine()
error = engine.getStatusLog()

