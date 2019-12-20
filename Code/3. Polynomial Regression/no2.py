#!/usr/bin/env python
# coding: utf-8

# In[1]:


import polyEngine


# In[8]:


filename = "Data\merged_data_DminusOne.csv"
col_list = ['Day No.','Air Temperature', 'Pressure Station Level','Horizontal Visibility', 'D-1 NO2'];
target_y = ['NO2']
test_size = 0.2
degree = 2
engine = polyEngine.polyEngine(filename, col_list, target_y, test_size, degree)
poly_regressor = engine.runEngine()
error = engine.getStatusLog()

