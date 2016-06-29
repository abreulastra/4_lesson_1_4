# -*- coding: utf-8 -*-
"""
Created on Mon Jun 27 23:36:33 2016

@author: AbreuLastra_Work
"""

import numpy as np
import statsmodels.formula.api as smf
import pandas as pd
from sklearn.metrics import mean_squared_error

# Set seed for reproducible results
np.random.seed(414)

# Gen toy data
X = np.linspace(0, 15, 1000)
y = 3 * np.sin(X) + np.random.normal(1 + X, .2, 1000)

train_X, train_y = X[:700], y[:700]
test_X, test_y = X[700:], y[700:]

train_df = pd.DataFrame({'X': train_X, 'y': train_y})
test_df = pd.DataFrame({'X': test_X, 'y': test_y})

# Linear Fit
poly_1 = smf.ols(formula='y ~ 1 + X', data=train_df).fit()

# Quadratic Fit
poly_2 = smf.ols(formula='y ~ 1 + X + I(X**2)', data=train_df).fit()


#Test

results_ols = poly_1.summary()
params_ols = poly_1.params
ols_predict = poly_1.predict(train_df['X'])
print (results_ols)
print(params_ols)


#Quadratic model
results_quad = poly_2.summary()
params_quad = poly_2.params
quad_predict = poly_2.predict(train_df['X'])
print (results_quad)
print(params_quad)

## similarly if you use test set

ols_predict_test = poly_1.predict(test_df['X'])



#Quadratic model
quad_predict_test= poly_2.predict(test_df['X'])
 
print('so far so good')
#Evaluation with training and test mean-square error (MSE)
#Linear model
mean_squared_error(train_df['y'], ols_predict[:700])
mean_squared_error(test_df['y'], ols_predict_test[700:])
#Quadratic model
mean_squared_error(train_df['y'], quad_predict[:700])
mean_squared_error(test_df['y'], quad_predict_test[700:])

#Conclusion: when we use the trainig set MSE is higher with linear estimates, lower with quadratic. 
#but when we us the test dataset, linear model performs better than quadratic, hinting at t overfitting issue



