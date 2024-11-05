# -*- coding: utf-8 -*-
"""
Created on Wed Nov 16 16:16:56 2022

@author: tanish
"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


#----------------------------------------------------------------------------
# DATA IMPORT ###############################################################
#----------------------------------------------------------------------------



import os
cwd = os.getcwd()
print(cwd)
os.chdir('C:/Users/tanis')

data= pd.read_csv("data.csv", index_col=0)



#-------------------------------------------------------------------------
#2. DATA CLEANING ########################################################
#-------------------------------------------------------------------------


# Checking is any data has a null value in it.  
numberMissing=data.isnull().sum()
# we found out there is no null value hence we dont have to change anything.

# Checking if minimym room is negative or not
print(data.ROOM.min())
# Min room is 1 hence we dont have to remove anything

# checking how many rooms are there in the data
print(data.ROOM.unique())
# My brother only wanted to have a look at 7 or less room appartments. 
# Hence we will remove all the data which have rooms more than 7
data = data.drop(data[data.ROOM>7].index)

# Checking if we successfully removed all rooms by counting rooms. 
numberRooms = data.ROOM.value_counts()
print(numberRooms)

# Checking min room so that there is no negative
print(data.BEDROOMS.min())
# Checking unique rooms 
print(data.ROOM.unique())
# we found 7 which is perfect hance we go ahead without changing anything

# Checking min price of appartment hope none is negative.
print(data.PRICE.min())
print(data.PRICE.unique())
# it shows min price is 1. we dont have negative values but rooms with 50 or less per month can be fake.
# Hence removing data with price less than 50. 
data = data.drop(data[data.PRICE<50].index)
# Checking if successfully removed.
data[data.PRICE < 50].PRICE.count()

# Checking if any appartment has cost more than 900 as it is out of budget.
# And dropping those values
data[data.PRICE > 900].PRICE.count()
data = data.drop(data[data.PRICE>900].index)

# Checking min area of the appartment
print(data.AREA.min())
# none is negative hence we dont have to remove anything 

# Checking year built of the appartment and removing ones under 1800 as they are very old
data[data.YEAR_BUILT < 1800].PRICE.count()
data = data.drop(data[data.YEAR_BUILT<1800].index)

# Checking if any appartment is made after 2022 as it can be fake. 
data[data.YEAR_BUILT > 2022].PRICE.count()
# none were found

# We dont need address and hence we will drop it from the table.
data.drop('ADDRESS', axis = 1, inplace = True)
# We also dont need POSTAL_CODE and hence we will drop it from the table.
data.drop('POSTAL_CODE', axis = 1, inplace = True)




#----------------------------------------------------------------------------
# FEATURE ENGINEERING #######################################################
#----------------------------------------------------------------------------

#Data relates to ..
data.info()
data.head()
data.describe()

# Include details on variables
#     Column         Non-Null Count  Dtype  
#---  ------         --------------  -----  
# 0   YEAR_BUILT     2451 non-null   int64      - Predictor - Numerical
# 1   AREA           2451 non-null   int64      - Predictor - Numerical
# 2   PRICE          2451 non-null   float64    - Response  - Numerical (So use Regression Methods)
# 3   BEDROOMS       2451 non-null   int64      - Predictor - Numerical
# 4   ROOM           2451 non-null   int64      - Predictor - Numerical
# 5   PROPERTY_TYPE  2451 non-null   object     - Predictor - Categorical
# dtypes: float64(1), int64(4), object(1)

######### Feature Engineering: Drop certain variables if not required
# Keep all as it is already done in data cleaning process.
#########Feature Engineering: Scale Data
# Not needed here


#########Feature Engineering: Construct New Variables if required
#Change Type to numerical giving 1 to appartment and 0 to house                                                           ,,,,,,,,
data['HOUSE_OR_APPARTMENT']= np.where(data.PROPERTY_TYPE=='apartment',1,0)


#Produce scatter and correlation plots - Pay particular attention to the Response variable
figure(num=None, figsize=(8, 6), dpi=80, facecolor='w', edgecolor='k')
figure(num=None, figsize=(8, 6), dpi=80, facecolor='w', edgecolor='k')
sns.pairplot(data)
# We can identify an increasing graph between area and price

#Correlations
figure(num=None, figsize=(8, 6), dpi=80, facecolor='w', edgecolor='k')
sns.heatmap(data.corr(), annot=True, cmap = 'Reds')
plt.show()
# more area results in high rent price
# More rooms result in high rent price
# Least important is Year Built

#Box plot
figure(num=None, figsize=(8, 8), dpi=80, facecolor='w', edgecolor='k')
sns.boxplot(data.PROPERTY_TYPE, data.PRICE)
plt.show()
# We can see that houce have higher price than appartments, which shows this veriable is important.

corrVals=data.corr()

######### Feature Engineering: Multicolinearity
#You will probably end up dropping the BEDROOMS as it is similar to ROOMS.
#ROOMS and Price are almost perfectly correlated as rental price increases with 
# the increase in not just the bedroom but also additional rooms sutch as storage rooms.
data.drop('BEDROOMS', axis = 1, inplace = True)


#----------------------------------------------------------------------------
# REGRESSION MODELLING ######################################################
#----------------------------------------------------------------------------

######### Split Data into Train and Test
#Setting the Response and the predictor variables

x = data[['YEAR_BUILT', 'AREA', 'ROOM', 'HOUSE_OR_APPARTMENT']] #pandas dataframe
y = data['PRICE'] #Pandas series

#split train 66.7%, test 33.3%. 
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.333)


#----------------------------------------------------------------------------
# REGRESSION MODELLING : Model Selection ####################################
#----------------------------------------------------------------------------

model1 = LinearRegression()
model2 = LinearRegression()
model3 = LinearRegression()
model4 = LinearRegression()
model5 = LinearRegression()

#Fitting the variables in order of strongest correlation with Price and 
#calculate adjusted R squared in each step.




#---------------------Model 1 - First add AREA to model------------------------
model1.fit(x_train[['AREA']], y_train)
#Show the model parameters
coeffM1 = pd.DataFrame(model1.coef_, ['Average Increasing Area Of Property'], columns = ['Coeff'])
interceptM1 = pd.DataFrame(model1.intercept_, ['intercept'], columns = ['Intercept'])

print(coeffM1)
print(interceptM1)
# As PRICE = inter+(coeff-n*Predictors-n)
# PRICE = 52.36+(3.35*AREA)

#Generate predictions for the train data
predictions_train = model1.predict(x_train[['AREA']])
raw_sum_sq_errors = sum((y_train.mean() - y_train)**2)
prediction_sum_sq_errors = sum((predictions_train - y_train)**2)
Rsquared1 = 1-prediction_sum_sq_errors/raw_sum_sq_errors
# Rsquared1 = 0.4674862381407572

N=len(y_train) #1634 data rows
p=1 # one predictor used

Rsquared_adj1 = 1 - (1-Rsquared1)*(N-1)/(N-p-1)
print("Rsquared Regression Model with AREA: "+str(Rsquared1))
print("Rsquared Adjusted Regression Model with AREA: "+str(Rsquared_adj1))

# RSquaredAdj = 0.46715 Explained 46.71% of the variation or error




#---------------------Model 2 - Next adding room variable----------------------

model2.fit(x_train[['AREA', 'ROOM']], y_train)

#Show the model parameters
coeffM2 = pd.DataFrame(model2.coef_, ['AREA Coeff', 'ROOM Coeff'], columns = ['Coeff'])
interceptM2 = pd.DataFrame(model2.intercept_, ['AREA Intercept', 'ROOM Intercept'], columns = ['Intercept'])
print(coeffM2)
print(interceptM2)

# As PRICE = inter+(coeff-n*Predictors-n)
# PRICE = 64.32485 + (3.83619 * AREA) + (-19.71299 * ROOM)

#Generate predictions of model 2 for the train data
predictions_train = model2.predict(x_train[['AREA', 'ROOM']])
raw_sum_sq_errors = sum((y_train.mean() - y_train)**2)
prediction_sum_sq_errors = sum((predictions_train - y_train)**2)
Rsquared2 = 1-prediction_sum_sq_errors/raw_sum_sq_errors

N=len(y_train) 
p=2 # As we use two predictors
Rsquared_adj2 = 1 - (1-Rsquared2)*(N-1)/(N-p-1)
print("Rsquared Regression Model with AREA & ROOM: "+str(Rsquared2))
print("Rsquared Adjusted Regression Model with AREA & ROOM: "+str(Rsquared_adj2))

# RSquaredAdj = 0.48146 explained 48.15% of the variation or error
# This valie is a slight better than the previous value.




#------------Model 3 - Next adding HOUSE_OR_APPARTMENT variable----------------

model3.fit(x_train[['AREA','ROOM', 'HOUSE_OR_APPARTMENT']], y_train)

#Show the model parameters
coeffM3 = pd.DataFrame(model3.coef_, ['AREA Coeff', 'ROOM Coeff', 'HOUSE_OR_APPARTMENT Coeff'], columns = ['Coeff'])
interceptM3 = pd.DataFrame(model3.intercept_, ['AREA Intercept', 'ROOM Intercept', 'HOUSE_OR_APPARTMENT Intercept'], columns = ['Intercept'])
print(coeffM3)
print(interceptM3)

# As PRICE = inter+(coeff-n*Predictors-n)
# PRICE = -73.61808 + (3.847563 * AREA) + (0.105023 * ROOM) + (95.99397 * HOUSE_OR_APPARTMENT)

#Generate predictions of model 3 for the train data
predictions_train = model3.predict(x_train[['AREA', 'ROOM', 'HOUSE_OR_APPARTMENT']])
raw_sum_sq_errors = sum((y_train.mean() - y_train)**2)
prediction_sum_sq_errors = sum((predictions_train - y_train)**2)
Rsquared3 = 1-prediction_sum_sq_errors/raw_sum_sq_errors

N=len(y_train)
p=3 # As we have 3 predictors now
Rsquared_adj3 = 1 - (1-Rsquared3)*(N-1)/(N-p-1)
print("Rsquared Regression Model with AREA & ROOM & HOUSE_OR_APPARTMENT: "+str(Rsquared3))
print("Rsquared Adjusted Regression Model with AREA & ROOM & HOUSE_OR_APPARTMENT: "+str(Rsquared_adj3))

# RSquaredAdj = 0.50940 explained 50.94% of the variation or error
# This valie is a slight better than the previous value.



#-----------------Model 4 - Next adding YEAR_BUILT variable--------------------

model4.fit(x_train[['AREA','ROOM', 'HOUSE_OR_APPARTMENT', 'YEAR_BUILT']], y_train)

#Show the model parameters
coeffM4 = pd.DataFrame(model4.coef_, ['AREA Coeff', 'ROOM Coeff', 'HOUSE_OR_APPARTMENT Coeff', 'YEAR_BUILT Coeff'], columns = ['Coeff'])
interceptM4 = pd.DataFrame(model4.intercept_, ['AREA Intercept', 'ROOM Intercept', 'HOUSE_OR_APPARTMENT Intercept', 'YEAR_BUILT Intercept'], columns = ['Intercept'])
print(coeffM4)
print(interceptM4)

# As PRICE = inter+(coeff-n*Predictors-n)
# PRICE = 3218.94397 + (4.39588 * AREA) + (-5.34687 * ROOM) + (68.43051 * HOUSE_OR_APPARTMENT) + (-1.68424 * YEAR_BUILT)

#Generate predictions of model 4 for the train data
predictions_train = model4.predict(x_train[['AREA', 'ROOM', 'HOUSE_OR_APPARTMENT', 'YEAR_BUILT']])
raw_sum_sq_errors = sum((y_train.mean() - y_train)**2)
prediction_sum_sq_errors = sum((predictions_train - y_train)**2)
Rsquared4 = 1-prediction_sum_sq_errors/raw_sum_sq_errors

N=len(y_train)
p=4 # As we have 4 predictors now
Rsquared_adj4 = 1 - (1-Rsquared4)*(N-1)/(N-p-1)
print("Rsquared Regression Model with AREA & ROOM & HOUSE_OR_APPARTMENT & YEAR_BUILT: "+str(Rsquared4))
print("Rsquared Adjusted Regression Model with AREA & ROOM & HOUSE_OR_APPARTMENT & YEAR_BUILT: "+str(Rsquared_adj4))

# RSquaredAdj = 0.64715 explained 64.72% of the variation or error
# This valie is a better than the previous value.




# After looking at all Rsquared_adj values we find that the best value is of 
# Model 4 and hence we move forward with Rsquared_adj of model 4.

Output = pd.DataFrame(model4.coef_, ['AREA Coeff', 'ROOM Coeff', 'HOUSE_OR_APPARTMENT Coeff', 'YEAR_BUILT Coeff'], columns = ['Coeff'])
print(Output)
print("Intercept = " + str(model4.intercept_))

# PRICE = 3218.94397 + (4.39588 * AREA) + (-5.34687 * ROOM) + (68.43051 * HOUSE_OR_APPARTMENT) + (-1.68424 * YEAR_BUILT)




#----------------------------------------------------------------------------
# REGRESSION MODELLING : Model Evaluation Based on TEST set #################
#----------------------------------------------------------------------------

# Calculate the Mean Absolute Error and the Root Mean Square Error for the 
# model 4 based on the TEST set.

#MAE - Mean Absolute Error
#MAPE - Mean Absolute Percentage Error
#RMSE - Root Mean Square Error

predictions_test = model4.predict(x_test[['AREA','ROOM', 'HOUSE_OR_APPARTMENT', 'YEAR_BUILT']])

#MAE - Mean Absolute Error
Prediction_test_MAE = sum(abs(predictions_test - y_test))/len(y_test)
#MAPE - Mean Absolute Percentage Error
Prediction_test_MAPE = sum(abs((predictions_test - y_test)/y_test))/len(y_test)
#RMSE - Root Mean Square Error
Prediction_test_RMSE = (sum((predictions_test - y_test)**2)/len(y_test))**0.5

print(Prediction_test_MAE) #82.447. So error in prediction for the test prices is 82.45 Euro.
# This above prediction shows me that the agents are surely going to make profit of rental business.
print(Prediction_test_MAPE)  #0.28140. So 28.14% Error on average
print(Prediction_test_RMSE)



#----------------------------------------------------------------------------
# PLOT PREDICTION RESULTS ###################################################
#----------------------------------------------------------------------------

#First plot the y test values and the predictions for the model
figure(num=None, figsize=(8, 8), dpi=80, facecolor='w', edgecolor='k')
plt.scatter(y_test, predictions_test)
plt.title("Predictions v Actual Test Values")
plt.xlabel("Actual values")
plt.ylabel("Predicted Values")
plt.show() 

figure(num=None, figsize=(8, 8), dpi=80, facecolor='w', edgecolor='k')
plt.scatter(y_test, predictions_test - y_test)
plt.title("Errors v Actual Test Values")
plt.xlabel("Actual values")
plt.ylabel("Error Values")
plt.show()
