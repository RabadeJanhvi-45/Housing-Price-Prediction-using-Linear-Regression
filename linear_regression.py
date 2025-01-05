# -*- coding: utf-8 -*-
"""Linear-Regression.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1A6xJ1bQbl_oNTSd8ZTimVpfJcsWdOs3P
"""

import numpy as np
import pandas as pd
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import seaborn as sns
from scipy import stats

df=pd.read_csv("/content/drive/MyDrive/Semester-2-Practical files/AIML/HousingData.csv")

df.info()

#cleaning data
missing_values = df.isna().sum() #for each column this stores the number of na in an array
print(missing_values)
#no missing values

import pandas as pd

# List of columns with NaN values
na_columns = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'AGE', 'LSTAT']

# Fill NaN values in these columns with the mean of the column
df[na_columns] = df[na_columns].fillna(df.mean())

df

#finding correlation
corr_matrix_medv = df.corr()["MEDV"]

#correlation with target variable
target_corr = np.abs(df.corrwith(df["MEDV"]))#finding absolute
print(target_corr.sort_values(ascending=False)) #printing in descending order

#heatmap
corr_matrix = df.corr()
plt.figure(figsize=(15,10))
sns.heatmap(corr_matrix,cmap="viridis", annot=True)
plt.show()

#scatter matrix
columns = ["MEDV", "LSTAT", "RM", "PTRATIO", "INDUS", "TAX", "NOX","B"]
scatter_matrix(df[columns], alpha=0.5, figsize=(10,10))

df.hist(figsize=(10,10))

#dropping unnecessary columns
df.drop(["CHAS","DIS"], axis=1, inplace=True)
df

# Define threshold
z_threshold = 3

# Filter DataFrame based on z-scores
df_filtered = df.loc[(stats.zscore(df) < z_threshold).all(axis=1)]

df.describe()

#after removing outliers
df.boxplot()
plt.show()

#after removing outliers
df.hist(figsize=(10,10))

#splitting the data into training and testing sets
X = df.drop(["MEDV"], axis=1) #all attributes except the dependent
y = df["MEDV"] #dependent attribute
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

#feature scaling is import before training any data
scaler = StandardScaler()

#only scaling X coz there are multiple rows and y has just one so no need scale it
X_train_scaled = scaler.fit_transform(X_train)#fit_transform is used on training data
X_test_scaled = scaler.transform(X_test)

#Creating Linear Regression Model
model = LinearRegression()
model.fit(X_train_scaled, y_train)

#prediction
y_pred = model.predict(X_test_scaled)
print(y_pred)

#Mean Squared Error
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error is: {mse}")

rmse = mean_squared_error(y_test, y_pred, squared=False)
print(f"Root mean squared error: {rmse}")

#R squared
r2 = r2_score(y_test, y_pred)
print(f"R2 Score: {r2}")

#plotting predicted values against actual values
plt.scatter(y_test,y_pred)
plt.xlabel("Actual values")
plt.ylabel("Predicted values")
plt.title("Actual vs Predicted Values (Linear Regression)")

# Plot a line of best fit
x = [min(y_test), max(y_test)]
y = [min(y_test), max(y_test)]
plt.plot(x, y, 'r')
plt.show()

df

# create a new DataFrame with the custom input values

custom_input = pd.DataFrame({
    'CRIM': [0.147],
    'ZN':[2],
    'INDUS': [8.50],
    'NOX': [0.53],
    'RM': [6.728],
    'AGE': [79.5],
    'RAD': [5],
    'TAX': [385],
    'PTRATIO':[20.9],
    'B':[395.0],
    'LSTAT':[9.42]
})
# scale the input values using the same scaling parameters as the training set
custom_input_scaled = scaler.transform(custom_input)

# make a prediction using the trained model
prediction = model.predict(custom_input_scaled)

# print the predicted value
print("Predicted value:", prediction[0])