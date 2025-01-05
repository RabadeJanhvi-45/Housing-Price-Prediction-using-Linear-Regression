# Housing Price Prediction using Linear Regression

This project demonstrates how to build a machine learning model using linear regression to predict housing prices based on various features. The dataset used contains various features such as crime rate, number of rooms, property tax rate, and others, which are used to predict the median house value (`MEDV`).

## Table of Contents

- [Project Overview](#project-overview)
- [Libraries Used](#libraries-used)
- [Dataset](#dataset)
- [Data Cleaning](#data-cleaning)
- [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
- [Modeling](#modeling)
- [Model Evaluation](#model-evaluation)
- [Custom Predictions](#custom-predictions)
- [Conclusion](#conclusion)

## Project Overview

In this project, we:
1. Preprocess and clean the housing dataset.
2. Perform exploratory data analysis (EDA) to understand the relationships between features and target variable.
3. Apply feature scaling to normalize the data.
4. Build and train a Linear Regression model.
5. Evaluate the model using key performance metrics such as Mean Squared Error (MSE), Root Mean Squared Error (RMSE), and \( R^2 \) score.
6. Make predictions on custom input data.

## Libraries Used

- **pandas**: Data manipulation and analysis.
- **numpy**: Numerical computations.
- **matplotlib**: Plotting and visualization.
- **seaborn**: Advanced data visualization.
- **sklearn**: Machine learning library for model building, training, and evaluation.
- **scipy**: Statistical analysis (used for z-score calculation).

## Dataset

The dataset used in this project is a housing dataset that includes the following columns:

- `CRIM`: Crime rate
- `ZN`: Proportion of residential land zoned for large lots
- `INDUS`: Proportion of non-retail business acres per town
- `CHAS`: Charles River dummy variable (1 if tract bounds river; 0 otherwise)
- `NOX`: Nitrogen oxides concentration (parts per 10 million)
- `RM`: Average number of rooms per dwelling
- `AGE`: Proportion of owner-occupied units built before 1940
- `DIS`: Weighted distance to employment centers
- `RAD`: Index of accessibility to radial highways
- `TAX`: Property tax rate per $10,000
- `PTRATIO`: Pupil-teacher ratio by town
- `B`: Proportion of residents of African American descent
- `LSTAT`: Percentage of lower status population
- `MEDV`: Median value of owner-occupied homes (target variable)

## Data Cleaning

- Missing values in columns (`CRIM`, `ZN`, `INDUS`, `CHAS`, `AGE`, and `LSTAT`) are imputed using the mean of the respective columns.
- Unnecessary columns (`CHAS` and `DIS`) are dropped based on correlation analysis.

## Exploratory Data Analysis (EDA)

- A correlation matrix is created to identify strong relationships between features and the target variable (`MEDV`).
- A heatmap and scatter matrix are visualized to understand the data distribution and relationships.
- Histograms of all features are plotted to analyze their distributions.

## Modeling

- The dataset is split into training and testing sets using an 80/20 split.
- Feature scaling is performed using `StandardScaler` to normalize the input data.
- A **Linear Regression** model is created and trained on the scaled data.

## Model Evaluation

The model is evaluated using the following metrics:
- **Mean Squared Error (MSE)**: 40.15
- **Root Mean Squared Error (RMSE)**: 6.34
- **R² Score**: 0.51

These results suggest that the model explains about 51% of the variance in the dataset and has moderate prediction accuracy.

## Custom Predictions

- A custom input set can be provided for prediction.
- The input features are scaled using the same scaler as the training data before making predictions.

### Example Code for Custom Input:
```python
custom_input = np.array([{
    'TAX': [385],
    'PTRATIO': [20.9],
    'B': [395.0],
    'LSTAT': [9.42]
}])

custom_input_scaled = scaler.transform(custom_input)
prediction = model.predict(custom_input_scaled)
