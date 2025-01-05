# Housing Price Prediction using Linear Regression

## Objective
The objective of this project is to build a predictive model for housing prices using **Linear Regression**. By analyzing various features of the housing dataset, the model predicts the price of a house based on factors such as crime rate, average number of rooms, and other socio-economic factors.

## Dataset
The dataset used in this project is a **Housing Data** dataset, which contains several attributes such as crime rate, proportion of residential land, number of rooms, and more. The target variable to predict is the median value of owner-occupied homes (`MEDV`). 

### Key features in the dataset include:
- **CRIM**: Crime rate per capita
- **ZN**: Proportion of residential land zoned for large lots
- **INDUS**: Proportion of non-retail business acres
- **CHAS**: Charles River dummy variable
- **AGE**: Proportion of owner-occupied homes built before 1940
- **LSTAT**: Percentage of lower status population
- **MEDV**: Median value of homes in $1000s (Target Variable)

## Steps
1. **Data Cleaning**:
    - Load the dataset and check for missing values.
    - Fill missing values with the mean of respective columns.
  
2. **Exploratory Data Analysis**:
    - Calculate and visualize correlation with the target variable `MEDV`.
    - Plot heatmaps and scatter matrices to better understand the relationships between variables.

3. **Outlier Detection**:
    - Detect and remove outliers using Z-scores to improve model performance.

4. **Data Preprocessing**:
    - Drop unnecessary columns such as `CHAS` and `DIS`.
    - Perform feature scaling to standardize the features.

5. **Model Training**:
    - Split the data into training and testing sets.
    - Train a **Linear Regression** model using the training data.

6. **Model Evaluation**:
    - Evaluate the model performance using metrics such as Mean Squared Error (MSE), Root Mean Squared Error (RMSE), and R² Score.
    - Visualize predicted vs actual values to assess the model's accuracy.

7. **Prediction**:
    - Make predictions for a custom set of input values and display the predicted housing price.

## Conclusion
The Linear Regression model was able to predict housing prices based on various features with reasonable accuracy. The model was evaluated using performance metrics such as Mean Squared Error and R² score. Despite some outliers and noise in the data, the model showed promising results and can be further improved with feature engineering or more complex algorithms.

---

**Note**: The dataset used is from the UCI Machine Learning Repository and is publicly available for academic and research purposes.

