# Rossmann Store Sales Prediction

This project was inspired by the "Rossmann Store Sales" challenge published on kaggle (https://www.kaggle.com/c/rossmann-store-sales). 

It is a fictitious project but with all the steps of a real project.

## Business scenario
Rossmann operates over 3,000 drug stores in 7 European countries. Currently, Rossmann store managers are tasked with predicting their daily sales for up to six weeks in advance. Store sales are influenced by many factors, including promotions, competition, school and state holidays, seasonality, and locality. With thousands of individual managers predicting sales based on their unique circumstances, the accuracy of results can be quite varied.

The sales director of the Rossmann stores wants to estimate the sales forecast for the next 6 weeks on its different units spread across Europe.

## Solution methodology
The solution to this business problem is a machine learning that forecasts sales for the next 6 weeks. 

The resolution of the challenge was carried out following the CRISP (CRoss-Industry Standard Process for data mining) methodology, which is a cyclical approach that streamlines the delivery of value.

![crisp_cycle](/img/CRISP.png)

## Steps:
### 1 - Data Description
Cleaning of data and construction of statistical metrics in order to have a more holistic view of the business model and make better decisions throughout the project.

### 2 - Feature Engineering
Deepening knowledge about the given dataset attributes and deriving new features to help build the model. As a by-product, get a list of hypotheses for the purpose of generating insights for the business team.

### 3 - Filtering
Filtering the data for model learning.

### 4 - Exploratory Data Analysis
Validation of hypotheses in order to generate insights, understanding the correlation of variables and their impact on the model, and understanding the behavior of the response variable on the other variables.

### 5 - Data Preparation
Resizing numeric variables and encoding categorical variables so that machine learning models can learn specific behavior.

### 6 - Feature Selection
Selection of the most relevant variables for learning the model.

### 7 - Machine Learning Models
Training Machine Learning models, applying cross-validation and choosing the best model.

### 8 - Hyperparameters Fine Tunning
Choice of the best parameters to refine the model chosen in the previous step.

### 9 - Error Translation and Interpretation
Conversion of model performance into business results.

### 10 - Model Deployment to Production
Publishing the model to production so that the business team can use the service from anywhere.

## Exploratory data analysis (EDA) guided by a mind map of hypotheses
A mind map of hypotheses was made in order to guide the EDA, to generate insights and to understand a little more about the database and the most important attributes

![MINDMAP](/img/roadmap.png)

## Insights: summary of the analysis of hypotheses 
With the feature diagram above, several and main hypotheses were generated: 

### 1. Stores with a larger assortment should sell more
False - Stores with bigger assortment sell less

![H1.1](/img/h1-1.png)

![H1.2](/img/h1-2.png)

### 2. Stores with closer competitors should sell less
False - Stores with closer competitors sell more

![H2](/img/h2.png)

### 3. Stores with longer-standing competitors should sell more
False - Store with longer competitors sell less

![H3](/img/h3.png)

### 4. Stores where products cost less for longer (active promotions) should sell more
False - Stores with longer active promotions sell less after a certain period of promotion

![H4](/img/h4.png)

### 5. Stores with more promotion days should sell more
FALSE - Stores with more consecutive promotions sell less

![H5](/img/h5.png)

### 6. Stores with more extended promotions should sell more
FALSE - Stores open during the Christmas holiday sell less

![H6](/img/h6.png)

### 7. Stores should sell more over the years
FALSE - Stores sell less over the years

![H7](/img/h7.png)

### 8. Stores should sell more in the second half of the year
FALSE - Stores sell less in the second half of the year

![H8](/img/h8.png)

### 9. Stores should sell more after the 10th of each month
TRUE - Stores sell more after the 10th of each month.

![H9](/img/h9.png)

### 10. Stores should sell less on weekends
TRUE - Stores sell less on weekends

![H10](/img/h10.png)

### 11. Stores should sell less during school holidays.
TRUE - Stores sell less during school holidays, except July and August

![H11](/img/h11.png)

## Machine learning - Real Performance - Cross Validation
Four different models (linear regression, regularized linear regression - lasso, random forest and XGBoost ) were evaluated using the cross-validation on a rolling basis:

![CROSS VALIDATION](/img/ts_cross_validation.png)

It started with a reduced portion of the training database, whose last 6 weeks were separated for validation; then, the model was trained and its performance was calculated.
New iterations were performed, each time increasing the training dataset and always separating the last 6 weeks for the test.
The cross-validation performance was the average of each of these iterations.

The results in terms of Mean Absolute Error (MAE), Mean absolute percentage error (MAPE) and Root Mean Square Error (RMSE) were:

![MAE, MAPE AND RMSE](/img/cross-validation.png)

Although the random forest model was the best, the model chosen to go ahead with the tuning of the hyperparameters was XGBoost. 

The reason for this is that it is a much lighter model to operate in production and does not have a significant difference in performance; the operability in production is an extremely important requirement in this project.

## Hyperparameter tuning
After choosing the XGBoost Regression, the Fine Tuning of the model was performed, in which it is possible to observe the following performance:

![XGBoost Regression](/img/xgboost.png)

## Business Performance
Considering the MAE obtained in the forecast for each store, during the test period, the best and worst sales scenarios for each store were projected.

Below, the expected business performance of the first 5 stores:

![Business Performance](/img/business_performance.png)

![Business Performance - MAPE](/img/business_performance_mape.png)

From the test set, we obtain the following financial results:

![Scenario - MAPE](/img/scenario.png)

In other words, we have a margin of error of R$1,000.00 for both the best and worst case scenarios. Therefore, we have an excellent performance, as there is a small margin of error compared to the total amount that is in the range of R$ 290,000.00.

## Conclusion
Finally, in this project, we achieved the objective of forecasting the next 6 weeks of sales for each Rossmann chain store and, in the general context, we obtained an excellent performance from our model in the first cycle.

In addition, in problems involving regression it is relevant that we apply linear (Linear Regression and Lasso) and non-linear (Random Forest Regressor and XGBoost Regressor) behavior models so that we can understand how our data behaves. 

Based on this information, choose the best model to be implemented so that you can obtain the best possible performance. At the time of fine tuning, you can make small parameter adjustments to improve accuracy and achieve better financial returns for the business.

## Next steps
For future projects, it is possible to have new approaches to work with the following points:

1. Implementation of the Random Forest Regestor model
2. Selection of new features to improve model performance
3. Improved model performance in stores where accurace was very low
4. Validation of other hypotheses
5. Collect usability feedback
6. Increase model accuracy by 10%
