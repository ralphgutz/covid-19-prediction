## COVID-19 Analysis & Prediction using Regression & ARIMA Techniques

Linear and Polynomial Regression and Time Series Analysis (ARIMA Model) applied to COVID-19 Philippines dataset of [_Our World in Data_](https://github.com/owid). Composed of simple analyses and prediction models of daily and total cases, 7-day moving average, and positivity rate. 

### Model Parameters
Parameters can be changed to fit the current dataset.

- Linear Regression (Ridge Regression can be applied)
- Polynomial Regression (5th, 6th, and 7th degrees)
- Auto Regressive Integrated Moving Average - ARIMA(1, 1, 1) and ARIMA(0, 2, 3) from auto_arima
- Moving Average (7-day interval)

### Metrics
Data split are discarded to fit the model. Coefficient of Determination (R^2), Root Mean Squared Error, and Mean Absolute Percentage Error are used to identify the models' errors.

_Note: This is just an initial analysis and prediction. High error values may come out. Model tuning should be applied for future datasets._
