# Forecasting

Let's take a Time series with Trends + Noise + Seasonality.

## 1. Naive Forecasting

- For example we could take last values and assume that next value will be same one.

1. Fixed Partitioning

- To measure performance to train for time series, we can partition into 3 time series.
    1. Training Period: Train model
    2. Validation Period: Evaluate model, here to experiment to find right architecture for training. And also the hyper
       parameters until we get desired performance using validation set. Once Done that, we can retrain on training &
       validation data.
    3. Testing Period: Test model to evaluate performance, after that, we can retrain using also the test data. It is
       because, test data is closest to latest data we have. Hence, offers strongest signal in determining future
       values.
- If there is seasonality in our time series, we must include whole number of seasons in each period. For eg, 1 year or
  2 years for yearly seasonality.

    <img src="images/Fixed%20Partitioning%20for%20Forecasting.jpg" width="500">

1. Roll-Forward Partitioning
    - We start with short period for **Training period** and gradually increase it by one day at a time or one week at a
      time.
    - At each iteration, we train our model on a training period, so we forecast data of a following day or week in **
      Validation period**.
    - Thus roll forward partitioning can be seen as doing fixed partitioning a number of times and continually refining
      the model as such.

    <img src="images/Roll%20Forward%20Partitioning.jpg" width="500">

## 2. Moving Average

- Idea here is yellow line is plot for average of blue values over fixed period called **Averaging window** (like 30
  days).
- This nicely eliminates lot of noise, and gives a curve emulating original series but does not anticipate trend or
  seasonality.

  <img src="images/Moving%20Average%20Forecasting.jpg" width="500">

- Depending on period after which you want to forecast for future can be worse than naive forecasting.
- To avoid this, we can remove the trend & seasonality, using a technique called **Differencing**. So instead of
  studying time series it self, we study difference b/w value at time T and value at earlier period. So depending on
  time of data(like a day, month or a year).
  <img src="images/Moving%20Average%20on%20differences%20TS.jpg" width="500">

- Hence we get a time series based on average of the forecasts for difference time series.
- Now to get final forecasts based on original time series, we add back the values at time T-minus 365.
  <img src="images/Restored%20trend%20and%20seasonality.jpg" width="500">
- Now upon this forecast, if we try to remove noise based on original forecast we get smoother data.
  <img src="images/Smoothing%20Pats%20and%20present%20values.jpg" width="500">

> If we use trailing window to compute moving average ie T-30 to T-1 and when we use centered window to compute ie T-1year minus 5 days to T-1year plus 5 days, then centered windows can be more accurate.
> But we can't use centered window to smooth present values since we don't know future values.

# Metrics to calculate performance

1. errors = forecasts_values - actual_values
    - Over evaluation period.

2. mse = np.square(errors).mean()
    - Mean squared error, We square to get rif of negative values.
    - So for example, if large values are potentially dangerous & may cost more than smaller errors, then mse is
      preferred.

3. rmse = np.sqrt(mse)
    - If we want the mean of our errors calculation to be on same scale as original errors, we then sq root the mse, to
      get Root mean squared values.

4. mae = np.abs(errors).mean()
    - Mean absolute error/Mean absolute deviation(mad), instead of squaring, we use absolute values, this doesn't
      penalize large values as mse does.
    - If gain/loss is just proportional to size of errors, then mae may be better.

5. mape = np.abs(errors / x_valid).mean()
    - Mean absolute percentage error, which is mean ratio b/w absolute error and absolute value.
    - It gives idea of size of errors compared to values.

```python
keras.metrics.mean_absolute_error(x_valid, naive_forecast).numpy()
```
