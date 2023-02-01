# Crypto-Predict

Time series forecasting bitcoin price.<br/>
Types of time series patterns: Trend, Seasonal, Cyclic.<br/>
Types of time series data: Univariate, Multivariate.

## Dataset
Bitcoin dataset:<br/>
* [Dataset 1 - Year 2013 to 2021](https://github.com/Logeswaran123/Crypto-Predict/blob/main/dataset/BTC_USD_2013-10-01_2021-05-18-CoinDesk.csv)
* [Dataset 2 - Year 2014 to 2023](https://github.com/Logeswaran123/Crypto-Predict/blob/main/dataset/BTC-USD.csv)

## How to run
```python
python run.py --data <path to dataset> --exp <enable experiments>
```
Argumets:<br/>
<path to dataset\> - Path to Dataset directory.<br/>
<enable experiments\> - Enable model experiments. True or False. Default: False.

## Experiments
Following are test results with different modelling experiments, <br/>
```
TODO
```

## References/Materials
* Time series forecasting in TensorFlow | [Colab](https://colab.research.google.com/github/mrdbourke/tensorflow-deep-learning/blob/main/10_time_series_forecasting_in_tensorflow.ipynb#scrollTo=vlVtweEv7nAx) | [Book](https://dev.mrdbourke.com/tensorflow-deep-learning/10_time_series_forecasting_in_tensorflow/)
* [What can be forecast?](https://otexts.com/fpp3/what-can-be-forecast.html#what-can-be-forecast)
* [Time series patterns](https://otexts.com/fpp3/tspatterns.html)
* [Evaluating point forecast accuracy](https://otexts.com/fpp3/accuracy.html)
* [Fast and Robust Sliding Window Vectorization with NumPy](https://towardsdatascience.com/fast-and-robust-sliding-window-vectorization-with-numpy-3ad950ed62f5)
* [How (not) to use Machine Learning for time series forecasting: Avoiding the pitfalls](https://towardsdatascience.com/how-not-to-use-machine-learning-for-time-series-forecasting-avoiding-the-pitfalls-19f9d7adf424)
* N-BEATS: Neural basis expansion analysis for interpretable time series forecasting | [Paper](https://arxiv.org/pdf/1905.10437.pdf)
* [Uncertainty in Deep Learning — Brief Introduction](https://towardsdatascience.com/uncertainty-in-deep-learning-brief-introduction-1f9a5de3ae04)
* [Subway vs. Coconut uncertainty](https://www.noahbrier.com/archives/2016/01/subway-uncertainty-vs-coconut-uncertainty/)
