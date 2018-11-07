import datetime, xlrd
import itertools

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# from fbprophet import Prophet
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima_model import *
import statsmodels.api as sm


# Time series analysis using
# https://www.analyticsvidhya.com/blog/2016/02/time-series-forecasting-codes-python/
# https://towardsdatascience.com/an-end-to-end-project-on-time-series-analysis-and-forecasting-with-python-4835e6bf050b


file_location1 = "C:\\Users\\Tanguy\\Documents\\ABP private\\PLaneTicket\\ryanair_fare_data.xlsx"
file_location2 = "C:\\Users\\Tanguy\\Documents\\ABP private\\PLaneTicket\\AF U2 fares.xlsx"
odf1 = pd.read_excel(file_location1)
odf2 = pd.read_excel(file_location2)
Ryanair = odf1
ndf2 = odf2

easyJet = pd.DataFrame(ndf2[ndf2['Carrier'] == 'easyJet'])
AirFrance = pd.DataFrame(ndf2[ndf2['Carrier'] == 'Air France'])

prices = Ryanair.merge(easyJet, how='outer', on='Date')
prices = prices.merge(AirFrance, how='outer', on='Date')

prices.rename(columns={'WeightedAverage_x': 'RyanairWA', 'WeightedAverage_y': 'easyJetWA', 'WeightedAverage': 'AirFranceWA'}, inplace=True)

prices = prices.loc[:, ['Date', 'RyanairWA', 'easyJetWA', 'AirFranceWA']]

prices.drop(index=[len(prices.index) - 1, len(prices.index) - 2], inplace=True)
prices['Date'] = pd.to_datetime(prices.Date)

prices = prices.set_index('Date')
prices.sort_index(inplace=True)

###################################
# # analysis of the data
#
# def analysis(ts):
#     meanWA = ts.rolling(12).mean()
#     stdWA = ts.rolling(12).std()
#     plt.plot(ts.RyanairWA, 'r', label='RyanairWA')
#     plt.plot(ts.easyJetWA, 'b', label='easyJetWA')
#     plt.plot(ts.AirFranceWA, 'g', label='AirFranceWA')
#     plt.plot(stdWA.RyanairWA, 'r', label = 'RyanairWA')
#     plt.plot(stdWA.easyJetWA, 'b', label = 'easyJetWA')
#     plt.plot(stdWA.AirFranceWA, 'g', label = 'AirFranceWA')
#     plt.plot(meanWA.RyanairWA, 'r', label = 'RyanairWA')
#     plt.plot(meanWA.easyJetWA, 'b', label = 'easyJetWA')
#     plt.plot(meanWA.AirFranceWA, 'g', label = 'AirFranceWA')
#     plt.title('Fare WeightedAverage')
#     plt.ylabel('Price (euro)')
#     plt.xlabel('Year')
#     plt.legend()
#     plt.show()
#
# # logWA = np.log(prices)
# #
# # rollMeanWA = logWA.rolling(12).mean()
# #
# # diffLogMA = logWA - rollMeanWA
# #
# # exwaWA = logWA.ewm(halflife=12).mean()
# # logexwaDiffWA = logWA - exwaWA
#
# #########
# # differencing
# diffLogShift = logWA-logWA.shift()
#
# #########
# # decomposing
# # RA = logWA.RyanairWA
# # RA.dropna(inplace=True)
# #
# # AF = logWA.AirFranceWA
# # AF.dropna(inplace=True)
# #
# # # EJ = logWA.easyJetWA
# # # EJ.dropna(inplace=True)
# #
# # RAlogDecompWA = seasonal_decompose(RA)
# # AFlogDecompWA = seasonal_decompose(AF)
# #
# # RAtrend = RAlogDecompWA.trend
# # RAseasonal = RAlogDecompWA.seasonal
# # RAresidual = RAlogDecompWA.resid
# # print(AFlogDecompWA.trend)
# # # AFtrend = AFlogDecompWA.trend
# # # AFseasonal = AFlogDecompWA.seasonal
# # # AFresidual = AFlogDecompWA.resid
# #
# # plt.subplot(411)
# # plt.plot(RA, label='Original')
# # # plt.plot(AF, label='Original')
# # plt.legend(loc='best')
# # plt.subplot(412)
# # plt.plot(RAtrend, label='Trend')
# # # plt.plot(AFtrend, label='Trend')
# # plt.legend(loc='best')
# # plt.subplot(413)
# # plt.plot(RAseasonal,label='Seasonality')
# # # plt.plot(AFseasonal,label='Seasonality')
# # plt.legend(loc='best')
# # plt.subplot(414)
# # plt.plot(RAresidual, label='Residuals')
# # # plt.plot(AFresidual, label='Residuals')
# # plt.legend(loc='best')
# # plt.tight_layout()

######################

RA = prices.RyanairWA
RA.dropna(inplace=True)
RA.index = pd.DatetimeIndex(RA.index.values, freq=RA.index.inferred_freq)

# Forecasting (using SARIMA)
#
# # Ryanair difference between log and log.shift
# # RADLS = RA - RA.shift()
# # RADLS.dropna(inplace=True)
# #
# # lag_acf = acf(RADLS, nlags=20)
# # lag_pacf = pacf(RADLS, nlags=20, method='ols')
# #
# # plt.subplot(121)
# # plt.plot(lag_acf)
# # plt.axhline(y=0,linestyle='--',color='gray')
# # plt.axhline(y=-1.96/np.sqrt(len(RADLS)),linestyle='--',color='gray')
# # plt.axhline(y=1.96/np.sqrt(len(RADLS)),linestyle='--',color='gray')
# # plt.title('Autocorrelation Function')
# #
# # plt.subplot(122)
# # plt.plot(lag_pacf)
# # plt.axhline(y=0,linestyle='--',color='gray')
# # plt.axhline(y=-1.96/np.sqrt(len(RADLS)),linestyle='--',color='gray')
# # plt.axhline(y=1.96/np.sqrt(len(RADLS)),linestyle='--',color='gray')
# # plt.title('Partial Autocorrelation Function')
# # plt.tight_layout()
# #
# # model = ARIMA(RA, order=(0, 1, 1))
# # results_MA = model.fit(disp=-1)
# # plt.plot(RA)
# # plt.plot(results_MA.fittedvalues, color='red')
# # plt.title('RSS: %.4f'% sum((results_MA.fittedvalues-RA)**2))
#
#
# estimating which values are the best for SARIMAX(lowest AIC)
# p = d = q = range(0, 2)
# pdq = list(itertools.product(p, d, q))
# seasonal_pdq = [(x[0], x[1], x[2], 12) for x in list(itertools.product(p, d, q))]
#
#
# for param in pdq:
#     for param_seasonal in seasonal_pdq:
#         try:
#             mod = sm.tsa.statespace.SARIMAX(RA,
#                                             order=param,
#                                             seasonal_order=param_seasonal,
#                                             enforce_stationarity=False,
#                                             enforce_invertibility=False)
#
#             results = mod.fit()
#
#             print('ARIMA{}x{}12 - AIC:{}'.format(param, param_seasonal, results.aic))
#         except:
#             continue

model = sm.tsa.statespace.SARIMAX(RA, order=(0, 1, 1), seasonal_order=(0, 1, 1, 12), enforce_stationarity=False, enforce_invertibility=False)
results_SARIMA = model.fit(disp=-1)

# # # infos about validity of the graph
# # print(results_SARIMA.summary().tables[1])
# # results_SARIMA.plot_diagnostics(figsize=(16, 8))
# plt.plot(RA)
# plt.plot(results_SARIMA.fittedvalues, color='red')
# plt.title('RSS: %.4f'% sum((results_SARIMA.fittedvalues-RA)**2))
#
# pred = results_SARIMA.get_prediction(start=pd.to_datetime('2017-01-01'), dynamic=False)
# pred_ci = pred.conf_int()
#
# ax = RA['2014':].plot(label='observed')
# pred.predicted_mean.plot(ax=ax, label='One-step ahead Forecast', alpha=.7, figsize=(14, 7))
#
# ax.fill_between(pred_ci.index, pred_ci.iloc[:, 0], pred_ci.iloc[:, 1], color='k', alpha=.2)
#
# ax.set_xlabel('Date')
# ax.set_ylabel('Fare price')
# plt.legend()

# # mean and square root of error
# y_forecasted = pred.predicted_mean
# y_truth = RA['2017-01-01':]
# MeanSquaredError = ((y_forecasted - y_truth) ** 2).mean()
# RootMeanSquaredError = np.sqrt(MeanSquaredError)

pred_uc = results_SARIMA.get_forecast(steps=100)
pred_ci = pred_uc.conf_int()

ax = RA.plot(label='observed', figsize=(14, 7))
pred_uc.predicted_mean.plot(ax=ax, label='Forecast')
ax.fill_between(pred_ci.index, pred_ci.iloc[:, 0], pred_ci.iloc[:, 1], color='k', alpha=.25)

ax.set_xlabel('Date')
ax.set_ylabel('Fare price')
plt.legend()
plt.show()
