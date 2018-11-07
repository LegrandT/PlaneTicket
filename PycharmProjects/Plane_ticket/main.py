import datetime, xlrd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# import fbprophet
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose


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

def analysis(ts):
    meanWA = ts.rolling(12).mean()
    stdWA = ts.rolling(12).std()
    plt.plot(ts.RyanairWA, 'r', label='RyanairWA')
    plt.plot(ts.easyJetWA, 'b', label='easyJetWA')
    plt.plot(ts.AirFranceWA, 'g', label='AirFranceWA')
    plt.plot(stdWA.RyanairWA, 'r', label = 'RyanairWA')
    plt.plot(stdWA.easyJetWA, 'b', label = 'easyJetWA')
    plt.plot(stdWA.AirFranceWA, 'g', label = 'AirFranceWA')
    plt.plot(meanWA.RyanairWA, 'r', label = 'RyanairWA')
    plt.plot(meanWA.easyJetWA, 'b', label = 'easyJetWA')
    plt.plot(meanWA.AirFranceWA, 'g', label = 'AirFranceWA')
    plt.title('Fare WeightedAverage')
    plt.ylabel('Price (euro)')
    plt.xlabel('Year')
    plt.show()








logWA = np.log(prices)

rollMeanWA = logWA.rolling(12).mean()

diffLogMA = logWA - rollMeanWA

exwaWA = logWA.ewm(halflife=12).mean()
logexwaDiffWA = logWA - exwaWA

# differencing
diffLogShift = logWA-logWA.shift()

# decomposing
RA = logWA.RyanairWA
RA.dropna(inplace=True)



logDecompWA = seasonal_decompose(RA)

trend = logDecompWA.trend
seasonal = logDecompWA.seasonal
residual = logDecompWA.resid

plt.subplot(411)
plt.plot(RA, label='Original')
plt.legend(loc='best')
plt.subplot(412)
plt.plot(trend, label='Trend')
plt.legend(loc='best')
plt.subplot(413)
plt.plot(seasonal,label='Seasonality')
plt.legend(loc='best')
plt.subplot(414)
plt.plot(residual, label='Residuals')
plt.legend(loc='best')
plt.tight_layout()
plt.show()



#print(logexwaDiffWA)
#analysis(diffLogShift)



# plt.plot(logWA-logWA.shift())
# meanWA.dropna(inplace=True)
# stdWA.dropna(inplace=True)


# print(prices)
# print(meanWA)
# print(stdWA)
