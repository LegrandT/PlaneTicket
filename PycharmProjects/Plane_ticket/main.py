import datetime, xlrd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas import ExcelWriter
from pandas import ExcelFile

file_location = "C:\\Users\\Tanguy\\Documents\\ABP private\\ryanair_fare_data.xlsx"
odf = pd.read_excel(file_location)
ndf = odf


# def lineplot(x_data, y_data, x_label="", y_label="", title=""):
#     # Create the plot object
#     _, ax = plt.subplots()
#
#     # Plot the best fit line, set the linewidth (lw), color and
#     # transparency (alpha) of the line
#     ax.plot(x_data, y_data, lw = 2, color = '#539caf', alpha = 1)
#
#     # Label the axes and provide a title
#     ax.set_title(title)
#     ax.set_xlabel(x_label)
#     ax.set_ylabel(y_label)

print(ndf)

ndf.plot()


plt.show()