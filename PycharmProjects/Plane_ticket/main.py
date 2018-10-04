import datetime, xlrd
#import panda as pd
import numpy as np
import matplotlib.pyplot as plt

file_location = "C:\\Users\\Tanguy\\Documents\\ABP private\\ryanair_fare_data.xlsx"
print(file_location)
workbook = xlrd.open_workbook(file_location)
osheet = workbook.sheet_by_index(0)
nsheet = osheet
def xlToDate(xldate):
    temp = datetime.datetime(1900, 1, 1)
    delta = datetime.timedelta(days=xldate)
    return temp + delta

# def dateToXl(xldate):
#     temp = dt.datetime.strptime('18990101', '%Y%m%d')
#     delta = date1 - temp
#     total_seconds = delta.days * 86400 + delta.seconds
#     return total_seconds

def toDate(a, b, c):
    return datetime.datetime(a, b, c)

def scatterplot(x_data, y_data, x_label="", y_label="", title="", color = "r", yscale_log=False):

    # Create the plot object
    _, ax = plt.subplots()

    # Plot the data, set the size (s), color and transparency (alpha)
    # of the points
    ax.scatter(x_data, y_data, s = 10, color = color, alpha = 0.75)

    if yscale_log == True:
        ax.set_yscale('log')

    # Label the axes and provide a title
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)

def lineplot(x_data, y_data, x_label="", y_label="", title=""):
    # Create the plot object
    _, ax = plt.subplots()

    # Plot the best fit line, set the linewidth (lw), color and
    # transparency (alpha) of the line
    ax.plot(x_data, y_data, lw = 2, color = '#539caf', alpha = 1)

    # Label the axes and provide a title
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)

cell = nsheet.cell(1,0).value

print(toDate(1900, 1, 1))

print(xlToDate(cell))

if toDate(1900,1,1) > xlToDate(cell):
    print(1)
else:
    print(0)
column1 = []
for i in range(0, nsheet.row_len(0)):
    print(nsheet.col_values(i))

   # column1.append(nsheet.cell(i, 2).value)

print(nsheet.cell(1,2))
time = nsheet.r_values(0)
time.pop(0)
time.pop(71)
time.pop(71)
for i in range(0, len(time)):
    time[i] = xlToDate(time[i])
# for i in range(1, nsheet.col_len(0)):
#     time.append(xlToDate(nsheet.cell(i, 0)))
value = nsheet.col_values(7)
value.pop(0)
value.pop(71)
value.pop(71)
for i in range(0, len(time)):
    if value[i] is None:
        value.pop(i)
        time.pop(i)
        i = i-1
print(time)
print(value)
lineplot(time, value, 'time', 'value')
plt.show()
print(3)
print(4)

