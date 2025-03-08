import pandas as pd
import os
import numpy as np
# Working directory
os.chdir("D:\Trainings\python_Basic")

#Set PANDAS to show all columns in DataFrame
pd.set_option('display.max_columns', None)
#Set PANDAS to show all rows in DataFrame
pd.set_option('display.max_rows', None)
pd.set_option('precision', 2)

# Read data
train = pd.read_csv("./data/mpg.csv")
train.columns = map(str.upper, train.columns)
train.head()

#Few Constants from above
catColumns = ['ORIGIN']; strResponse = 'MPG'

#%% Handle Missing Data
# Is there any missing data anywhere
train.isnull().sum()

# Impute some missing data for practice
df_with_missing_data = train.copy()
# https://pandas.pydata.org/pandas-docs/version/0.23.4/generated/pandas.Series.html
df_with_missing_data.loc[df_with_missing_data['ACCELERATE'].between(18.0,20.0),'ACCELERATE'] = None
df_with_missing_data.head(20) # see the NA
df_with_missing_data.isnull().sum()

# See the count rowwise
naRow = df_with_missing_data.apply(lambda x: x.isnull().sum(), axis=1)
naRow
sum(naRow)

# use of chain function
df_with_missing_data.apply(lambda x: x.isnull().sum(), axis=1).sum()

# CW: get missing count of each col and use do.call to get total sum

# Few important functions
#DataFrame.dropna([axis, how, thresh, …])	Remove missing values.
#DataFrame.fillna([value, method, axis, …])	Fill NA/NaN values using the specified method
#DataFrame.replace([to_replace, value, …])	Replace values given in to_replace with value.
#DataFrame.interpolate([method, axis, limit, …])	Interpolate values according to different methods.

# Get complete records only
df_with_missing_data.shape
df_with_missing_data.dropna(inplace = True)
df_with_missing_data.shape

del(df_with_missing_data)

#%% Encode Categorical features
#from sklearn import preprocessing

# See unique values of categorical data
train[catColumns[0]].unique()

# Create one hot coding
df = pd.get_dummies(train[catColumns])
df.head()

# delete old category column
train.drop('ORIGIN', axis=1, inplace = True)

# Combine
train = pd.concat([train,df],axis=1)
train.head()

del(df)

##################### Data sorting ##########################################
df = pd.DataFrame({'A': [0,1,3,4], 'B': [1,0.1,0.1,1], 'C' : [9,8,10,11]})
df

#sort ascending
df.sort_values("B", ascending=True, inplace=True)
df

#sort descending
df.sort_values("B", ascending=False, inplace=True)
df

#sort ascending and descending together
df.sort_values(['B','C'], ascending=[True,False], inplace=True)
df

##################### Mathematical Functions ##########################################
# Note: These are few of commonly used functions

# Numeric Functions
abs(-10)	# absolute value
np.sqrt(10)	# or math.
np.ceil(1.23)	# 2
np.floor(1.23)	# 1
np.trunc(1.23) # remove decimals similar to making int
round(1.23, 1) #

# Trigonometric
#cos(x), sin(x), tan(x),	 acos(x), cosh(x), acosh(x)

# Log
np.log(20)	#natural logarithm
np.log10(20)	#common logarithm
np.exp(2)	# e^2

# start = 5, stop = 20, step = 2
for r in range(5, 20, 2):
    print(r)

###################### Statistical & Probability Functions #########################
# plot standard normal curve
import matplotlib.pyplot as plt
from scipy.stats import norm
# https://docs.scipy.org/doc/scipy-0.16.1/reference/generated/scipy.stats.norm.html

x = np.arange(-3,3,0.001)
plt.plot(x, norm.pdf(x, 0, 1))
plt.ylabel("Density", size=10); plt.xlabel("Sigma", size=10)
plt.show()

# Nice graph from follwoing link
# https://stackoverflow.com/questions/46375553/standard-normal-distribution-in-python
def draw_z_score(z0, title, mu = 0, sigma = 1):
    x = np.arange(-3,3,0.001)
    cond = x<z0
    y = norm.pdf(x, mu, sigma)
    z = x[cond]
    plt.plot(x, y)
    plt.fill_between(z, 0, norm.pdf(z, mu, sigma))
    plt.title(title); plt.ylabel("Sigma", size=10); plt.xlabel("Density", size=10)
    plt.show()

# Call above

z0 = -1.96
draw_z_score(z0, 'z<-1.96')

# cumulative normal probability for q (area under the normal curve to the left of q)
norm.cdf(-1.96) # 0.025
norm.ppf(0.025) # Percent point function (inverse of cdf — percentiles)
# Same as above
norm.ppf(.975) # Normal quantile: value at the p percentile of normal distribution

m=50; sd=10
np.random.normal(m, sd, 100) # n random normal deviates with mean m and standard deviation sd.

#CW: Similarly for Uniform and poisson distributions

#%% Sampling
train.sample(100, replace = True)
train.sample(100, replace = False)

##################### Logging the various types of messages ##########################################
import os
os.chdir("D:\Trainings\python_Basic") # Working directory

import logging
import datetime

# create logger
logger = logging.getLogger('My_Logger')

# Before adding handlers to an instance of the logger, make sure not to add duplicate handlers
if len(logger.handlers) == 0:
    logger.setLevel(logging.INFO) # DEBUG

    # create formatter
    formatter = logging.Formatter('%(asctime)s %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

    # create console handler and set level to debug, add formatter to ch
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    # Similar to above for file self.g_folder_log,
    fh = logging.FileHandler(''.join(["./log/",'my_log', datetime.datetime.now().strftime("%b_%d_%Y"), ".log"]))
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)
    logger.addHandler(fh)
# end of if

# 'application' code
logger.info('info message')
logger.warning('warn message')
logger.error('error message')
logger.critical('critical message')

##################### Exceptional Handling in details ##########################################
import sys
try:
    # Some operations
    a = 10/0
except:
    print('%s%s' %("Unexpected error:", str(sys.exc_info()[0])))
    #raise or pass
