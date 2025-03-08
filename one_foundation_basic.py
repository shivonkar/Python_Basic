#%% String data type
#In Python, Strings are immutable.

str1 = "Hello!"
print("str1[0]:",str1[0]) # Single char
print ("str1[1:3]:",str1[1:3])

#String Operators
str2 = "Sir"
print(str1 + " " +str2)
print('%s%s%s' %(str1,  " ", str2)) # Preferable. %d for int
# format function does away with %s %d

# *	Repeats
print(str1*4)

x = "Hello World!"
print(x[:6])
print(x[0:6] + "Hello")

#String replace
my_string = 'I am learning basic Python'
my_string.replace('basic', 'abc')

#upper and lower case
my_string.upper()
my_string.lower()
my_string.capitalize()

# "join" string. A more flexible ways
",".join(["Python", "learn"])

# Reverse
''.join(reversed(my_string))

#Split
my_string.split(' ')

#Tuple: It is just like a list of a sequence of immutable python objects.
#The difference between list and tuple is that list are declared in square brackets and
#can be changed while tuple is declared in parentheses and cannot be changed. However,
#you can take portions of existing tuples to make new tuples.

my_tuple = ('I', 'am', 'learning', 'basic', 'Python')

#Packing and Unpacking
my_tuple = ('I', 'am', 'learning', 'basic', 'Python') # packing
(_,_, activity, _, tech) = my_tuple #unpacking

activity
tech

#tuples are hashable: Hashing is a concept in computer science which is used to create
# high performance, pseudo random access data structures where large amount of data is
#to be stored and accessed quickly
#https://stackoverflow.com/questions/14535730/what-do-you-mean-by-hashable-in-python

#Slicing of Tuple
print(my_tuple[2:4])

# TBD Built-in functions with Tuple
#To perform different task, tuple allows you to use many built-in functions like all(),
#any(), enumerate(), max(), min(), sorted(), len(), tuple(), etc.

#Advantages of tuple over list
# 1. Iterating through tuple is faster than with list, since tuples are immutable.
# 2. Tuples that consist of immutable elements can be used as key for dictionary, which
#    is not possible with list
# 3. If you have data that is immutable, implementing it as tuple will guarantee that it
#    remains write-protected

#Dictionary: It is used to map or associate things like Keys and Values
# Keys will be a single element and Values can be a list or list within a list, numbers, etc.

my_dict = {'who': 'I','activity':'learning','what':'Python','count':1}
print(my_dict['activity'])

#Updating Dictionary
my_dict.update({"hours":2})
print(my_dict)

# Delete
del my_dict['hours']
print(my_dict)

#Dictionary items() returns a list of tuple pairs (Keys, Value) in the dictionary.
print("Items are: %s" % list(my_dict.items()))

# Keys
for key in list(my_dict.keys()):
    print('key: ' + str(key) + ", Value: " + str(my_dict[key]))

# Values
for value in list(my_dict.values()):
    print("Value: " + str(value))

# Sorting
my_keys = list(my_dict.keys())
my_keys.sort()
for k in my_keys:
      print(":".join((k,str(my_dict[k]))))

#len() Method
print("Length : %d" %len(my_dict))

#Str() method make a dictionary into a printable string format.
print("printable string:%s" % str(my_dict))

#%% Python Operators: Arithmetic, Logical, Comparison, Assignment, Bitwise & Precedence
my_num = 0
my_num = my_num + 1
my_num
my_num += 1
my_num

a = True
b = False
print(('a and b is',a and b))
print(('a or b is',a or b))
print(('not a is',not a))

#Membership Operators
x = 4
y = 8
list = [1, 2, 3, 4, 5 ];
if ( x in list ):
   print("Line 1 - x is available in the given list")
else:
   print("Line 1 - x is not available in the given list")
if ( y not in list ):
   print("Line 2 - y is not available in the given list")
else:
   print("Line 2 - y is available in the given list")

#%% Python Functions : Call, Indentation, Arguments & Return Values

#What is a Function in Python?

#define a function
def my_func(some_input):
   print("I am inside Python function")
   some_output = some_input + 1
   return(some_output)

my_func(5)

#%% Python IF, ELSE, ELIF, Nested IF & Switch Case Statement
x = 2; y = 5; st = ''
if(x < y):
    st= "x is less than y"
elif (x == y):
    st= "x is same as y"
else:
    st= "x is greater than y"

print(st)

#%% Regex: re.match(), re.search(), re.findall() with Example

#What is Regular Expression: A regular expression in a programming language is a
#special text string used for describing a search pattern. It is extremely useful for
#extracting information from text such as code, files, log, spreadsheets or even documents.

import re
# See the slide for few conditions

#Example of w+ and ^ Expression
#"^": This expression matches the start of a string
#"w+": This expression matches the alphanumeric character in the string

my_str = "Hello, I am learning basic Python"
re.findall(r"^\w+",my_str) # First word
re.findall(r"^\w",my_str) # First letter

#Example of \s expression in re.split function
#"s": This expression is used for creating a space in the string
re.split(r'\s',my_str) # at each space
re.split(r's',my_str) # where 's' is found

# Search pattern
re.search('learn', my_str)

#CW: Explore: re.match(),re.search(), re.findall()

#%% Python DateTime, TimeDelta, Strftime(Format) with Examples

#Date and datetime are an object in Python

#The datetime classes in Python are categorized into main 5 classes.
#date – Manipulate just date ( Month, day, year)
#time – Time independent of the day (Hour, minute, second, microsecond)
#datetime – Combination of time and date (Month, day, year, hour, second, microsecond)
#timedelta— A duration of time used for manipulating dates
#tzinfo— An abstract class for dealing with time zones

import datetime

now = datetime.datetime.now()
now

# Extarct attributes
now.day # day of the month
now.weekday() #  WeekDay Number - Monday as 0
now.year

# Now generalises using strftime
now.strftime('%d')
now.strftime('%w') # 0 is Sunday
now.strftime('%Y')

#%a	Weekday, short version	Wed
#%A	Weekday, full version	 Wednesday
#%w	Weekday as a number 0-6, 0 is Sunday
#%d	Day of month 01-31	31
#%b	Month name, short version	Dec
#%B	Month name, full version	December
#%m	Month as a number 01-12	12
#%y	Year, short version, without century	18
#%Y	Year, full version	2018
#%H	Hour 00-23	17
#%I	Hour 00-12	05
#%p	AM/PM	PM
#%M	Minute 00-59	41
#%S	Second 00-59	08
#%f	Microsecond 000000-999999	548513
#%z	UTC offset	+0100
#%Z	Timezone	CST
#%j	Day number of year 001-366	365
#%U	Week number of year, Sunday as the first day of week, 00-53	52
#%W	Week number of year, Monday as the first day of week, 00-53	52

# Using custome date
my_date = datetime.datetime(2018, 12, 2) # 02-Dec-2018
my_date.strftime('%d')
my_date.strftime('%w') # 0 is Sunday
my_date.strftime('%Y')

# Additions
my_date + datetime.timedelta(days=10) # 10 days
my_date + datetime.timedelta(hours=20)
my_date + datetime.timedelta(days=1, hours=8, minutes=10)

# CW: Practie with various date and durations

#%% File Handling: Create, Open, Append, Read, Write

#%%  Panda data frame explorations
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
train.head()
train.columns = map(str.upper, train.columns)
train.head()

#Infer few Constants from above
catColumns = ['ORIGIN']; strResponse = 'MPG'

# First view
train.dtypes
train.shape
train.index # get row index # train.index[0] # 1st row label # train.index.tolist() # get as a list
train.info()

# Change data types
train[catColumns] = train[catColumns].apply(lambda x: x.astype('category'))

# View summary
print(train.describe(include = 'all'))

# Few basic statistics
cr = train.corr() # pairwise correlation cols
cr
train.kurt() # kurtosis over cols (def)

# Data Extraction. Note: label slices are inclusive, integer slices exclusive
train.loc[0,'MPG'] # by labels
train.iloc[0,0]
train.iloc[0, :] # 0th row and all column
train.iloc[:, 0] # 0th column and all row
train.at[0,'MPG']
#train.ix[0,'MPG'] #Depricated. mixed label and integer position indexing

#Column level
train.MPG.head()
train['MPG'].head()

# Few more operations at data frame level
train.count()
train.min()

# Few more operations at column level
train['MPG'].idxmin() # get the index number where minimum is present
train['MPG'].where(train['MPG']>15)
train['MPG'].where(train['MPG']>15,other=0)
train[1:3] # 2 rows excluding row number 3. label slices are inclusive, integer slices exclusive.

# Creation of new column
train['temp_col'] = train.MPG / train.CYLINDERS
train.head()
train = train.drop('temp_col', axis=1)

train['temp_col'] = train['MPG'] / train['CYLINDERS']
train.drop('temp_col', axis=1, inplace=True)

# to get array of index where criteria mets
a = np.where(train['MPG'] > 15) # a is tuple(immutable, sequences like lists, parentheses lists use square brackets)
type(a), type(a[0])
len(a[0])

# Get index as one column so that it will help in merge
train = train.reset_index()
train.head()
train.drop('index', axis=1, inplace=True)

# strings operation
train['ORIGIN'] = train['ORIGIN'].str.lower() # upper, contains, startswith, endswith, replace('old', 'new'), extract('(pattern)')
train.head()

# Save data frame
train.to_csv('t.csv', index=False)

#%% Merge and concate various ways
df_hr = pd.DataFrame({'NAME':['A','B'], 'AGE': [35, 46]})
df_sal = pd.DataFrame({'NAME':['A','B'], 'SALARY': [1000, 2000]})

# Merge column wise
pd.merge(df_hr, df_sal, on='NAME')

#Simple concatenation is often the best
df_hr_2 = pd.DataFrame({'NAME':['C','D'], 'AGE': [25, 40]})
pd.concat([df_hr, df_hr_2],axis=0) #top/bottom

df_proj = pd.DataFrame({'PROJ':['R','Python']})
pd.concat([df_hr,df_proj],axis=1)#left/right

#%% Various Tables
train.head(2)

train_summary = train.groupby('ORIGIN').size()
train_summary

train_summary = train.groupby('ORIGIN')['MPG'].agg(['mean','count'])
train_summary

train_summary = train.groupby('ORIGIN')['MPG'].agg(['sum','mean','count'])
train_summary
#%% Basic plots
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix

# Some standard settings
plt.rcParams['figure.figsize'] = (13, 9) #(16.0, 12.0)
plt.style.use('ggplot')

numericColumn = 'HORSEPOWER'

# Scatter
fig, ax = plt.subplots(figsize=(17, 9)) # set size
ax.margins(0.05) # Optional, just adds 5% padding to the autoscaling
train.plot.scatter(x=numericColumn, y=strResponse, ax = ax)
plt.ylabel(strResponse, size=10)
plt.xlabel(numericColumn, size=10)
plt.title(numericColumn + " vs " + strResponse + " (Response variable)", size=10)
plt.tight_layout()  # To avoid overlap of subtitles
plt.show()

# Histogram
fig, ax = plt.subplots(figsize=(17, 9)) # set size
ax.margins(0.05) # Optional, just adds 5% padding to the autoscaling
train[numericColumn].plot.hist(bins=10, color='blue')  # alpha=0.5
plt.ylabel('Count', size=10)
plt.xlabel(numericColumn, size=10)
plt.title("Distribution of " + numericColumn, size=10)
plt.tight_layout()  # To avoid overlap of subtitles
plt.show()

# Box Plot
fig, ax = plt.subplots(figsize=(17, 9)) # set size
ax.margins(0.05) # Optional, just adds 5% padding to the autoscaling
bp = train[numericColumn].plot.box(sym='r+', showfliers=True, return_type='dict')
plt.setp(bp['fliers'], color='Tomato', marker='*')
plt.ylabel('Count', size=10)
plt.title("Distribution of " + numericColumn, size=10)
plt.tight_layout()  # To avoid overlap of subtitles
plt.show()

#Class work: Box Plot for 'HORSEPOWER' and 'WEIGHT' together

#Class work: For all features in one view. Run it and write the explanation
train.hist()
plt.show()

#Class work: For all features in one view. Run it and write the explanation
scatter_matrix(train)
plt.show()

# Save images in pdf and png
from matplotlib.backends.backend_pdf import PdfPages

pdf = PdfPages('abc.pdf')
train[numericColumn].hist()
pdf.savefig(bbox_inches='tight'); pdf.close()

# In png
train[numericColumn].hist()
plt.savefig("abc.png")

#%% Class work: Open one_foundation_numpy.py and Practice
