# Here Customer Lifetime Value is a monetary value that represents the amount of revenue a customer will provide
# the business over the lifetime of the relationship.
# Here we are training our model using last 3 years data of existing customer.so we can predict the CLV(the total revenue) for
# next 3 years from customers account creation date for new customers

#Import modules
import pandas as pd # for dataframes
import matplotlib.pyplot as plt # for plotting graphs
import seaborn as sns # for plotting graphs
import datetime as dt
import numpy as np
from scipy.stats import norm, skew #for some statistics
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_squared_error


#Loading Dataset
data = pd.read_csv("/home/gawshalini/Documents/sampath customer profiling/test example python/Customer profitability profiling/Bank Customer Transaction Details.csv")
print(data.head())

#Info about data set
data.info()

#DataType setting
print(data.dtypes)

data["TransNo"]=data["TransNo"].astype("object")
data["CID"]=data["CID"].astype("object")
data["AccNo"]=data["AccNo"].astype("object")
data["AccCreatedDate"]=data["AccCreatedDate"].astype("datetime64")
data["FirstTransDate"]=data["FirstTransDate"].astype("datetime64")
data["TransDate"]=data["TransDate"].astype("datetime64")
data["TransAmt"]=data["TransAmt"].astype("float64")
data["Revenue"]=data["Revenue"].astype("float64")




data.describe()


#Filter required columns
data_filtered=data[["CID","TransDate","Revenue"]]

data_filtered.dtypes




# Extract month and year from TransactionDate.
data_filtered['month_yr'] = data_filtered['TransDate'].apply(lambda x: x.strftime('%b-%Y'))
data_filtered.head()



#Creating pivot table
# The pivot table takes the columns as input, and groups the entries into a two-dimensional table
# in such a way that provides a multidimensional summarization of the data.

Revenue=data_filtered.pivot_table(index=['CID'],columns=['month_yr'],values='Revenue',aggfunc='sum',fill_value=0).reset_index()



# Let's sum all the months Revenue and Create CLV column
Revenue['CLV']=Revenue.iloc[:,2:].sum(axis=1)
Revenue.head()


# Here we select last 3 years data
#TODO :for X add customer demographic data too.
X=Revenue[['Mar-2017','Mar-2018', 'Mar-2019']]
y=Revenue[['CLV']]


# Target variable
#The target variable is left skewed. As (linear) models love normally distributed data ,
#we need to transform this variable and make it more normally distributed

# Log-transformation of the target variable
#We use the numpy fuction log1p which  applies log(1+x) to all elements of the column
y = np.log1p(y)

#Check the new distribution
sns.distplot(y , fit=norm);

# Get the fitted parameters used by the function
(mu, sigma) = norm.fit(y)
print( '\n mu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))

#Now plot the distribution
plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],
            loc='best')
plt.ylabel('Frequency')
plt.title('Revenue distribution')
#
# #Get also the QQ-plot
# fig = plt.figure()
# res = stats.probplot(y, plot=plt)
# plt.show()




# Selecting Feature
# Here, you need to divide the given columns into two types of variables dependent(or target variable) and independent variable(or feature variables).
# Select latest 6 month as independent variable.
# X=Revenue[['Dec-2011','Nov-2011', 'Oct-2011','Sep-2011','Aug-2011','Jul-2011']

# Here we select last 3 years data
X=Revenue[['Mar-2019','Mar-2019', 'Mar-2019']]
y=Revenue[['CLV']]

#Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

# Revenue prediction for comming 3 years Regression model building
model = LinearRegression()
model.fit(X_train,y_train)
print("Coefficients: {} \n".format( model.coef_))
print("Intercept: {}".format(model.intercept_))


#Test on testing data
y_pred = model.predict(X_test)

print("RMSE: {}".format(np.sqrt(mean_squared_error(y_test, y_pred))))

