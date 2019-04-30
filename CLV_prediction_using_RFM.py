#CLTV Implementation in Python(Using Formula)

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
data_filtered=data[["CID","TransNo","FirstTransDate","TransDate","Revenue"]]


#RFM Calculation
#STEP1: Calculate R,F,M for each customer
PRESENT = dt.datetime(2019,4,18)
data_filtered['FirstTransDate'] = pd.to_datetime(data_filtered['FirstTransDate'])
data_filtered['TransDate'] = pd.to_datetime(data_filtered['TransDate'])



rfm= data_filtered.groupby('CID',as_index=False).agg({'TransDate': lambda date: (PRESENT - date.max()).days,
                                        'TransNo': lambda num: len(num),
                                        'Revenue': lambda Reve: Reve.sum()})

rfm.columns
rfm.dtypes


    # Change the name of columns
rfm.columns=['recency','frequency','monetary']


#STEP2: Add segment bin values to RFM table using quartile.
# qcut() is Quantile-based discretization function. qcut bins the data based on sample quantiles.
# For example, 1000 values for 4 quantiles would produce a categorical object indicating quantile membership for each customer.

# Customers with the lowest recency, highest frequency and monetary amounts considered as top customers.


rfm['r_quartile'] = pd.qcut(rfm['recency'], 4, ['1','2','3','4'])
rfm['f_quartile'] = pd.qcut(rfm['frequency'].rank(method='first'), 4, ['4','3','2','1'])   #Rank is added due to ValueError: Bin edges must be unique: array([ 1,  2,  2,  4, 13]).You can drop duplicate edges by setting the 'duplicates' kwarg
rfm['m_quartile'] = pd.qcut(rfm['monetary'], 4, ['4','3','2','1'])

rfm.head()


rfm['RFM_Value'] = rfm.r_quartile.astype(str)+ rfm.f_quartile.astype(str) + rfm.m_quartile.astype(str)
rfm.head()
rfm.info()

#Label the RFM value
rfm['RFM_Label'] = rfm['RFM_Value'].apply(lambda x: 'Best Customer' if x=='111' else
                                                    'Most Loyal Customer' if x=='312' else
                                                    'Highest Paying customer' if x=='111' else
                                                    'Faithful customer' if x=='113' else
                                                    'Faithful customer' if x=='213' else
                                                    'Faithful customer' if x=='313' else
                                                    'Faithful customer' if x=='413' else

                                                    'Faithful customer' if x=='114' else
                                                    'Faithful customer' if x=='214' else
                                                    'Faithful customer' if x=='314' else
                                                     'Faithful customer' if x=='414' else

                                                     'New customer' if x=='141' else
                                                    'New customer' if x=='142' else
                                                     'New customer' if x=='143' else
                                                    'New customer' if x=='144' else

                                                    'Once Loyal,Now Gone' if x=='441' else
                                                    'Once Loyal,Now Gone' if x=='442' else
                                                     'Once Loyal,Now Gone' if x=='443' else
                                                    'Once Loyal,Now Gone' if x=='444' else None)


# Filter out Top/Best cusotmers
rfm[rfm['RFM_Value']=='111'].sort_values('monetary', ascending=False).head()      #No data

#RFM Score
rfm.dtypes
rfm['r_quartile'] = rfm['r_quartile'] .astype('int64')
rfm['f_quartile'] = rfm['f_quartile'].astype('int64')
rfm['m_quartile'] = rfm['m_quartile'].astype('int64')

# (4*4*4=64 ) This is maximum m*f*r value
rfm['RFM_Score'] = (64-(rfm['m_quartile']*rfm['f_quartile']*rfm['r_quartile']))/64*100
rfm.sort_values('RFM_Score', ascending=False).head()


# Here we select Customer demographic and transactional data to train
#TODO :for X add customer demographic data too.
X=rfm[['r_quartile','f_quartile','m_quartile']]
y=rfm[['RFM_Score']]


# Explotary Analysis
# plot a graph
ax=rfm.groupby('RFM_Score').count().plot(kind='bar', colormap='copper_r')
ax.set_xlabel("RFM Score")
ax.set_ylabel("Count of customers")

#sort
beforesortdf=pd.DataFrame(rfm.groupby('RFM_Score',as_index=False).agg('monetary').mean())
aftersortdf=beforesortdf.sort_values(['monetary'], ascending=[True])


# predictive model building
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
plt.title('RFM score distribution')

# #Get also the QQ-plot
# fig = plt.figure()
# res = stats.probplot(y, plot=plt)
# plt.show()


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

