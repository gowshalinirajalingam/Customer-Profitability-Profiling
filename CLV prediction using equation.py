#CLTV Implementation in Python(Using Formula)

#Import modules
import pandas as pd # for dataframes
import matplotlib.pyplot as plt # for plotting graphs
import seaborn as sns # for plotting graphs
import datetime as dt
import numpy as np
from scipy.stats import norm, skew #for some statistics
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LinearRegression


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



rfm= data_filtered.groupby('CID').agg({'TransDate': lambda date: (PRESENT - date.max()).days,
                                        'TransNo': lambda num: len(num),
                                        'Revenue': lambda Reve: Reve.sum()})

rfm.columns
rfm.dtypes


    # Change the name of columns
rfm.columns=['recency','frequency','monetary']



#CLTV score
#Formula
# CLTV = ((Average Order Value x Purchase Frequency)/Churn Rate) x Profit margin.
# Customer Value = Average Order Value * Purchase Frequency


# Creating KPIs
# 1. Average Transaction Value
rfm['avg_order_value']=rfm['monetary']/rfm['frequency']

rfm.head()

# 2. Calculate Purchase frequency
purchase_frequency=sum(rfm['frequency'])/rfm.shape[0]

#  3. Calculate repeat rate and frequency rate
repeat_rate=rfm[rfm.frequency > 1].shape[0]/rfm.shape[0]

# 4.churn rate
churn_rate=1-repeat_rate

# purchase_frequency= 4.555555555555555
# repeat_rate=0.7777777777777778
# churn_rate =0.2222222222222222


# 5. profit margin
# It represents how much percentage of profit earned.
rfm['profit_margin']=rfm['monetary']*0.05

# 6. Calculate Customer Life Time Value
# Customer Value
rfm['CLV']=(rfm['avg_order_value']*purchase_frequency)/churn_rate

#Customer Lifetime Value
rfm['cust_lifetime_value']=rfm['CLV']*rfm['profit_margin']

rfm.head()



# Selecting Feature


# Here we select Customer demographic and transactional data to train
#TODO :for X add customer demographic data too.
X=rfm[['recency','frequency','monetary']]
y=rfm['cust_lifetime_value']

# Target variable
# Log-transformation of the target variable
#We use the numpy fuction log1p which  applies log(1+x) to all elements of the column
y = np.log1p(y)

#Check the new distribution
sns.distplot(y , fit=norm);

# Get the fitted parameters used by the function
(mu, sigma) = norm.fit(train_dataset['SalePrice'])
print( '\n mu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))

#Now plot the distribution
plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],
            loc='best')
plt.ylabel('Frequency')
plt.title('SalePrice distribution')

# #Get also the QQ-plot
# fig = plt.figure()
# res = stats.probplot(train_dataset['SalePrice'], plot=plt)
# plt.show()

def regression_Modeling(X,y):
    
   
    # split training set and test set


    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)


    # instantiate
    linreg = LinearRegression()

    # fit the model to the training data (learn the coefficients)
    linreg.fit(X_train, y_train)

    # make predictions on the testing set
    y_pred = linreg.predict(X_test)

    # print the intercept and coefficients
    print(linreg.intercept_)
    print(linreg.coef_)

    # R squared
    # Value of R-squared lies between 0 and 1.
    # Higher value or R-squared is considered better because it indicates the larger variance explained by the model.
    from sklearn import metrics

    # compute the R Square for model
    print("R-Square:", metrics.r2_score(y_test, y_pred))

    # This model has a higher R-squared (0.96). This model provides a better fit to the data.

    #
    # Model Evaluation
    # For regression problems following evaluation metrics used (Ritchie Ng):
    #
    # Mean Absolute Error (MAE) is the mean of the absolute value of the errors.
    # Mean Squared Error (MSE) is the mean of the squared errors.
    # Root Mean Squared Error (RMSE) is the square root of the mean of the squared errors.

    # calculate MAE using scikit-learn
    print("MAE:", metrics.mean_absolute_error(y_test, y_pred))

    # calculate mean squared error
    print("MSE", metrics.mean_squared_error(y_test, y_pred))
    # compute the RMSE of our predictions
    print("RMSE:", np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
    # RMSE is more popular than MSE and MAE because RMSE is interpretable with y because of the same units.

regression_Modeling(X,y)