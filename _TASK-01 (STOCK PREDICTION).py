#!/usr/bin/env python
# coding: utf-8

# # task 1 stock market prediction

# In[1]:


import numpy as np 
import pandas as pd
get_ipython().system('pip install yfinance')
get_ipython().system('pip install textblob')
get_ipython().system('pip install xgboost')
import yfinance as yf
from matplotlib import pyplot as plt
from  sklearn.metrics import mean_squared_error,mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
from textblob import TextBlob
import xgboost as xgb


# In[65]:


stock_data = yf.download("^BSESN",start="2012-11-11",end="2022-11-11")
news_data = pd.read_csv("india-news-headlines.csv")


# In[66]:


stock_data.to_csv("stock_data.csv")


# In[67]:


stock_data


# In[39]:


news_data



# In[40]:


news_data["publish_date"]= pd.to_datetime(news_data['publish_date'],format="%Y%m%d")


# In[41]:


news_data


# In[68]:


stock_data.reset_index(inplace=True)
stock_data


# In[43]:


stock_data.info()


# In[44]:


news_data.info()


# Sentiment Analysis From News Headlines

# In[45]:


news_data["sentiment"]= news_data["headline_text"].apply(lambda x: TextBlob(x).sentiment.polarity)


# In[46]:


average_sentiment = news_data.groupby("publish_date")["sentiment"].mean().reset_index()


# In[47]:


average_sentiment


# In[48]:


average_sentiment.rename(columns = {"publish_date":"date"},inplace=True)
average_sentiment


# Merging Stock And News Dataset

# In[69]:


stock_data.rename(columns = {"Date":"date"},inplace=True)


# In[70]:


merged_data=pd.merge(stock_data,average_sentiment,how="left",on="date")


# In[71]:


merged_data


# Creating New Features(Feature Engg.)
# 

# In[72]:


merged_data["M10"]=merged_data["Close"].rolling(window=10).mean()
merged_data["M20"]=merged_data["Close"].rolling(window=20).mean()
merged_data["M30"]=merged_data["Close"].rolling(window=30).mean()
merged_data["daily_return"]=merged_data["Close"].pct_change()
merged_data["volatility"]=merged_data["daily_return"].rolling(window=10).std()


# In[73]:


merged_data


# In[74]:


merged_data.dropna(inplace=True)
merged_data


# In[75]:


merged_data.reset_index(inplace=True)
merged_data.drop(columns='index',inplace=True)
merged_data


# In[76]:


final_data=pd.DataFrame(merged_data)
final_data


# visualization of the Features
# 

# In[77]:


plt.figure(figsize=(12,6))
Date = final_data["date"].values
Sentiment = final_data["sentiment"].values

plt.plot(Date, Sentiment)
plt.xlabel("Date")
plt.ylabel("Sentiment")
plt.title("Sentiment Over Time")
plt.xticks(rotation=45)
plt.show()


# In[78]:


#plotting the moving Averages
plt.figure(figsize=(12, 6))
close_value = final_data["Close"].values
MA10 = final_data["M10"].values
MA20= final_data["M20"].values
MA30 = final_data["M30"].values
plt.plot(Date,close_value, label='Closing Price')
plt.plot(Date, MA10, label='MA10')
plt.plot(Date, MA20, label='MA20')
plt.plot(Date, MA30, label='MA30')
plt.xlabel('Date')
plt.ylabel('Price')
plt.title('Moving Averages')
plt.legend()
plt.xticks(rotation=45)
plt.show()


# In[79]:


plt.figure(figsize=(12,6))
volatility=final_data["volatility"].values
plt.plot(Date,volatility)
plt.xlabel("Date")
plt.ylabel("Volatility")

plt.title("Volatility Over Time")
plt.xticks(rotation=45)
plt.show()


# In[80]:


# creating training and test data
train_data=final_data.iloc[:len(final_data)-100]
test_data=final_data.iloc[len(final_data)-100:]


# Feature scaling to normalise the features

# In[81]:


scaler=MinMaxScaler()
numerical_features=["Open","High","Low","Volume","sentiment","M10","M20","M30","daily_return","volatility"]
train_data.loc[:,numerical_features]=scaler.fit_transform(train_data[numerical_features])
test_data.loc[:,numerical_features]=scaler.transform(test_data[numerical_features])


#  using EXTREME GRADIENT BOOSTING REGRESSOR to predict the closing stock prices
# 
# 

# In[84]:


xgb_regressor = xgb.XGBRegressor(random_state=0)
xgb_regressor.fit(train_data.loc[:,numerical_features],train_data["Close"])
test_data.loc[:,"predicted_close_price"]=xgb_regressor.predict(test_data.loc[:,numerical_features])


# In[85]:


test_data


# Model Evaluation:

# In[86]:


mse= mean_squared_error(test_data["Close"],test_data["predicted_close_price"])
mae= mean_absolute_error(test_data["Close"],test_data["predicted_close_price"])
print("mean squared error:",np.sqrt(mse))
print("mean absolute error:",np.sqrt(mae))


# In[90]:


plt.figure(figsize=(12,6))
Date=test_data["date"].values
Close=test_data["Close"].values
pred=test_data["predicted_close_price"].values
plt.plot(Date,Close,label="actual closing price")
plt.plot(Date,pred,label="predicted closing price")
plt.xlabel("Date")
plt.ylabel("Closing Price")
plt.title("Actual vs Predicted Closing Price")
plt.legend()
plt.xticks(rotation=45)
plt.show()

