
import streamlit as st
import datetime as date
import yfinance as yf
from fbprophet import Prophet
from fbprophet.plot import plot_plotly
from plotly import graph_objects as go 
import matplotlib.pyplot as plt



START="2015-01-01"
TODAY= date.datetime.now()

st.title(" Cryptocurrencies price prediction")
crypto=("BTC-USD","ETH-USD","ADA-USD","XRP-USD","SOL-USD","MATIC-USD","LINK-USD","FTM-USD","LTC-USD")
selected_crypto=st.selectbox("Select dataset for prediction",crypto)
n_year=st.slider("Year of prediction:",1,5)
period=n_year*365
@st.cache
def load_data(tricker):
    data=yf.download(tricker,START,TODAY)
    data.reset_index(inplace=True)
    return data

data_load_state=st.text("Load Data ...")
data=load_data(selected_crypto)
data_load_state=st.text("Loading data ... done!")
st.subheader("Data")
st.write(data.tail(5))
st.subheader("Discover the Data set: ")
st.write("DataSet Column:")
st.write(data.columns)


#visualization

st.subheader("Visualize the data Now")
st.write("It is considered a good practice to visualize the data at hand. So let’s plot our time series data:")

def plot_raw_data():
    fig=go.Figure()
    fig.add_trace(go.Scatter(x=data["Date"],y=data['Open'],name="Crypto Open"))
    fig.add_trace(go.Scatter(x=data["Date"],y=data['Close'],name="Crypto Close"))
   
    fig.layout.update(title="Time Series Data",xaxis_rangeslider_visible=True)
    st.plotly_chart(fig,use_container_width=True)
   

plot_raw_data()




#forecasting
df_train=data[['Date','Close']]
df_train=df_train.rename(columns={"Date":"ds","Close":"y"})

st.write("Prophet also imposes the strict condition that the input columns must be named as ds (Date) and y (Close).So, we must rename the columns in our dataframe.")
st.write(df_train.tail())

# set the uncertainty interval to 95% (the Prophet default is 80%)

m=Prophet(interval_width=0.95)
#Now that our Prophet model has been initialized, we can call its fit method with our DataFrame as input.
m.fit(df_train)

#In order to obtain forecasts of our time series, we must provide Prophet with a new DataFrame containing a ds column that holds the dates for which we want predictions.

#Conveniently, we do not have to concern ourselves with manually creating this DataFrame, as Prophet provides the make_future_dataframe helper function.
st.write("In order to obtain forecasts of our time series, we must provide Prophet with a new DataFrame containing a ds column that holds the dates for which we want predictions.")
st.write("So here we will use the make_future_dataframe helper function")


future=m.make_future_dataframe(periods=period)

# In the code snippet above, we instructed Prophet to generate 365 day*number of the year  in the future.
# When working with Prophet, it is important to consider the frequency of our time series.

# Because we are working with monthly data, we clearly specified the desired frequency of the timestamps (in this case, MS is the start of the month).

# Therefore, the make_future_dataframe generated 12*number of year monthly timestamps for us.

# In other words, we are looking to predict future values of our time series 5 years into the future

forecast=m.predict(future)   
st.subheader('forecast data')
st.write(forecast.tail())
# Prophet returns a large DataFrame with many interesting columns, but we subset our output to the columns most relevant to forecasting. These are:

# ds: the datestamp of the forecasted value
# yhat: the forecasted value of our metric (in Statistics, yhat is a notation traditionally used to represent the predicted values of a value y)
# yhat_lower: the lower bound of our forecasts
# yhat_upper: the upper bound of our forecasts

forecast_fig=plot_plotly(m,forecast)
st.subheader("Prophet plots the observed values of our time series the black dots)")
st.plotly_chart(forecast_fig,use_container_width=True,uncertainty=True)

st.write("the forecasted values (blue line) and the uncertainty intervals of our forecasts (the blue shaded regions).One other particularly strong feature of Prophet is its ability to return the components of our forecasts.")

st.write("This can help reveal how daily, weekly and yearly patterns of the time series contribute to the overall forecasted values.")

st.subheader("Plotting the forecasted components")
fig2=m.plot_components(forecast)

st.write(fig2)


st.write("The above plot provides interesting insights.")
st.write("The first plot shows that the monthly volume of coin price has been linearly increasing over time.")
st.write("The second plot highlights the fact that the weekly price of the coin  peaks towards the end of the week and on Saturday. and go down on Tuesday and Thursday")
st.write("The third plot shows that the Highest price occurs between  January and May")

st.subheader("Adding ChangePoints to Prophet")

st.write("Changepoints are the datetime points where the time series have abrupt changes in the trajectory.")
st.write("By default, Prophet adds 25 changepoints to the initial 80% of the data-set.")
st.write("Let’s plot the vertical lines where the potential changepoints occurred")

from fbprophet.plot import add_changepoints_to_plot
fig = m.plot(forecast)

a = add_changepoints_to_plot(fig.gca(), m, forecast)
st.pyplot(fig)

st.write("We can view the dates where the chagepoints occurred.")
st.write(m.changepoints)
st.subheader("Adjusting Trend ")
st.write("Prophet allows us to adjust the trend in case there is an overfit or underfit.")

st.write("changepoint_prior_scale helps adjust the strength of the trend.")

# Default value for changepoint_prior_scale is 0.05.

# Decrease the value to make the trend less flexible.

# Increase the value of changepoint_prior_scale to make the trend more flexible.

# Increasing the changepoint_prior_scale to 0.08 to make the trend flexible.

pro_change= Prophet(n_changepoints=20, yearly_seasonality=True, changepoint_prior_scale=0.08)
forecast = pro_change.fit(df_train).predict(future)
fig= pro_change.plot(forecast);
a = add_changepoints_to_plot(fig.gca(), pro_change, forecast)
st.pyplot(fig)


# 10. Conclusion 

# In this tutorial, we described how to use the Prophet library to perform time series forecasting in Python.

# We have been using out-of-the box parameters, but Prophet enables us to specify many more arguments.

# In particular, Prophet provides the functionality to bring your own knowledge about time series to the table.


# 11. References 

# The concepts and ideas in this notebook are tgaken from the following websites-
# https://facebook.github.io/prophet/
# https://facebook.github.io/prophet/docs/quick_start.html
# https://peerj.com/preprints/3190.pdf
# https://www.digitalocean.com/community/tutorials/a-guide-to-time-series-forecasting-with-prophet-in-python-3
