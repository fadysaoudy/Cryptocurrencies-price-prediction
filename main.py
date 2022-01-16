
from fbprophet.plot import add_changepoints_to_plot
import streamlit as st
import datetime as date
import yfinance as yf
from fbprophet import Prophet
from fbprophet.plot import plot_plotly
from plotly import graph_objects as go
import matplotlib.pyplot as plt


START = "2015-01-01"
TODAY = date.datetime.now()

st.title(" Cryptocurrencies price prediction")
crypto = ("BTC-USD", "ETH-USD", "ADA-USD", "XRP-USD", "SOL-USD",
          "MATIC-USD", "LINK-USD", "FTM-USD", "LTC-USD")
selected_crypto = st.selectbox("Select dataset for prediction", crypto)
n_year = st.slider("Year of prediction:", 1, 5)
period = n_year*365

# --------------------------------------------------------


@st.cache
def load_data(parg):
    data = yf.download(parg, START, TODAY)
    data.reset_index(inplace=True)
    return data


data_load_state = st.text("Load Data ...")
data = load_data(selected_crypto)
data_load_state = st.text("Loading data ... done!")

# --------------------------------------------------------

st.subheader("Data")
st.write(data.tail(5))

# --------------------------------------------------------

st.subheader("Exploring the Dataset")
st.write("Dataset Columns:")
st.write(data.columns)

# -------------------------------------------------------

st.subheader("Data Visualization")
st.write("It is considered a good practice to visualize the data at hand. So let’s plot our time series data:")


def plot_raw_data():
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data["Date"],
                  y=data['Open'], name="Crypto Open"))
    fig.add_trace(go.Scatter(x=data["Date"],
                  y=data['Close'], name="Crypto Close"))

    fig.layout.update(title="Time Series Data", xaxis_rangeslider_visible=True)
    st.plotly_chart(fig, use_container_width=True)


plot_raw_data()

# -------------------------------------------------------

st.subheader("Data Preparation")
st.write("The prophet has some limitations where it requires only to have 2 columns and in this case: the date and the close columns are the 2 columns needed.")
st.write("So, we must rename the columns in our dataframe. The rename function is used to change the name of the columns following the correct name conventions")
st.write("The .tail method is used to check the new data.")
df_train = data[['Date', 'Close']]
df_train = df_train.rename(columns={"Date": "ds", "Close": "y"})
st.write(df_train.tail())

# -------------------------------------------------------

st.subheader('Prophet Model')
st.write("The Prophet model has been used here.")
st.write("A fit method with the new data frame as input is being utilized.")
st.write("After that, The make_future_dataframe which is a handy method in prophet will generate daily timestamps based on the user choice of prediction")
st.write("The data frame with future dates is subsequently passed to the fitted model's forecast method.")
st.write("Again, the .tail method is used to check the new data.")

m = Prophet(interval_width=0.95)
m.fit(df_train)
future = m.make_future_dataframe(periods=period)
forecast = m.predict(future)
st.write(forecast.tail())

# -------------------------------------------------------


st.subheader("Visulaization of the Prophet Model")
forecast_fig = plot_plotly(m, forecast)
st.plotly_chart(forecast_fig, use_container_width=True, uncertainty=True)
st.write("•	The black dots represent the observed value of the time series.")
st.write("•	The blue line represents the forecasted values.")
st.write("•	The uncertainty intervals of the forecasts are the blue shaded region")


# -------------------------------------------------------


st.subheader("Plotting the Forecasted components")
st.write("Components of the prediction is one of Prophet's capabilities, and it can be used to demonstrate how daily, weekly, and annual patterns of the time series contribute to the overall forecasted values. ")
fig2 = m.plot_components(forecast)
st.write(fig2)

st.write("Wow!!!! The above plot provides interesting insights.")
st.write("•The first plot shows that the trend of coin prices that simulates non-periodic variations in time series data and how its linearly increasing over time.")
st.write("•The second plot highlights the fact that the weekly price of the coin  peaks towards the end of the week and on Saturday. and go down on Tuesday and Thursday")
st.write("•The third plot shows that the Highest price occurs between  January and May")


# -------------------------------------------------------

st.subheader("Adding ChangePoints to Prophet")

st.write("Changepoints are the datetime points where the time series have abrupt changes in the trajectory.")
st.write("By default, Prophet adds 25 changepoints to the initial 80% of the data-set.")
st.write("Let’s plot the vertical lines where the potential changepoints occurred")

# another way of plotting
#fig = m.plot(forecast)
#a = add_changepoints_to_plot(fig.gca(), m, forecast)
# st.pyplot(fig)

figure = m.plot(forecast)
for changepoint in m.changepoints:
    # axvline is used to add a vertical line across the axis
    plt.axvline(changepoint, ls='--', lw=1)
st.pyplot(figure)

# -------------------------------------------------------

st.write("We can view the dates where the chagepoints occurred.")
st.write(m.changepoints)

# -------------------------------------------------------

st.subheader("Adjusting Trend ")
st.write("Prophet allows us to adjust the trend in case there is an overfit or underfit.")
st.write("changepoint_prior_scale helps adjust the strength of the trend.")

# another way of visualizing the trend flexibilty
#pro_change = Prophet(yearly_seasonality=True, changepoint_prior_scale=0.08)
#forecast = pro_change.fit(df_train).predict(future)
#fig = pro_change.plot(forecast)
#a = add_changepoints_to_plot(fig.gca(), pro_change, forecast)
# st.pyplot(fig)

# Increasing Changepoint_prior_scale  will make the trend more flexible:
st.subheader("Higher Flexibilty:")
m = Prophet(changepoint_prior_scale=0.08)
forecast = m.fit(df_train).predict(future)
fig = m.plot(forecast)
st.pyplot(fig)
st.subheader("Lower Flexibilty:")
# Decreasing Changepoint_prior_scale  will make the trend less flexible:
n = Prophet(changepoint_prior_scale=0.001)
forecastn = n.fit(df_train).predict(future)
fign = n.plot(forecastn)
st.pyplot(fign)

# -------------------------------------------------------
