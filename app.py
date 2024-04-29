import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
import datetime
from datetime import date, timedelta
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
import statsmodels.api as sm

# Title

app_name='Stock marketing analysis'
st.title(app_name)
st.subheader('This App is created to forcast the stock market price of the selected company')
# add an image
st.image('https://m.economictimes.com/thumb/msid-102245149,width-400,height-250,align-center,resizemode-4,imgsize-104000/unleashing-the-bulls-how-the-stock-market-achieved-unprecedented-record-levels.jpg')

# take input from user
st.sidebar.header('Select the parametes from below')

start_date=st.sidebar.date_input('Start date', date(2020,1,1))
end_date=st.sidebar.date_input('End date', date(2020,1,1))
#add ticker symbol list
ticker_list=["AAPL","MSFT","GOOGL","META","TSLA","NVDA","ADBE","PYPL","INTC","CMCSA","NFLX","PEP"]
ticker=st.sidebar.selectbox('Select company', ticker_list)

data=yf.download(ticker,start=start_date,end=end_date)
#add date as a colom
data.insert(0,"Date",data.index,True)
data.reset_index(drop=True,inplace=True)
st.write('Data from', start_date,'to' ,end_date)
st.write(data)

#plot the data
st.header('Data Visualization')
st.subheader('Plot of the data')
st.write("**Note:** Select your specific date range on the sidebar, or zoom in on the plot and select your specific column")
fig=px.line(data, x="Date", y=data.columns, title='Closing price of the stock', template='plotly_dark', width=1000, height=600)
st.plotly_chart(fig)

#add a selct box to column from data
column=st.selectbox('Selct the column to be used for forcasting', data.columns[1:])

#subseting the data
data=data[['Date',column]]
st.write("Selected Data")
st.write(data)

#ADF test check stationarity
st.header("Is data stationary")
st.write('**NOTE:** IF p-value is less than 0.05, then data is stationary')
st.write(adfuller(data[column])[1]<0.05)

#Decomposition data
st.header('Decomposition of the data')
decomposition=seasonal_decompose(data[column],model='additive',period=12)
st.write(decomposition.plot())

#make same plot in plotly
st.plotly_chart(px.line(x=data["Date"],y=decomposition.trend, title='Trend',width=800,height=400,labels={'x':'Date','y':'Price'}).update_traces(line_color='Blue'))
st.plotly_chart(px.line(x=data["Date"],y=decomposition.seasonal, title='Seasonal',width=800,height=400,labels={'x':'Date','y':'Price'}).update_traces(line_color='Green'))
st.plotly_chart(px.line(x=data["Date"],y=decomposition.resid, title='Residuals',width=800,height=400,labels={'x':'Date','y':'Price'}).update_traces(line_color='Blue',line_dash='dot'))

#run model
#user input for three parameters
p=st.slider('Select the value of p',0,5,2)
d=st.slider('Select the value of d',0,5,1)
q=st.slider('Select the value of q',0,5,2)
seasonal_order=st.number_input('Select the value of seasonal p',0,24,12)

model=sm.tsa.statespace.SARIMAX(data[column],order=(p,d,q),seasonal_order=(p,d,q,seasonal_order))
model=model.fit()

#print model summary
st.header('Model Summary')
st.write(model.summary())
st.write("---")

st.write("<p style='color:green; font-size:50px; font-weight:bold;width:250;'>Forcasting the data</p>",unsafe_allow_html=True)


#predit the future values(Forcasting)
forcast_period=st.number_input('## Enter forcast period in days',1,365,10)
#predict the future values
predictions=model.get_prediction(start=len(data),end=len(data)+forcast_period)
predictions=predictions.predicted_mean

predictions.index=pd.date_range(start=end_date,periods=len(predictions),freq='D')
predictions=pd.DataFrame(predictions)
predictions.insert(0,'Date',predictions.index)
predictions.reset_index(drop=True,inplace=True)
st.write("## Predictons",predictions)
st.write("## Actual Data", data)

#lets ploot
fig=go.Figure()
#ass actual data to the plot
fig.add_trace(go.Scatter(x=data["Date"],y=data[column],mode='lines', name='Actual', line=dict(color='blue')))
#add predicted data to the plot
fig.add_trace(go.Scatter(x=predictions["Date"],y=predictions['predicted_mean'],mode='lines', name='Predicted', line=dict(color='red')))
#set the titel and axis labels
fig.update_layout(title='Actual v/s Predicted',xaxis_title='Date',yaxis_title='Price',width=800,height=400)
#display plot
st.plotly_chart(fig)


#add puttons to show and hide plots
show_plots=False
if st.button('Show separate plots'):
    if not show_plots:
        st.write(px.line(x=data["Date"],y=data[column],title='Actual',width=800,height=400,labels={'x':'Date','y':'Price'}).update_traces(line_color="Blue"))
        st.write(px.line(x=predictions["Date"],y=predictions['predicted_mean'],title='Predicted',width=800,height=400,labels={'x':'Date','y':'Price'}).update_traces(line_color="Red"))
        show_plots=True
    else:
        show_plots=False    

hide_plot=False
if st.button("Hide Separate plots"):
    if not hide_plot:
        hide_plot=True
    else:
        hide_plot=False    


show_plots1=False
if st.button('Compare plots'):
    if not show_plots1:
        fig2=go.Figure()
        fig2.add_trace(go.Scatter(x=data["Date"],y=data[column],mode='lines', name='Actual', line=dict(color='blue')))

        fig2.add_trace(go.Scatter(x=predictions["Date"],y=predictions['predicted_mean'],mode='lines', name='Predicted', line=dict(color='red')))

        fig2.update_layout(title='Predicted',xaxis_title='Date',yaxis_title='Price',width=800,height=400)

        #st.plotly_chart(fig2)

        newend_date=end_date + timedelta(days=forcast_period)

        data=yf.download(ticker,start=start_date,end=newend_date)

        data.insert(0,"Date",data.index,True)
        data.reset_index(drop=True,inplace=True)
        # st.write('Data from', start_date ,'to' ,end_date )
        # st.write(data)

        #fig1=go.Figure()

        fig2.add_trace(go.Scatter(x=data["Date"],y=data[column],mode='lines', name='Real', line=dict(color='green')))
        fig2.update_layout(title='Actual',xaxis_title='Date',yaxis_title='Price',width=800,height=400)
        st.plotly_chart(fig2)
        
        
        
        show_plots=True

hide_plot1=False
if st.button("Hide compared plots"):
    if not hide_plot1:
        hide_plot1=True
    else:
        hide_plot1=False  
st.write("---")

st.write("<p style='color:Blue; font-weight:bold; font-size:50px;'>Created by:- Madhuri Lad & Pallavi Jha</p>",unsafe_allow_html=True)
