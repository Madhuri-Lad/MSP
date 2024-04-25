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
st.image('data:image/jpeg;base64,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')

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

st.write("<p style='color:green; font-size:50px; font-weight:bold;'>Forcasting the data</p>",unsafe_allow_html=True)


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
