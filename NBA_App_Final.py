# -*- coding: utf-8 -*-
"""
Created on Sat Aug  6 17:13:11 2022

@author: nagelnn1
"""
import datetime
from datetime import date

from sklearn.linear_model import LinearRegression
from sklearn import linear_model
from statsmodels.graphics.tsaplots import plot_predict
from statsmodels.tsa.arima.model import ARIMA
import plotly.express as px
import pandas as pd
from basketball_reference_scraper.players import get_stats, get_game_logs, get_player_headshot
from basketball_reference_scraper.teams import get_roster, get_team_stats, get_opp_stats, get_roster_stats, get_team_misc
from PandasBasketball import pandasbasketball as pb
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm
import warnings
warnings.filterwarnings("ignore")
import plotly.io as pio
import streamlit as st
import time

import plotly.graph_objects as go




#Data Retrieval==============================================================
players_final = pd.DataFrame()
teams = ['PHO','ATL', 'BRK','BOS', 'CHA', 'CHI', 'CLE', 'DAL', 'DEN', 'DET', 'GSW', 'HOU', 'IND', 'LAC','LAL', 'MEM', 'MIA', 'MIL', 'MIN', 'NOP', 'NYK', 'OKC', 'ORL', 'PHI', 'POR', 'SAC', 'SAS', 'TOR', 'UTA', 'WAS']

def get_data(start: int, teams: list) -> pd.core.frame.DataFrame:
    players_final = pd.DataFrame()
    for i in (range(start, (date.today().year)+1)):
        for j in (teams):
            df=get_roster_stats(j, i)
            df['YEAR'] = i
            players_final = pd.concat([df, players_final]).reset_index(drop=True)
    return players_final
error_list = []
error_years = []

#players = get_data
#name_list = players["PLAYER"].values.tolist()
#year_list = players['YEAR'].values.tolist()
def get_code(name):
    url = get_player_headshot(name)
    code = url.split('/')[-1][:-4]
    return code
def retrieve_data():
    players = get_data
    year_list = players['YEAR'].values.tolist()
    name_list = players["PLAYER"].values.tolist()
    games = pd.DataFrame()
    for ind in (range((len(name_list)))):
        try:
    
            code = get_code(name_list[ind])
            df = pb.get_player_gamelog(code, year_list[ind])
            df["NAME"] = name_list[ind]
            df['YEAR'] = year_list[ind]
            games = pd.concat([df, games], ignore_index=True)
            games.reset_index(inplace=True, drop=True)
    
        except AttributeError:
            
            error_list.append(name_list[ind])
            error_years.append(year_list[ind])
            next
        except:
            error_list.append(name_list[ind])
            error_years.append(year_list[ind])
            next
    games.to_csv('Final_5_year_log.csv')
    games2=pd.DataFrame()
    for i in (range(len(error_list))):
        try:
            try:
                code=pb.generate_code(error_list[i])
            except ValueError:
                code = get_code(error_list[i])
            except:
                next
    
            df = pb.get_player_gamelog(code, error_years[i])
            df["NAME"] = error_list[i]
            df['YEAR'] = error_years[i]
            games2 = pd.concat([df, games2], ignore_index=True)
            games2.reset_index(inplace=True, drop=True)
        except AttributeError:
            next
        except ValueError:
                code = get_code(error_list[i])
                df = pb.get_player_gamelog(code, error_years[i])
                df["NAME"] = error_list[i]
        except:
            next
    games = pd.concat([games2, games], ignore_index=True)
    games.to_csv('5_year_data_august Test.csv')

#=======================================================================





#DataFrame cleaning and adjusting
def get_data_name(name):
    names_df = pd.DataFrame()
    for ind in game_logs.index:
        if game_logs['NAME'][ind]==name:
            df1 = game_logs.loc[[ind]]
            names_df = pd.concat([names_df, df1])
    names_df.reset_index(inplace=True, drop=True)
    names_df['Year'] = 0
    for i in names_df.index:
        year = int((names_df['Date'][i])[-4:])
        names_df['Year'][i] = year
    return names_df

def set_df(df):
    df.set_index('Date', inplace = True)
    df.index = pd.to_datetime(df.index)
    df.drop(df.columns.difference(['Date', 'PTS', 'TRB', 'AST', 'STL', 'BLK', 'TOV', 'Year']), 1, inplace=True)
    df.dropna(inplace=True)
    df.drop_duplicates(inplace=True)
    df=df.resample('M', convention = 'start').mean().dropna()
    df.index= df.index.to_period("M")
    df.index = df.index.to_timestamp()
    return df

#Make a dropdown of every NBA Player and then get_data for selection and set it

def break_dates(df):
    break_dates=[]
    df_dates = pd.date_range(start=df.index[0], end = df.index[-1])
    for i in df_dates:
        series = (df.index).to_series()
        if(i not in series):
            break_dates.append(i)
    return break_dates


#moving averages
def smas(df, metric):
    df['15SMA'] = df[metric].rolling(window=20).mean()
    df['15dayEWM'] = df[metric].ewm(span=15, adjust=False).mean()


#more DF Formatting for the Modles
def fill_df2(metric, df):
    dates = pd.date_range(start=df.index[0], end = df.index[0]+datetime.timedelta(days=len(df.index)-1))
    df_dates2 = pd.DataFrame({'Date': dates})
    df_dates2[metric] = 5
    df_dates2 = df_dates2.set_index('Date')
    for i in range(0,len(df)):
        df_dates2[metric][i] = df[metric][i]

    return df_dates2


def fix(df1, forecast):
    df1['game_number']=0 
    for i in range(len(df1)):
        df1['game_number'][i] = i+1
    for i in range(len(forecast)-len(df1)): #values to predict
        last_date = df1.index[-1] + datetime.timedelta(days=1)
        df1 = df1.append(pd.DataFrame(index=[last_date]))
        df1['game_number'][-1] =  len(df1)
    return df1

def put_game_num(df):
    indexs= len(df)
    df['game_number'] = 0
    for i in range(0, len(df)):
        df['game_number'][i] = i+1
        print(df.head)
    return df


#ARMA FOrecast, returns a DF With the predicted and actual lines
def ARMA_forecast(df, stat):
    df3 = fill_df2(stat,df)


    datalen = len(df3.index)
    idx = pd.date_range(df3.index[0], df3.index[-1]) #change to last day on index
    #df3 = df3.reindex(idx, fill_value=0) #Fill with SMA At that point
    res = ARIMA(df3[stat], order=(25,0,15)).fit() #choosing right parameters
    fig, ax = plt.subplots()
    ax = df3.loc[(df3.index[-1]-datetime.timedelta(days=180)).strftime('%Y-%m-%d'):].plot(ax=ax)
    plot_predict(res, (df3.index[0]+datetime.timedelta(days=round(datalen*.7))).strftime('%Y-%m-%d'), (df3.index[-1]+datetime.timedelta(days=30)).strftime('%Y-%m-%d'), ax=ax)
    data = res.predict((df3.index[0]+datetime.timedelta(days=round(datalen*.7))).strftime('%Y-%m-%d'), (df3.index[-1]+datetime.timedelta(days=30)).strftime('%Y-%m-%d')) #Join to main DF Loop 4 times
        #checks at beginning before model like take out injured, run through tests
    plt.show()
    df3 = pd.concat([df3, data], axis=1)
    
    df3['game_number'] = 0
    for i in range(0, len(df3)):
        df3['game_number'][i] = i+1
        
    return df3

#Linear Reg Forecast: returns DF with predicted and actual lines, print r^2, put error to show how good it fits

def linear_reg_forecast(df, metric):
    smas(df, metric)
    df['game_number']=0 
    for i in range(len(df)):
        df['game_number'][i] = i+1 #make column
    X = df['game_number'].values.reshape(-1, 1)
    Y = df['15dayEWM'].values
    model = LinearRegression()  
    model.fit(X, Y)  # perform linear regression
    vals = []
    Y_pred = model.predict(X)  # make predictions
    for i in range(30): #values to predict
        vals.append(df['game_number'][-1]+(i+1))
    new_x = pd.DataFrame()
    new_x['X'] = vals
    new_x = new_x['X'].values.reshape(-1,1)
    #print(mean_squared_error(Y, Y_pred)) #not needed
    #print(r2_score(Y, Y_pred)) #not needed
    y_pred = model.predict(new_x)
    y_final = np.append(Y_pred,y_pred)

    points_df = fill_df2(metric,df)
    df2 = pd.DataFrame(y_final) 
    points_df = points_df.reset_index()
    df2=pd.concat([df2, points_df],axis=1)
    df2['game_number'] = 0
    for i in range(0, len(df2)):
        df2['game_number'][i] = i+1
    df2.drop(columns=['Date'],inplace=True)
    df2.rename(columns={0: 'predicted_mean'}, inplace=True)
    return df2




#Streamlit App============================================================

def players_dropdown():
    players = []
    for i in game_logs.index:
        players.append(game_logs['NAME'][i])
    res = [*set(players)]
    res.sort()
    return res

#Get data===================================================================================
game_logs = pd.read_csv(r'C:\Users\nagelnn1\Documents\aspire\NBA_Predictions_App\Final_5_year_log.csv')
game_logs.drop_duplicates()
game_logs.drop('Unnamed: 0', axis=1, inplace=True)
players_list = players_dropdown()



#SideBar
CHOICES = {'PTS': "Points", 'TRB': "Total Rebounds", 'AST': "Assists", 'STL': 'Steals', 'BLK': "Blocks", 'TOV': 'Turnovers'}

def format_func(option):
    return CHOICES[option]



players_select = st.sidebar.selectbox(
    "Choose a Player:",
    (players_list), index= players_list.index('Stephen Curry')
)


with st.sidebar:
    player2 = st.selectbox('Choose a Player to Compare: ', (players_list),index= players_list.index('LeBron James') )

with st.sidebar:
    stat_select = st.selectbox("Statistic:", options=list(CHOICES.keys()), format_func=format_func, index=0)

with st.sidebar:
    model_selector = st.radio(
        "Choose a model to use:",
        ("Simple Linear Regression", "ARIMA"), index=1
    )
with st.sidebar:
    st.image(image = get_player_headshot(players_select))
    st.image(image = get_player_headshot(player2))
   
    st.subheader('Selected Parameters')

    st.write(("Name: " + players_select))
    st.write(("Player to Compare to: " + player2))
    st.write(('Stat: ' + format_func(stat_select)))
df1 = get_data_name(players_select)
set_df(df1)

df_compare = get_data_name(player2)
set_df(df_compare)

if (model_selector == 'Simple Linear Regression'):
    df2 = linear_reg_forecast(df1, stat_select)
    df_c = linear_reg_forecast(df_compare, stat_select)
elif(model_selector == 'ARIMA'):
    df2 = ARMA_forecast(df1, stat_select)
    df_c = ARMA_forecast(df_compare, stat_select)




#Graphs
df_smas = df2.drop(df2.tail(30).index)
smas(df_smas, stat_select)

smas_graph = px.line(df_smas, x=df_smas['game_number'], y=[df_smas[stat_select], df_smas['15SMA']], markers=True, width = 900, height = 500)
smas_graph.update_layout(xaxis_title='Games Since 2018', yaxis_title=format_func(stat_select),xaxis={"rangeslider":{"visible":True}})

EWMA_graph = px.line(df_smas, x=df_smas['game_number'], y=[df_smas[stat_select], df_smas['15dayEWM']], markers=True,width = 900, height = 500 )
EWMA_graph.update_layout(xaxis_title='Games Since 2018', yaxis_title=format_func(stat_select))
EWMA_graph.update_layout(hovermode="x",xaxis={"rangeslider":{"visible":True}})

df2.rename(columns = {'predicted_mean':'Predictions'}, inplace = True)
prediction_graph = px.line(df2, x=df2['game_number'], y=[df2[stat_select], df2['Predictions']], title=players_select + ' Predicted Stats', markers=True,width = 900, height = 600)
prediction_graph.update_layout( xaxis_title='Games Since 2018', yaxis_title=format_func(stat_select))
prediction_graph.update_layout(hovermode="x",xaxis={"rangeslider":{"visible":True}})



df_c.rename(columns = {'predicted_mean':'Predictions'}, inplace = True)
compare_graph = px.line(df2, x=df2['game_number'], y=df2['Predictions'], title='Future Comparisons', markers=True,width = 900, height = 600)



compare_graph = go.Figure(layout=go.Layout(height=600, width=900))


compare_graph.add_trace(go.Line(x=df2['game_number'], y=df2['Predictions'], name = players_select + " predictions"))
compare_graph.update_layout(xaxis_title='Games Since 2018', yaxis_title=format_func(stat_select))


compare_graph.add_trace(go.Line(x=df_c['game_number'], y=df_c['Predictions'], name =(player2 + " predictions")))
compare_graph.update_layout(hovermode="x", showlegend = True, xaxis={"rangeslider":{"visible":True}})







#App Formatting


#to do: add title, show raw data and all the game logs

st.title('NBA Stats Predictor')
st.markdown('The NBA is a game of change, and a players performance frequently changes. If you ever wanted to know how a player will progress and play in their next few games, this is the app for you. Select a player from the dropdown, the stat, and the model to use. You will then be given the moving averages for tracking and the predictions below, which can all be exported to graphs for your own use. Good luck! ')
def convert_df(df):
     return df.to_csv().encode('utf-8')




tab1, tab2, tab3, tab4 = st.tabs(['Simple Moving Average', "Exponentially Weighted Moving Average", 'Predictions', 'Comparison'])

with tab1:
    st.subheader("Simple Moving Average")
    st.write(smas_graph)
    with st.expander("See Raw Data"):
                df_export1 = df1.assign(EWM=df_smas['15dayEWM'], SMA = df_smas['15SMA'])
                st.write(df_export1)
                st.download_button(label = 'Download Data⬇️', key = 'download1', data = convert_df(df_export1), file_name =players_select+ "_data.csv")


with tab2:
    st.subheader("Exponentially Weighted Moving Average")
    st.write(EWMA_graph)
    with st.expander("See Raw Data"):
                df_export1 = df1.assign(EWM=df_smas['15dayEWM'], SMA = df_smas['15SMA'])
                st.write(df_export1)
                st.download_button(label = 'Download Data⬇️', key='download2', data = convert_df(df_export1), file_name =players_select+ "_data.csv")


with tab3:
    st.subheader("Predictions")
    st.write(prediction_graph)
    with st.expander("Export Predictions"):
        st.write(df2)
        st.download_button(label = 'Download Predictions⬇️', data = convert_df(df2), file_name =players_select+'_'+stat_select + '_'+"predictions.csv")

with tab4:
    st.subheader('Comparison of ' + players_select + " and " + player2)
    player1 = st.checkbox('Show Stats of ' + players_select)
    player2_cb = st.checkbox('Show Stats of ' + player2)
    if player2_cb:
        compare_graph.add_trace(go.Scatter(x=df_c['game_number'], y=df_c[stat_select], name =(player2 + " " + stat_select)))
    if player1:
        compare_graph.add_trace(go.Scatter(x=df2['game_number'], y=df2[stat_select], name =(players_select + " " + stat_select)))
    st.write(compare_graph)
    with st.expander("Export Predictions"):
        st.write(df_c)
        st.download_button(label = 'Download Predictions⬇️', data = convert_df(df_c), file_name =players_select+'_'+stat_select + '_'+"predictions.csv")   


st.header('Credits')
st.write("App made by Naman Nagelia")
st.write("API to gather Basketball Reference Data https://github.com/vishaalagartha/basketball_reference_scraper")
st.write("All data from basketball Reference")    

#Deploy in a cloud and try to schedule : Amazon S3 to store file, EC2 - Server to run jobs

#1. Tabs, put predicitons in a tab
#2: Put each table under graph and reformat