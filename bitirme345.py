# -*- coding: utf-8 -*-
"""
Created on Fri May  5 21:09:38 2023

@author: Bilge
"""

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
        
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

url='C:/Users/Bilge/Desktop/vaccination_tweets.csv'
df=pd.read_csv(url)
df.head()
      
df.info()
df.columns
df['user_verified']=df['user_verified'].apply(lambda x:'verified' if x==True else 'not_verified')

from datetime import date
df['today']=date.today()
df['user_created']=pd.to_datetime(df['user_created']).dt.year
df['today']=pd.to_datetime(df['today'])
df['today']=df['today'].dt.year
df['acc_age']= df['today']-df['user_created']

print(max(df['date']))
print(min(df['date']))

df['date']=pd.to_datetime(df['date'])

L = ['year', 'month', 'day', 'dayofweek', 'dayofyear', 'weekofyear', 'quarter']
df = df.join(pd.concat((getattr(df['date'].dt, i).rename(i) for i in L), axis=1))

df['total_engagement']=df['retweets']+df['favorites']
df['text'].values[:2]
df['tweet_lenght']=df['text'].apply(lambda x:len(x))
df['tweet_lenght'].describe()

df['tweet_length']=df['text'].apply(lambda x:'short' if len(x)<=130 else 'long')
df['user_location'].value_counts()
 
loc_df = df['user_location'].str.split(',',expand=True)
loc_df=loc_df.rename(columns={0:'fst_loc',1:'snd_loc'})

# Remove Spaces 
loc_df['snd_loc'] = loc_df['snd_loc'].str.strip()
# Rename States 
state_fix = {'Ontario': 'Canada','United Arab Emirates': 'UAE','TX': 'USA','NY': 'USA'
                  ,'FL': 'USA','England': 'UK','Watford': 'UK','GA': 'USA','IL': 'USA'
                  ,'Alberta': 'Canada','WA': 'USA','NC': 'USA','British Columbia': 'Canada','MA': 'USA','ON':'Canada'
            ,'OH':'USA','MO':'USA','AZ':'USA','NJ':'USA','CA':'USA','DC':'USA','AB':'USA','PA':'USA','SC':'USA'
            ,'VA':'USA','TN':'USA','New York':'USA','Dubai':'UAE','CO':'USA'}
loc_df = loc_df.replace({"snd_loc": state_fix}) 
loc_df['snd_loc'].value_counts()[:20]

df['Hash'] = df['text'].apply(lambda word:word.count('#'))

df['Men'] = df['text'].apply(lambda word:word.count('@'))

#('https://t.co/) this part in tweets refers to photos,videos
df['med'] = df['text'].apply(lambda word:word.count('https://t.co/'))
df['med'] = df['med'].apply(lambda x:'No Media' if x==0 else 'Media')


df['user_followers'].value_counts()

df['acc_class'] = df['user_followers'].apply(lambda x:'week'if x<=100 else ('norm' if 1000>=x>100 else 
                                                                       ('strong' if 10000>=x>1000
                                                                        else 'influencer')))
df.head()

df.columns

df=df[['user_name','text','date', 'acc_age','user_verified','retweets','favorites','total_engagement', 'day', 'tweet_length',
       'Hash', 'Men', 'med', 'acc_class','month']]
df_copy=df.copy()
df.head()

corr=df.corr()
plt.figure(figsize=(10,7))
sns.heatmap(corr,annot=True)

plt.figure(figsize=(7,7))
sns.countplot(x='tweet_length',data=df);

plt.figure(figsize=(7,7))
sns.barplot(x=df['Hash'],y=df['tweet_length'],data=df);

plt.figure(figsize=(7,7))
sns.barplot(x=df['tweet_length'],y=df['Men'],data=df);

df['user_verified'].value_counts()

labels = 'not_verified', 'verified'
sizes = [1888, 319]
explode = (0.1, 0)  
plt.figure(figsize=(10,5))
plt.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
        shadow=True, startangle=90);
plt.axis('equal'); 

class_eng = df.groupby('acc_class',as_index=False).agg({'total_engagement':'sum',})
fig = px.bar(class_eng,
             x='acc_class',
             y='total_engagement',
             color='total_engagement',
             color_continuous_scale='Rainbow',
             title='Engagement By Account_Class')
fig.show()

Media = len(df[df['med']=='Media'])
No_Media = len(df[df['med']=='No Media'])
Platform = ['Media','No Media']
Count = [Media,No_Media]
#====
fig = px.pie(names = Platform,
             values = Count,
             title='Media/No Media',
            color_discrete_sequence = px.colors.sequential.Rainbow)
fig.update_traces(textposition='inside', textinfo='percent+label')

line = df.groupby('date',as_index=False).agg({'total_engagement':'sum'})
fig = go.Figure()
fig.add_trace(go.Scatter(x=line.date, y=line.total_engagement,
                    mode='lines+markers'))

december=df.loc[df['month']==12]
day_december = december.groupby('day',as_index=False).agg({'total_engagement':'sum'})

fig = px.scatter(day_december,
                 x='day',
                 y='total_engagement',
                 color_continuous_scale='Rainbow',
                 color='total_engagement',
                 size='total_engagement',
                 title='Most engage days in Decembre')
fig.show()

ret = df.groupby('user_name',as_index=False).agg({'retweets':'sum'}).sort_values('retweets',ascending=False).head(10)
like = df.groupby('user_name',as_index=False).agg({'favorites':'sum'}).sort_values('favorites',ascending=False).head(10)
tot_eng = df.groupby('user_name',as_index=False).agg({'total_engagement':'sum'}).sort_values('total_engagement',ascending=False).head(10)


fig = px.bar(tot_eng,
             x='user_name',
             y='total_engagement',
             color='total_engagement',
             color_continuous_scale='Viridis',
             title='Accounts / Engagements')
fig.show()

age=df.groupby('acc_age',as_index=False).agg({'total_engagement':'sum'})
px.line(age,x='acc_age',y='total_engagement',labels={'x':'age','y':'engagement'})

import seaborn as sns 
plt.figure(figsize=(14,7))
sns.countplot(x='acc_age',data=df_copy);

df3=pd.DataFrame(loc_df['snd_loc'].value_counts()[:20]).reset_index()
df3


fig = px.choropleth(df3, locations = df3['index'],
                    color = df3['snd_loc'],locationmode='country names',hover_name = df3['snd_loc'], 
                    color_continuous_scale = px.colors.sequential.Inferno)
fig.update_layout(title='Sales tracking')
fig.show()

fig = px.choropleth(df3, locations = df3['index'],
                    color = df3['snd_loc'],locationmode='country names',hover_name = df3['snd_loc'], 
                    color_continuous_scale = px.colors.sequential.Inferno)
fig.update_layout(title='Sales tracking')
fig.show()

fig = px.choropleth(df3, locations = df3['index'],
                    color = df3['snd_loc'],locationmode='country names',hover_name = df3['snd_loc'], 
                    color_continuous_scale = px.colors.sequential.Inferno)
fig.update_layout(title='Sales tracking')
fig.show()

fig = px.choropleth(df3, locations = df3['index'],
                    color = df3['snd_loc'],locationmode='country names',hover_name = df3['snd_loc'], 
                    color_continuous_scale = px.colors.sequential.Inferno)
fig.update_layout(title='Sales tracking')
fig.show()
