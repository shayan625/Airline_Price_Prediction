#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


train_data=pd.read_excel(r"C:\Users\91912\Downloads\Data_Train.xlsx")


# In[3]:


train_data.head(4)


# In[4]:


train_data.tail(4)


# In[5]:


train_data.isnull().sum()


# In[6]:


train_data['Total_Stops'].isnull()


# In[7]:


train_data[train_data['Total_Stops'].isnull()]


# In[8]:


train_data.dropna(inplace=True)


# In[9]:


train_data.isnull().sum()


# In[10]:


train_data.dtypes


# In[11]:


data=train_data.copy()


# In[12]:


data.head(2)


# In[13]:


def change_into_datetime(col):
    data[col]=pd.to_datetime(data[col])


# In[14]:


import warnings
from warnings import filterwarnings
filterwarnings("ignore")


# In[15]:


for feature in ['Dep_Time','Arrival_Time','Date_of_Journey']:
    change_into_datetime(feature)


# In[16]:


data.dtypes


# In[17]:


data['Journey_day']=data['Date_of_Journey'].dt.day


# In[18]:


data['Journey_month']=data['Date_of_Journey'].dt.month


# In[19]:


data['Journey_year']=data['Date_of_Journey'].dt.year


# In[20]:


data.head(3)


# In[21]:


def extract_hour_min(df,col):
    df[col+"_hour"]=df[col].dt.hour
    df[col+"_minute"]=df[col].dt.minute
    return df.head(3)


# In[22]:


extract_hour_min(data,"Dep_Time")


# In[23]:


extract_hour_min(data,"Arrival_Time")


# In[24]:


cols_to_drop=['Arrival_Time','Dep_Time']
data.drop(cols_to_drop,axis=1,inplace=True)


# In[25]:


data.head(3)


# In[26]:


data.shape


# In[27]:


def flight_dep_time(x):
    if(x>4) and (x<=8):
        return "Early Morning"
    elif(x>8) and (x<=12):
        return "Morning"
    elif(x>12) and (x<=16):
        return "Noon"
    elif(x>16) and (x<=20):
        return "Evening"
    elif (x>20) and (x<=24):
        return "Night"
    else:
        return "Late Night"


# In[28]:


data['Dep_Time_hour'].apply(flight_dep_time).value_counts()


# In[29]:


data['Dep_Time_hour'].apply(flight_dep_time).value_counts().plot(kind="bar",color="green")


# In[30]:


def preprocess_duration(x):
    if 'h' not in x:
        x = '0h' + ' ' + x
    elif 'm' not in x:
        x = x + ' ' + '0m'
    return x


# In[31]:


data['Duration']=data['Duration'].apply(preprocess_duration)


# In[32]:


data['Duration']


# In[33]:


'2h 50m'.split(' ')[0][0:-1]


# In[34]:


int('2h 50m'.split(' ')[0][0:-1])


# In[35]:


int('2h 50m'.split(' ')[1][0:-1])


# In[36]:


data['Duration_hours']=data['Duration'].apply(lambda x : int(x.split(' ')[0][0:-1]))


# In[37]:


data['Duration_minutes']=data['Duration'].apply(lambda x : int(x.split(' ')[0][0:-1]))


# In[38]:


data.head(2)


# In[39]:


data['Duration_total_mins']=data['Duration'].str.replace('h','*60').str.replace(' ','+').str.replace('m','*1').apply(eval)


# In[40]:


data.head(2)


# In[41]:


sns.scatterplot(x='Duration_total_mins',y='Price',data=data)


# In[42]:


sns.scatterplot(x='Duration_total_mins',y='Price',data=data,hue='Total_Stops')


# In[43]:


data[data['Airline']=='Jet Airways'].groupby('Route').size().sort_values(ascending=False)


# In[44]:


sns.boxplot(x='Airline',y='Price',data=data.sort_values('Price',ascending=False))
plt.xticks(rotation='vertical')
plt.show()


# In[45]:


cat_col = [col for col in data.columns if data[col].dtype=='object']


# In[46]:


num_col = [col for col in data.columns if data[col].dtype!='object']


# In[47]:


cat_col


# In[48]:


num_col


# In[49]:


data['Source'].unique()


# In[50]:


data['Source'].apply(lambda x:1 if x=='Banglore' else 0)


# In[51]:


for sub_category in data['Source'].unique():
    data['Source_'+sub_category]=data['Source'].apply(lambda x:1 if x==sub_category else 0)


# In[52]:


data.head(3)


# In[53]:


data['Airline'].unique()


# In[54]:


airlines=data.groupby(['Airline'])['Price'].mean().sort_values().index


# In[55]:


airlines


# In[56]:


dict_airlines={key:index for index,key in enumerate(airlines,0)}


# In[57]:


dict_airlines


# In[58]:


data['Airline']=data['Airline'].map(dict_airlines)


# In[59]:


data['Airline']


# In[60]:


data.head(3)


# In[61]:


data['Destination'].replace('New Delhi','Delhi',inplace=True)


# In[62]:


data['Destination'].unique()


# In[63]:


dest=data.groupby(['Destination'])['Price'].mean().sort_values().index


# In[64]:


dest


# In[65]:


dict_dest={key:index for index,key in enumerate(dest,0)}


# In[66]:


dict_dest


# In[67]:


data['Destination']=data['Destination'].map(dict_dest)


# In[68]:


data.head(3)


# In[69]:


data['Total_Stops']


# In[70]:


data['Total_Stops'].unique()


# In[71]:


stop={'non-stop':0, '2 stops':2, '1 stop':1, '3 stops':3, '4 stops':4}


# In[72]:


data['Total_Stops']=data['Total_Stops'].map(stop)


# In[73]:


data['Total_Stops']


# In[74]:


data.columns 


# In[75]:


data.drop(columns=['Date_of_Journey','Additional_Info', 'Duration_total_mins','Source','Route','Journey_year','Duration'],axis=1,inplace=True)


# In[76]:


data.head(3)


# In[85]:


def plot(df,col):
     fig,(ax1,ax2,ax3)=plt.subplots(3,1)
     sns.distplot(df[col],ax=ax1)
     sns.boxplot(df[col],ax=ax2)
     sns.distplot(df[col],ax=ax3,kde=False)


# In[86]:


plot(data,'Price')


# In[87]:


q1=data['Price'].quantile(0.25)
q3=data['Price'].quantile(0.75)

iqr=q3-q1

maximum=q3+1.5*iqr
minimum=q1-1.5*iqr


# In[90]:


print(maximum)


# In[91]:


print(minimum)


# In[93]:


print([price for price in data['Price'] if price>maximum or price<minimum])


# In[94]:


len([price for price in data['Price'] if price>maximum or price<minimum])


# In[95]:


data['Price']=np.where(data['Price']>=35000,data['Price'].median(),data['Price'])


# In[96]:


plot(data,'Price')


# In[97]:


x=data.drop(['Price'],axis=1)


# In[98]:


y=data['Price']


# In[99]:


from sklearn.feature_selection import mutual_info_regression


# In[100]:


imp=mutual_info_regression(x,y)


# In[101]:


imp


# In[104]:


imp_df=pd.DataFrame(imp,index=x.columns)


# In[108]:


imp_df.columns=['importance']


# In[109]:


imp_df


# In[111]:


imp_df.sort_values(by='importance',ascending=False)


# In[112]:


from sklearn.model_selection import train_test_split


# In[114]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=42)


# In[115]:


from sklearn.ensemble import RandomForestRegressor


# In[116]:


ml_model=RandomForestRegressor()


# In[117]:


ml_model.fit(x_train,y_train)


# In[118]:


y_pred=ml_model.predict(x_test)


# In[119]:


y_pred


# In[120]:


from sklearn import metrics


# In[121]:


metrics.r2_score(y_test,y_pred)


# In[122]:


def mape(y_true,y_pred):
    y_true,y_pred=np.array(y_true),np.array(y_pred)
    return np.mean((np.abs(y_true-y_pred)/y_true)*100)


# In[123]:


mape(y_test,y_pred)


# In[124]:


from sklearn import metrics


# In[127]:


def predict(ml_model):
    model=ml_model.fit(x_train,y_train)
    print('Training score : {}'.format(model.score(x_train,y_train)))
    y_prediction=model.predict(x_test)
    print('prediction are : {}'.format(y_prediction))
    print('\n')
    r2_score=metrics.r2_score(y_test,y_prediction)
    print('r2_score : {}'.format(r2_score))
    print('MAE : {}'.format(metrics.mean_absolute_error(y_test,y_prediction)))
    print('MSE : {}'.format(metrics.mean_squared_error(y_test,y_prediction)))
    print('RMSE : {}'.format(np.sqrt(metrics.mean_squared_error(y_test,y_prediction))))
    print('MAPE : {}'.format(mape(y_test,y_prediction)))
    sns.distplot(y_test-y_prediction)


# In[128]:


predict(RandomForestRegressor())

