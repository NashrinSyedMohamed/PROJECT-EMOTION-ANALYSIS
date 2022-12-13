import pandas as pd
import json
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.datasets import make_circles
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN

#-------------------------------------------------------------------#
#App Layout
st.title("Anomaly Detection")
sb=st.sidebar
Model = sb.selectbox(label = "Select the Machine-Learning Model", 
                      options = ['Isolation_Forest', 'DB_Scan'])
contamination = sb.slider("Set the Contamination", 10.0,0.01,0.05)
                          
#st.sidebar.write("Evaluation Metrics")
metrics = st.sidebar.selectbox(
label="Select the Metric",
    options=['Light_Level','Humidity','Pressure','Temperature'])

#-------------------------------------------------------------------#

#---------------------------------#
# Model Training Isolation Forest #
#---------------------------------#
def s3_access():
    from io import StringIO
    import boto3
    import pandas as pd
    import json
    s3 = boto3.client('s3', aws_access_key_id = 'AKIA4QAYWVI73V4KKXHC', aws_secret_access_key = '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')
    csv_obj = s3.get_object(Bucket = 'mylabdata', Key = 'Sensmitter_01.csv')
    body = csv_obj['Body']
    csv_string = body.read().decode('utf-8')
    df1 = pd.read_csv(StringIO(csv_string))
    p = df1['payload'].apply(json.loads)
    json_DF1 = pd.json_normalize(p)
    return json_DF1

def data_format(df_train):
    json_DF1['timestamp.S'] = pd.to_numeric(json_DF1['timestamp.S'])
    json_DF1['data.M.temperature.S'] = pd.to_numeric(json_DF1['data.M.temperature.S'])
    json_DF1['data.M.humidity.S'] = pd.to_numeric(json_DF1['data.M.humidity.S'])
    json_DF1['data.M.pressure.S'] = pd.to_numeric(json_DF1['data.M.pressure.S'])
    json_DF1['data.M.light_level.S'] = pd.to_numeric(json_DF1['data.M.light_level.S'])
    df1 = pd.DataFrame(json_DF1, columns = ['TimeStamp','Humidity', 'Pressure','Light_Level','Temperature'])
    df1['TimeStamp'] = json_DF1['timestamp.S']
    df1['Humidity'] = json_DF1['data.M.humidity.S']
    df1['Pressure'] = json_DF1['data.M.pressure.S']
    df1['Light_Level'] = json_DF1['data.M.light_level.S']
    df1['Temperature'] = json_DF1['data.M.temperature.S']
    return df1
json_DF1 = s3_access()
df_train=data_format(json_DF1)
#Isolation_Forest
from sklearn.ensemble import IsolationForest
clf = IsolationForest(contamination = contamination)
clf.fit(df_train)
    

#----------------------------------------------#
#Initialising connection to DynamoDB using Boto3
#----------------------------------------------#
import boto3
resource = boto3.resource('dynamodb', region_name='us-east-1',aws_access_key_id = 'AKIA4QAYWVI73V4KKXHC', aws_secret_access_key = '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')
client = boto3.client('dynamodb', region_name='us-east-1',aws_access_key_id = 'AKIA4QAYWVI73V4KKXHC', aws_secret_access_key = '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')
table = resource.Table('lab_sensmitter_1')
from boto3.dynamodb.conditions import Key, Attr
response = table.scan(
    FilterExpression=Attr('uid').gte('sensmitter_1')
)
items = response['Items']
json_DF = pd.json_normalize(items)
#----------------------------------------------#

#Converting the Datatypes
json_DF['uid'] =json_DF['uid'].astype(str)
json_DF['timestamp'] = pd.to_numeric(json_DF['timestamp'])
json_DF['payload.data.temperature'] = pd.to_numeric(json_DF['payload.data.temperature'])
json_DF['payload.data.humidity'] = pd.to_numeric(json_DF['payload.data.humidity'])
json_DF['payload.data.pressure'] = pd.to_numeric(json_DF['payload.data.pressure'])
json_DF['payload.data.light_level'] = pd.to_numeric(json_DF['payload.data.light_level'])
#----------------------------------------------#

#Formatting the dataset
df = pd.DataFrame(json_DF, columns = ['TimeStamp','Humidity', 'Pressure','Light_Level','Temperature'])
df['TimeStamp'] = json_DF['timestamp']
df['Humidity'] = json_DF['payload.data.humidity']
df['Pressure'] = json_DF['payload.data.pressure']
df['Light_Level'] = json_DF['payload.data.light_level']
df['Temperature'] = json_DF['payload.data.temperature']
#----------------------------------------------#

#Dataset
#------#
st.write("""
    *Sensimitter_01*
    """)
st.write(df)

#Plotting the data
#----------------#
import matplotlib.pyplot as plt
fig = plt.figure()
x = df['TimeStamp']
y = df[metrics]
plt.plot(x,y)
plt.grid(True)
st.plotly_chart(fig)
#------------------------------#

#---------------------#
#Finding the Anomalies
#---------------------#

if Model=="Isolation_Forest":
    df['anomaly'] = pd.Series(clf.predict(df))
    df['anomaly']=df['anomaly'].map({1:0,-1:1})
    Anomaly = df.loc[df.anomaly==1]
    st.write("""*Anomalous Points*""")
    st.write(df['anomaly'].value_counts())
else:
    df.drop('TimeStamp',axis=1)
    scaler = StandardScaler()
    data = pd.DataFrame(scaler.fit_transform(df))
    def CreateDataList(data):
        humidityList = list()
        pressureList = list()
        lightList = list()
        tempList = list()
        for x in range(len(data.index)):
            string = data.loc[x,:]
            value = float(string["payload"]["m"]["data"]["m"]["humidity"]["s"])
            humidityList.append(value)
            value = float(string["payload"]["m"]["data"]["m"]["pressure"]["s"])
            pressureList.append(value)
            value = float(string["payload"]["m"]["data"]["m"]["temperature"]["s"])
            tempList.append(value)
            value = float(string["payload"]["m"]["data"]["m"]["light_level"]["s"])
            lightList.append(value)
            dict = {"temp": tempList,"humidity": humidityList,"Pressure": pressureList,"Light": lightList}
            dataList = pd.DataFrame(dict)
        return dataList
    clustering = DBSCAN(eps=0.3, min_samples=10,n_jobs=6).fit(data)
    labels = clustering.labels_
    df['anomaly']=labels
    Anomaly = df.loc[df.anomaly==-1] 
    st.write("""*Anomalous Points*""")
    count = 0
    for x in clustering.labels_:
        if(x == -1):
            count += 1
    st.write(count)
  
#----------------------------------------------#

#Visualizing the Anomalies
#-------------------------#
st.subheader("Anomalous Datapoints")
import matplotlib.pyplot as plt
fig = plt.figure()
x = Anomaly['TimeStamp']
y = Anomaly[metrics]
plt.scatter(x,y,color='r')
x = df['TimeStamp']
y = df[metrics]
plt.plot(x,y)
plt.grid(True)
st.plotly_chart(fig)