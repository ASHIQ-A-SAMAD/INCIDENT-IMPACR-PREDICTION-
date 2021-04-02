import pandas as pd
import streamlit as st
import datetime
from sklearn import preprocessing 
from sklearn.preprocessing import LabelEncoder
import numpy as np

# Create encoder
le = LabelEncoder()
label_encoder = preprocessing.LabelEncoder()
st.title('INCIDENT IMPACT PREDICTION')


st.write("""
# impact Prediction App

""")

st.sidebar.header('User Input Features')

st.sidebar.markdown("""
""")


#input_df_test = pd.read_csv(uploaded_file)



#label_encoder = preprocessing.LabelEncoder()

#df = pd.concat([(X_test.iloc[:,1:8]),(y_test.iloc[:,1])])




# Collects user input features into dataframe
uploaded_file = st.sidebar.file_uploader("Upload your input CSV file", type=["csv"])


if uploaded_file is not None:
    input_df_test = pd.read_csv(uploaded_file)
    input_df_test1=input_df_test[['ID_status', 'active','count_reassign','count_updated','opened_time','location','user_symptom']]
else:

    def user_input_features():
        count_reassign =st.sidebar.number_input("Count Reassign",value=0)
        st.sidebar.text(" format location ***")
        location = st.sidebar.text_input('LOCATION', 'location ')
        ID_status=st.sidebar.selectbox("ID status", ('New', 'Resolved', 'Closed', 'Active', 'Awaiting User Info','Awaiting Problem'))
        active=st.sidebar.selectbox("Active", ('TRUE', 'FALSE'))
        count_updated = st.sidebar.number_input('Count Updated', value=0)
        st.sidebar.text(" format Symptom ***")
        user_symptom = st.sidebar.text_input('User Symptom', 'Symptom ')
        #date = st.sidebar.date_input('start date', datetime.datetime (2021,1,1))
        st.sidebar.text("format: d/m/year hr:minutes")
        opened_time = st.sidebar.text_input('opened_time')
        data= {'ID_status':ID_status,
               'active':active,
               'count_reassign':count_reassign,
               'count_updated':count_updated,
               'opened_time':opened_time,  
               'location':location,
               'user_symptom':user_symptom,
          
              }
        features=pd.DataFrame(data,index=[0])
        return features
    input_df_test = user_input_features()
    input_df_test1 = pd.DataFrame(data=input_df_test)


index = input_df_test1.index
number_of_rows = len(index)


input_df_raw = pd.read_csv('X_test_file3.csv')
input_df=input_df_raw[['ID_status', 'active','count_reassign','count_updated','opened_time','location','user_symptom']]
df1 = pd.concat([input_df_test1,input_df],axis=0)


df = df1.apply(lambda col: le.fit_transform(col.astype(str)), axis=0, result_type='expand')


#df=df1.apply(label_encoder.fit_transform)

if number_of_rows == 1:
    st.write(input_df_test)
    df_raw=df[1:]
    df_user_ip=df[:1]
else:
    st.write("NO.of rows in uploaded file:",number_of_rows)
    st.write("""# USER INPUT DATA""")
    #st.write(input_df_test)
    df_raw=df[number_of_rows:]
    df_user_ip=df[:number_of_rows]
    st.write(input_df_test1)
    
o_p_raw= label_encoder.fit_transform(input_df_raw['impact'])
o_p = o_p_raw
i_o = df_raw

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split( i_o, o_p, test_size=0.4, random_state=13)




from sklearn.ensemble import GradientBoostingClassifier

model_gradiant = GradientBoostingClassifier()
model_gradiant.fit(X_train, y_train)



prediction = model_gradiant.predict(df_user_ip)
prediction_proba = model_gradiant.predict_proba(df_user_ip)

#st.subheader('Prediction')
#st.write(input_df_test)
#st.write(prediction)



impact_status = np.array(['high','medium','low'])

st.subheader('Prediction Probability')
st.write(prediction_proba)


st.subheader('Prediction output')
st.write(impact_status[prediction])
    



