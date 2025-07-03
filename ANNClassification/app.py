import streamlit as st
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pandas as pd
import pickle

model = tf.keras.models.load_model('model.h5')

with open('gender_encoder.pkl','rb') as file:
    gender_encoder = pickle.load(file)
with open('ohe_encoder.pkl','rb') as file:
    ohe_encoder = pickle.load(file)
with open('scale.pkl','rb') as file:
    scalar = pickle.load(file)

## streamlit app
st.title('Customer Churn Prediction')

# USER INPUT

Geography = st.selectbox('Geography', ohe_encoder.categories_[0])
Gender = st.selectbox('Geography', gender_encoder.classes_)
Age = st.slider('Age', 18, 92)
Tenure = st.slider('Tenure', 1, 10)
CreditScore = st.number_input('Credit Score')
Balance = st.number_input('Balance')
NumOfProducts = st.slider('Num Of Products', 1, 4)
HasCrCard = st.selectbox('Has Cr Card', [0,1])
IsActiveMember = st.selectbox('Is Active Member', [0,1])
EstimatedSalary = st.number_input('Estimated Salary')

input_data = pd.DataFrame({
    'CreditScore':[CreditScore],
    'Geography':[Geography],
    'Gender':[Gender],
    'Age':[Age],
    'Tenure':[Tenure],
    'Balance': [Balance],
    'NumOfProducts': [NumOfProducts],
    'HasCrCard': [HasCrCard],
    'IsActiveMember': [IsActiveMember],
    'EstimatedSalary': [EstimatedSalary]
})

numerical_cols = input_data.select_dtypes(include=np.number).columns
numerical_cols
input_data[numerical_cols]= scalar.transform(input_data[numerical_cols])

input_data['Gender'] = gender_encoder.transform(input_data['Gender'])

X_encoded = ohe_encoder.transform(input_data[['Geography']].fillna('Unknown'))
X_encoded_df = pd.DataFrame(X_encoded, columns=ohe_encoder.get_feature_names_out(), index=input_data.index)
input_data = input_data.drop(columns=['Geography'])
input_data = pd.concat([input_data, X_encoded_df], axis=1)


# Prediction
prediction=model.predict(input_data)
prediction_proba = prediction[0][0]
st.write(f'Churn Probability: {prediction_proba:.2f}')

if(prediction_proba>0.5):
    st.write("customer is likely to churn")
else:
    st.write("customer is not likely to churn")