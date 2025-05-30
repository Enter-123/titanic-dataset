import pandas as pd
import numpy as np
import streamlit as st
import joblib 


scaler = joblib.load('knn_model.joblib')
classifier = joblib.load('classifier.joblib')

st.title('Titanic Dataset Prediction')

pclass = st.selectbox("Passenger Class (Pclass)",[1,2,3])
gender = st.selectbox("Gender",['male','female'])
age = st.number_input("Age :",min_value=1,max_value=100,value=28,step=1)
sibsp = st.number_input("Number of Sibling  : ",min_value=0,max_value=20,value=2,step=1)	
parch = st.number_input("Number of Parent  : ",min_value=0,max_value=20,value=2,step=1)
fare = st.number_input("Enter Fare Price : ",min_value=0,max_value=500,value=50,step=1)
embarked = st.selectbox("Enter the place of Embarked : ",['S','C','Q'])


gender_map = {'male':0,'female':1}
embarked_map = {'S':0,'C':1,'Q':2}


if st.button("Predict Survival"):
    input_data = pd.DataFrame([{
        'Pclass':pclass,
        'gender':gender_map[gender],
        'Age':age,
        'SibSp':sibsp,
        'Parch':parch,
        'Fare':fare,
        'Embarked':embarked_map[embarked],
    }])

    input_data = scaler.transform(input_data)
    prediction = classifier.predict(input_data)[0]
    probability = np.max(classifier.predict_proba(input_data)[0])

    if prediction == 1:
        st.success(f"Prediction will survived with the probability of {probability:.2%}")
    else:
        st.error(f"Prediction will NOT survived with the probability of {probability:.2%}")


