import os
import pickle
import numpy as np
import pandas as pd
import streamlit as st

#%% STATICS
MODEL_PICKLE_PATH=os.path.join(os.getcwd(),'model','model_heart.pkl')

#%% MODEL IMPORT
with open(MODEL_PICKLE_PATH,'rb') as file:
    model=pickle.load(file)
    
#%% APP CODE
# Input features:
# 'thalachh','oldpeak','caa','cp','thall'

st.title('Cardiovascular Disease **Predictor**')

st.image('img.jpg', 
          caption='Model of human heart. Source from https://www.uwa.edu.au/Research/Cardiovascular-Disease-Epidemiology')

with st.form("my_form"):
    st.info("This app predicts your risk of getting cardiovascular disease (CVD).")

    cp = st.selectbox('Chest pain type (0-Asymptomatic, 1-Typical angina, 2-Atypical angina, 3-Non-anginal pain):',
                      np.arange(0,4,1,dtype=int))
    
    thall = st.selectbox('Thallium Stress Test result (1-Fixed defect, 2-Normal, 3-Reversable defect):',
                           np.arange(1,4,1,dtype=int))
    
    col1,col2,col3 = st.columns([1,1,1])
    
    thalachh = col1.selectbox('Maximum heart rate:',
                              np.arange(20,201,1,dtype=int))
    caa = col2.selectbox('Number of major vessels (0-3):',
                         np.arange(0,4,1,dtype=int))

    oldpeak = col3.number_input('ST depression induced by exercise relative to rest:')
    
    
# Every form must have a submit button.
    submitted = st.form_submit_button("Submit")
    if submitted:
        X_new = [thalachh,oldpeak,caa,cp,thall]
        outcome = model.predict(np.expand_dims(X_new,axis=0))
        if outcome == 1:
            st.warning('Your health is at risk, please make an appointment with your doctor ASAP!')
            # st.snow()
        else:
            st.success('You have a low risk to get cardiovascular disease, take care!')
            # st.balloons()

st.caption('This model is built based on Heart Attack Analysis & Prediction Dataset on **kaggle**, source link https://www.kaggle.com/datasets/rashikrahmanpritom/heart-attack-analysis-prediction-dataset')
