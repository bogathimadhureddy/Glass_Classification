import os 
# changing the working directory.
os.chdir(r'C:\Users\madhu\OneDrive\Desktop\360 DigiTMG\DataScience\ASSIGNMENTS SOLVED BY ME\KNN Model')
import streamlit as st
import re
import pandas as pd
import numpy as np
import copy
import joblib
import pickle
from sqlalchemy import create_engine

engine = create_engine('mysql+pymysql://{}:{}@localhost/{}'.format('root','madhu123','salary_db'))

def sqrt_trans(x):
    return np.power(x,1/5)

model = pickle.load(open('knn.pkl', 'rb'))
preprocess_pipeline = joblib.load('preprocess_pipeline.pkl')
winsorizer = joblib.load('winsorizer.pkl')


form = st.form(key='my_form')
f = form.file_uploader('Upload A File') 
submit_button = form.form_submit_button(label='Submit')
if submit_button:
     if f is not None:
         try:
            glass = pd.read_csv(f)
         except:
                try:
                    glass = pd.read_excel(f)
                except:      
                    glass = pd.DataFrame(f)

         else:
            st.sidebar.warning("You need to upload a csv or excel file.")
        
     glass = glass.iloc[:,:-1] # When new data comes for that data there is no output varaible so no need this line.
    
     glass_data = glass.copy()

     outliers_treatment = ['RI', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Fe']

     glass[outliers_treatment] = winsorizer.transform(glass.loc[:,outliers_treatment])

     glass = pd.DataFrame(preprocess_pipeline.transform(glass))
            
     test_pred_lap = pd.DataFrame(model.predict(glass))
     test_pred_lap.columns = ["Type_Glass"]

     final = pd.concat([glass_data, test_pred_lap], axis = 1)
     
     final.to_sql('glass_predict', con = engine, if_exists = 'replace', index= False)
     
     st.table(final)
    