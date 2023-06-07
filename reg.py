
import numpy as np
import pickle
import pandas as pd
import streamlit as st 

from PIL import Image


pickle_in = open("classifier.pkl","rb")
classifier=pickle.load(pickle_in)


def welcome():
    return "Welcome All"


def predict_note_authentication(variance,skewness,curtosis,entropy):
    
    """Let's Authenticate the Banks Note 
    This is using docstrings for specifications.
    ---
    parameters:  
      - name: variance
        in: query
        type: number
        required: true
      - name: skewness
        in: query
        type: number
        required: true
      - name: curtosis
        in: query
        type: number
        required: true
      - name: entropy
        in: query
        type: number
        required: true
    responses:
        200:
            description: The output values
        
    """
   
    prediction=classifier.predict([[variance,skewness,curtosis,entropy]])
    print(prediction)
    return prediction



def main():
    st.title("Regression Project")
    html_temp = """
    <div style="background-color:green;padding:10px">
    <h2 style="color:yellow;text-align:center;">Store Sales -- Time Series Forecasting </h2>
    </div>
    """
    st.markdown(html_temp,unsafe_allow_html=True)
    variance = st.text_input("Variance","Number Only")
    skewness = st.text_input("skewness","Number Here")
    curtosis = st.text_input("curtosis","Type Here")
    entropy = st.text_input("entropy","Type Here")
    result=""
    if st.button("Predict"):
        result=predict_note_authentication(variance,skewness,curtosis,entropy)
    st.success('The answer is {}'.format(result))
    if st.button("About"):
        st.text("This is the work of the group Sydney")
 

if __name__=='__main__':
    main()
    
    
    