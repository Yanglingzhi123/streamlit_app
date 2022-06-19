import streamlit as st
import pandas as pd
import joblib

# Title
st.header("A machine learning-based classifier for PAH re-classification")

# Input bar 1
PDHB = st.number_input("Enter PDHB")

# Input bar 2
MTF1 = st.number_input("Enter MTF1")

LIPT1 = st.number_input("Enter LIPT1")


LIAS = st.number_input("Enter LIAS")
FDX1 = st.number_input("Enter FDX1")
DLD = st.number_input("Enter DLD")
# Input bar 2
DLAT = st.number_input("Enter DLAT")
CDKN2A = st.number_input("Enter CDKN2A")

# If button is pressed
if st.button("Submit"):
    
    # Unpickle classifier
    rfc = joblib.load("D:\肺动脉高压生信\streamlit\clf.pkl")
    
    # Store inputs into dataframe
    X = pd.DataFrame([[PDHB,MTF1,LIPT1,LIAS,FDX1,DLD,DLAT,CDKN2A]], 
                     columns = ["PDHB","MTF1","LIPT1","LIAS","FDX1","DLD","DLAT","CDKN2A"])
   # X = X.replace(["Brown", "Blue"], [1, 0])
    
    # Get prediction
    prediction = rfc.predict(X)
    
    # Output prediction
    st.text(f"This instance is a {prediction}")