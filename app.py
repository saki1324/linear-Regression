import streamlit as st
import pandas as pd
import numpy as np
import pickle

clf=pickle.load(open("mymodel.pkl","rb"))

def prediction(data):
    clf=pickle.load(open("mymodel.pkl","rb"))
    return clf.prediction(data)


st.title("Advertising Spends Prediction using Machine Learning")
st.markdown("This Model Identify total spends on advertising")

st.header("Advertising Spend on various Media")
coll,col2=st.columns(2)

with coll:
    st.text("TV")
    tv=st.slider("Adver. Spends on TV",1.0,10000.0,0.5)
    st.text("Radio")
    rd=st.slider("Adver. Spends on Radio",1.0,10000.0,0.5)
    st.text("NewsPaper")
    newspaper=st.slider("Adver. Spends on NewsPaper",1.0,10000.0,0.5)
    
    
st.text('')
if st.button("Sales Prediction"):
    result=clf.prediction(np.array([[tv,rd,newspaper]]))
    st.text(result[0])
                     
st.markdown("Developed By Jainee Patel at NIELIT Daman")                          
                          
    