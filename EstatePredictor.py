import streamlit as st
from PIL import Image
import pandas as pd
import joblib
import numpy as np
data1 = pd.read_csv("Delhi House.csv")
data2= pd.read_csv("processed_data.csv")
x = pd.read_csv("x.csv")
model = joblib.load("DecisionTreeReg.pkl")
def predict_price(location,BHK,Parking,Area,Type,Transaction):
    type_index = np.where(x.columns==Type)[0][0]
    loc_index = np.where(x.columns==location)[0][0]
    trans_index = np.where(x.columns==Transaction)[0][0]
    X = np.zeros(len(x.columns))
    if type_index>=0:
        X[type_index]= 1
    if loc_index>=0:
        X[loc_index]= 1
    if trans_index>=0:
        X[trans_index]= 1
    X[0]= Area
    X[1]=BHK
    X[2]=Parking
    return model.predict([X])[0]

st.title("Real Estate Price Predictor for Delhi")
nav = st.sidebar.radio("Navigation",["Home","Prediction"])
if nav=="Home":
    img = Image.open("real.jpg")
    st.image(img, width=800)
    if st.checkbox("Show Dataframe Used to Train the Model"):
        st.dataframe(data1)
        st.download_button(
   "Press to Download",
   data1.to_csv(index=False).encode("utf-8"),
   "estate.csv",
   "text/csv",
   key='download-csv'
)
if nav=="Prediction":
    column = []
    for i in (data2.columns):
          column.append(i)
    column.append("Other")
    loc = st.selectbox("Choose Your Location", column[9:])
    BHK= st.slider("Enter BHK", 1,10,1)
    parking= st.number_input("Enter Number of Parking Area", min_value=1, max_value=5)
    area = st.number_input("Enter Area in Square Ft.", min_value=50, max_value=25000)
    type =st.radio("Type of Flat",["Apartment","Builder_Floor"])
    trans = st.radio("Type of Transaction",["Resale","New_Property"])
    if st.button("Predict"):
        price = predict_price(loc,BHK,parking,area,type,trans)
        st.write("## Rs. ",price )


    



    
    


