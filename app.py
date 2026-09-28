
import joblib
import streamlit as st
import pandas as pd

pipeline = joblib.load("./pipeline_numbers.joblib")
st.set_page_config(page_title="House Price Prediction", layout="centered")

def predict(city, property_type, area, bedrooms, bathrooms):
    features ={
        "baths" : bathrooms,
        "bedrooms" : bedrooms,
        "area_marla" : area,
        "property_type" : property_type,
        "city" : city,
    }
    features = pd.DataFrame([features])
    result = pipeline.predict(features)[0]
    return str(result)

st.markdown("## House Price Prediction")

col1, col2 = st.columns(2)
with col1:
    city = st.selectbox("City", ["A", "B", "C"])
with col2:
    property_type = st.selectbox("Property Type", ["A", "B", "C"])

col3, col4, col5 = st.columns(3)
with col3:
    area = st.number_input("Area", min_value=0.0, max_value=10000.0, step=0.05)
with col4:
    bedrooms = st.slider("Bedrooms", min_value=0, max_value=100, step=1)
with col5:
    bathrooms = st.slider("Bathrooms", min_value=0, max_value=100, step=1)

if st.button("Predict", type='primary'):
    price = predict(city, property_type, area, bedrooms, bathrooms)
    st.text_input("Predict Price", value=price, disabled=True)
