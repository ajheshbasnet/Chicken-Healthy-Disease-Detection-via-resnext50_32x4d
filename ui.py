# save this as streamlit_ui.py
import streamlit as st
import requests

st.title("🐔 Poultry Disease Classifier (FastAPI)")

API_URL = "http://127.0.0.1:8000/upload-image"  # your FastAPI endpoint

# File uploader
uploaded_file = st.file_uploader("Upload an image of the chicken", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
    
    # Send image to FastAPI
    files = {"file": (uploaded_file.name, uploaded_file, "image/jpeg")}
    
    with st.spinner("Classifying..."):
        response = requests.post(API_URL, files=files)
    
    if response.status_code == 200:
        data = response.json()
        st.success(f"Prediction: {data['Status']}")
        st.info(f"Confidence: {data['Confidence Score']:.2f}%")
    else:
        st.error("Error occurred. Please check the API.")
