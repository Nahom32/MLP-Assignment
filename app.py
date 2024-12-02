import pickle
import streamlit as st

@st.cache
def load_model():
    with open('breast_model.pkl', 'rb') as file:
        model = pickle.load(file)
    return model

model = load_model()
st.title("Model Deployment")
user_input = st.text_input("Enter some input:")
if user_input:
    prediction = model.predict([user_input])
    st.write(f"Prediction: {prediction}")
