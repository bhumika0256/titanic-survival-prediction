import streamlit as st
import joblib
import numpy as np

st.set_page_config(page_title="Titanic Survival Predictor", page_icon="🚢")

st.title("🚢 Titanic Survival Predictor")
st.write("Would YOU have survived the Titanic? Enter passenger details to find out!")

try:
    bundle = joblib.load('titanic_model.pkl')
    model = bundle['model']
except:
    st.error("Run `python main.py` first!")
    st.stop()

st.success(f"Model: {bundle['model_name']} | Accuracy: {bundle['accuracy']:.2%}")

# User inputs
col1, col2 = st.columns(2)

with col1:
    pclass = st.selectbox("Ticket Class", [1, 2, 3], 
                          format_func=lambda x: f"Class {x} ({'Upper' if x==1 else 'Middle' if x==2 else 'Lower'})")
    sex = st.selectbox("Gender", ["Male", "Female"])
    age = st.slider("Age", 1, 80, 25)
    sibsp = st.number_input("Siblings/Spouses aboard", 0, 8, 0)

with col2:
    parch = st.number_input("Parents/Children aboard", 0, 6, 0)
    fare = st.number_input("Fare paid (£)", 0.0, 512.0, 32.0)
    embarked = st.selectbox("Port of Embarkation", 
                            ["Cherbourg", "Queenstown", "Southampton"])

# Encode inputs
sex_encoded = 1 if sex == "Male" else 0
embarked_map = {"Cherbourg": 0, "Queenstown": 1, "Southampton": 2}
embarked_encoded = embarked_map[embarked]

if st.button("🔮 Predict Survival"):
    features = np.array([[pclass, sex_encoded, age, sibsp, parch, fare, embarked_encoded]])
    prediction = model.predict(features)[0]
    probability = model.predict_proba(features)[0]
    
    if prediction == 1:
        st.balloons()
        st.success(f"🎉 YOU SURVIVED! (Confidence: {probability[1]*100:.1f}%)")
    else:
        st.error(f"💀 Sorry, you did not survive. (Death probability: {probability[0]*100:.1f}%)")
    
    st.write("---")
    st.subheader("Key Survival Factors from Titanic Data:")
    st.write("""
    - 👩 **Women** had much higher survival rates than men
    - 💰 **1st Class** passengers survived more than 3rd class
    - 👶 **Children** had higher survival rates
    - 🏷️ **Higher fare** = better survival chances
    """)

# Show charts
st.write("---")
st.subheader("Model Analysis")
try:
    st.image("eda_charts.png", caption="Exploratory Data Analysis")
    st.image("confusion_matrix.png", caption="Model Performance")
    st.image("feature_importance.png", caption="What factors matter most?")
except:
    pass