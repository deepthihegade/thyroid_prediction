import streamlit as st
import numpy as np

st.set_page_config(page_title="Thyroid Disease Predictor", page_icon="🩺")

st.title("🩺 Thyroid Disease Prediction System")
st.markdown("Enter the patient's blood test values below")

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", min_value=1, max_value=100, value=30)
    tsh = st.number_input("TSH Level (mIU/L)", min_value=0.0, value=2.5)
    t3 = st.number_input("T3 Level (nmol/L)", min_value=0.0, value=1.8)

with col2:
    gender = st.selectbox("Gender", ["Female", "Male"])
    t4 = st.number_input("T4 Level (nmol/L)", min_value=0.0, value=100.0)
    fti = st.number_input("FTI Level", min_value=0.0, value=100.0)

st.divider()

if st.button("🔍 Predict Thyroid Condition", use_container_width=True):

    # ---- THIS BLOCK WILL BE REPLACED ONCE YOU GET .pkl FROM CREW ----
    # Dummy prediction logic for now (just to test UI)
    if tsh < 0.4:
        prediction = "Hyperthyroidism"
    elif tsh > 4.0:
        prediction = "Hypothyroidism"
    else:
        prediction = "Normal"
    # -----------------------------------------------------------------

    if prediction == "Normal":
        st.success("✅ Prediction: Normal Thyroid Function")
        st.markdown("### 💚 Lifestyle Recommendations")
        st.markdown("""
        - Maintain a balanced diet rich in iodine (seafood, dairy)
        - Exercise regularly — at least 30 mins a day
        - Get thyroid levels checked annually
        - Avoid excessive stress
        """)

    elif prediction == "Hypothyroidism":
        st.warning("⚠️ Prediction: Hypothyroidism Detected")
        st.markdown("### 💛 Lifestyle Recommendations")
        st.markdown("""
        - Increase iodine-rich foods: eggs, fish, dairy
        - Avoid raw cruciferous vegetables (cabbage, broccoli)
        - Maintain regular exercise to manage weight
        - Take prescribed medications consistently
        - Get adequate sleep (7-8 hours)
        - Regular follow-up blood tests every 6 months
        """)

    elif prediction == "Hyperthyroidism":
        st.error("🚨 Prediction: Hyperthyroidism Detected")
        st.markdown("### ❤️ Lifestyle Recommendations")
        st.markdown("""
        - Reduce iodine-rich foods
        - Avoid caffeine and stimulants
        - Manage stress through yoga or meditation
        - Eat calcium-rich foods to protect bones
        - Avoid intense exercise during active phase
        - Consult doctor immediately for treatment
        """)

    st.divider()
    st.caption("⚠️ This tool is for educational purposes only. Always consult a doctor.")