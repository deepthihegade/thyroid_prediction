import streamlit as st
import numpy as np
import pandas as pd
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Thyroid Disease Predictor",
    page_icon="🩺",
    layout="wide"
)

# ── Load Model & Label Encoder ────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

@st.cache_resource
def load_model():
    model = joblib.load(os.path.join(BASE_DIR, 'thyroidmodel.pkl'))
    le    = joblib.load(os.path.join(BASE_DIR, 'labelencoder.pkl'))
    return model, le

model, le = load_model()

# ── Lifestyle Recommendations ─────────────────────────────────────────────────
LIFESTYLE = {
    'Normal': {
        'summary':   'Your thyroid function appears completely normal. Keep up the healthy habits!',
        'diet':      ['Balanced diet with adequate iodine (seafood, dairy, iodised salt).',
                      'Include selenium-rich foods: Brazil nuts, eggs, sunflower seeds.',
                      'Plenty of fruits, vegetables, and whole grains.'],
        'lifestyle': ['30 min exercise per day, 5 days a week.',
                      'Maintain healthy body weight.',
                      'Annual thyroid screening (TSH blood test).',
                      'Stress management: yoga or meditation.'],
        'medical':   ['No thyroid medication needed.',
                      'Routine TSH check once a year.']
    },
    'Hypothyroid': {
        'summary':   'Underactive thyroid detected (TSH likely elevated, T4 low). Please consult a doctor.',
        'diet':      ['Increase iodine: seafood, eggs, dairy, iodised salt.',
                      'Selenium-rich foods: Brazil nuts, tuna, sardines.',
                      'Avoid large amounts of raw goitrogenic foods (cabbage, soy) — cook them.',
                      'Take thyroid medication on empty stomach; wait 30–60 min before eating.'],
        'lifestyle': ['Low-impact cardio (walking, swimming) — helps with fatigue.',
                      'Monitor weight carefully (slowed metabolism).',
                      '7–9 hours of quality sleep.',
                      'Watch for: fatigue, cold intolerance, weight gain, dry skin.'],
        'medical':   ['Consult doctor about Levothyroxine therapy.',
                      'TSH blood test every 3–6 months during treatment.',
                      'Do not stop medication without medical advice.']
    },
    'Hyperthyroid': {
        'summary':   'Overactive thyroid detected (TSH very low, T4/T3 elevated). Please see an endocrinologist.',
        'diet':      ['Reduce iodine-rich foods: limit seaweed, kelp, iodised salt.',
                      'Increase calcium and Vitamin D (thyroid overactivity weakens bones).',
                      'Small, frequent, calorie-dense meals (manage weight loss).',
                      'Avoid caffeine — worsens heart palpitations.'],
        'lifestyle': ['Avoid high-intensity exercise during flare — gentle yoga only.',
                      'Active stress management (hyperthyroidism worsens under stress).',
                      'Watch for: rapid heartbeat, tremors, heat intolerance, anxiety.',
                      'UV-protective sunglasses if Graves disease is suspected.'],
        'medical':   ['Consult endocrinologist immediately.',
                      'Treatment: anti-thyroid drugs, radioactive iodine, or surgery.',
                      'ECG monitoring may be needed.',
                      'TSH + Free T4 every 4–6 weeks during treatment.']
    }
}

def show_lifestyle(label):
    r = LIFESTYLE.get(label, {})
    if not r:
        return
    st.markdown(f"### 📋 Summary\n_{r['summary']}_")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("#### 🥗 Diet")
        for tip in r['diet']:
            st.markdown(f"- {tip}")
    with col2:
        st.markdown("#### 🏃 Lifestyle")
        for tip in r['lifestyle']:
            st.markdown(f"- {tip}")
    with col3:
        st.markdown("#### 💊 Medical")
        for tip in r['medical']:
            st.markdown(f"- {tip}")

# ── UI Header ─────────────────────────────────────────────────────────────────
st.title("🩺 Thyroid Disease Prediction System")
st.markdown("*Enter patient details below. The model will predict thyroid condition and provide personalised recommendations.*")
st.divider()

# ── Input Form ────────────────────────────────────────────────────────────────
st.markdown("### 👤 Patient Basic Information")
col1, col2, col3 = st.columns(3)
with col1:
    age = st.number_input("Age (years)", min_value=1, max_value=100, value=40)
with col2:
    sex = st.selectbox("Biological Sex", ["Female", "Male"])
    sex_val = 0 if sex == "Female" else 1
with col3:
    referral_source = st.selectbox("Referred From", ["SVHC", "SVI", "STMW", "SVHD", "other"])
    ref_map = {'SVHC': 0, 'SVI': 1, 'STMW': 2, 'SVHD': 3, 'other': 4}
    ref_val = ref_map[referral_source]

st.markdown("### 🩸 Blood Test Results")
st.caption("Enter values directly from your thyroid function blood test report. Hover over each field for normal ranges.")
col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    TSH = st.number_input("TSH (mIU/L)", min_value=0.0, value=2.5,
                           help="Thyroid Stimulating Hormone — Normal: 0.4–4.0 mIU/L")
with col2:
    T3 = st.number_input("T3 (nmol/L)", min_value=0.0, value=1.8,
                          help="Triiodothyronine — Normal: 1.2–3.1 nmol/L")
with col3:
    TT4 = st.number_input("Total T4 (nmol/L)", min_value=0.0, value=100.0,
                           help="Total Thyroxine — Normal: 60–140 nmol/L")
with col4:
    T4U = st.number_input("T4 Uptake Ratio", min_value=0.0, value=0.95,
                           help="T4 Resin Uptake — Normal: 0.7–1.1")
with col5:
    FTI = st.number_input("Free T4 Index", min_value=0.0, value=100.0,
                           help="Free Thyroxine Index — Normal: 70–130")

st.markdown("### 🏥 Medical History")
col1, col2, col3, col4 = st.columns(4)
with col1:
    on_thyroxine        = 1 if st.checkbox("Currently on Thyroxine") else 0
    query_on_thyroxine  = 1 if st.checkbox("Possibly on Thyroxine (unconfirmed)") else 0
    on_antithyroid_meds = 1 if st.checkbox("On Anti-Thyroid Medication") else 0
    sick                = 1 if st.checkbox("Currently Sick / Unwell") else 0
with col2:
    pregnant            = 1 if st.checkbox("Currently Pregnant") else 0
    thyroid_surgery     = 1 if st.checkbox("Previous Thyroid Surgery") else 0
    I131_treatment      = 1 if st.checkbox("Had Radioiodine (I-131) Treatment") else 0
    query_hypothyroid   = 1 if st.checkbox("Suspected Hypothyroidism") else 0
with col3:
    query_hyperthyroid  = 1 if st.checkbox("Suspected Hyperthyroidism") else 0
    lithium             = 1 if st.checkbox("On Lithium Medication") else 0
    goitre              = 1 if st.checkbox("Goitre (Enlarged Thyroid)") else 0
    tumor               = 1 if st.checkbox("Thyroid Tumour History") else 0
with col4:
    hypopituitary       = 1 if st.checkbox("Hypopituitary Condition") else 0
    psych               = 1 if st.checkbox("Psychiatric Condition") else 0
    TSH_measured        = 1 if st.checkbox("TSH Test Done", value=True) else 0
    T3_measured         = 1 if st.checkbox("T3 Test Done", value=True) else 0
    TT4_measured        = 1 if st.checkbox("Total T4 Test Done", value=True) else 0
    T4U_measured        = 1 if st.checkbox("T4 Uptake Test Done", value=True) else 0
    FTI_measured        = 1 if st.checkbox("Free T4 Index Test Done", value=True) else 0

st.divider()

# ── Predict Button ────────────────────────────────────────────────────────────
if st.button("🔍 Predict Thyroid Condition", use_container_width=True, type="primary"):
    patient = {
        'age': age, 'sex': sex_val,
        'on_thyroxine': on_thyroxine,
        'query_on_thyroxine': query_on_thyroxine,
        'on_antithyroid_meds': on_antithyroid_meds,
        'sick': sick, 'pregnant': pregnant,
        'thyroid_surgery': thyroid_surgery,
        'I131_treatment': I131_treatment,
        'query_hypothyroid': query_hypothyroid,
        'query_hyperthyroid': query_hyperthyroid,
        'lithium': lithium, 'goitre': goitre,
        'tumor': tumor, 'hypopituitary': hypopituitary,
        'psych': psych,
        'TSH_measured': TSH_measured, 'TSH': TSH,
        'T3_measured': T3_measured,   'T3': T3,
        'TT4_measured': TT4_measured, 'TT4': TT4,
        'T4U_measured': T4U_measured, 'T4U': T4U,
        'FTI_measured': FTI_measured, 'FTI': FTI,
        'referral_source': ref_val
    }

    patient_df = pd.DataFrame([patient])

    pred_enc   = model.predict(patient_df)[0]
    pred_label = le.inverse_transform([pred_enc])[0]
    proba      = model.predict_proba(patient_df)[0]
    confidence = round(max(proba) * 100, 2)

    st.divider()

    # Result box
    if pred_label == 'Normal':
        st.success(f"✅ Prediction: **NORMAL** — Confidence: {confidence}%")
    elif pred_label == 'Hypothyroid':
        st.warning(f"⚠️ Prediction: **HYPOTHYROIDISM** — Confidence: {confidence}%")
    elif pred_label == 'Hyperthyroid':
        st.error(f"🚨 Prediction: **HYPERTHYROIDISM** — Confidence: {confidence}%")

    # Probability bars
    st.markdown("#### 📊 Class Probabilities")
    prob_col1, prob_col2, prob_col3 = st.columns(3)
    for col, cls, p in zip([prob_col1, prob_col2, prob_col3], le.classes_, proba):
        with col:
            st.metric(cls, f"{p*100:.1f}%")
            st.progress(float(p))

    st.divider()
    st.markdown("## 💡 Personalised Recommendations")
    show_lifestyle(pred_label)

    st.divider()
    st.caption("⚠️ This tool is for educational purposes only. Always consult a qualified doctor.")