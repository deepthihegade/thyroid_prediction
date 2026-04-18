# 🩺 Thyroid Disease Prediction with Lifestyle Recommendation

A machine learning web app that predicts thyroid disorders and provides personalized lifestyle recommendations.

## Predicts 3 conditions:
- 🟢 Normal
- 🔵 Hypothyroidism
- 🔴 Hyperthyroidism

## Models Used:
- SVM (RBF Kernel)
- XGBoost
- Stacking (XGB + RF + SVM → LR)
- Bagging (100 Decision Trees)

## Run the app:
```bash
pip install -r requirements.txt
streamlit run myapp.py
```

## Tech Stack:
Python, Scikit-learn, XGBoost, Streamlit, Pandas, NumPy

> ⚠️ For educational purposes only. Always consult a doctor.
