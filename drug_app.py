import streamlit as st
import pandas as pd
import numpy as np
import re
 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
 
st.title("💊 Drug Side Effect Prediction System")
 
# Upload CSV
uploaded_file = st.file_uploader("Upload your dataset", type=["csv"])
 
if uploaded_file is not None:
 
    df = pd.read_csv(r"C:\Users\SuryasimhaMoturi\OneDrive - Health Catalyst\Practice\AI_data\assignments\drug_interaction_project\drug_side_effect_dataset_real_100k.csv")
 
    st.subheader("Full Dataset")
    st.dataframe(df)
 
    # Select columns
    df = df[['age', 'gender', 'condition', 'drugA', 'drugB']]
    df.columns = ['age', 'gender', 'condition', 'druga', 'drugb']
 
    # Clean text
    def clean_text(x):
        x = str(x).lower()
        x = re.sub(r'[^a-z0-9 ]', '', x)
        x = re.sub(r'\s+', ' ', x).strip()
        return x
 
    for col in ['condition', 'druga', 'drugb']:
        df[col] = df[col].apply(clean_text)
 
    # Create side effects
    df['side_effects'] = np.where(
        ((df['age'] > 50) & (df['druga'] != df['drugb'])) |
        ((df['condition'].str.contains('diabetes|bp|heart'))),
        1, 0
    )
 
    # Encode gender
    df['gender'] = df['gender'].str.lower().map({'female': 0, 'male': 1})
 
    # Label encoding
    le_condition = LabelEncoder()
    le_druga = LabelEncoder()
    le_drugb = LabelEncoder()
 
    df['condition'] = le_condition.fit_transform(df['condition'])
    df['druga'] = le_druga.fit_transform(df['druga'])
    df['drugb'] = le_drugb.fit_transform(df['drugb'])
 
    # Feature engineering
    df['drug_interaction'] = df['druga'] * df['drugb']
    df['age_risk'] = df['age'] // 10
 
    # Features
    X = df[['age', 'gender', 'condition', 'druga', 'drugb', 'drug_interaction', 'age_risk']]
    y = df['side_effects']
 
    # Train model
    model = RandomForestClassifier(n_estimators=200, max_depth=10)
    model.fit(X, y)
 
    st.success("Model trained successfully!")
 
    # =========================
    # Prediction Section
    # =========================
 
    st.subheader("🔍 Predict Side Effects")
 
    age = st.number_input("Age", 0, 100)
    gender = st.selectbox("Gender", ["male", "female"])
    condition = st.text_input("Condition (e.g., diabetes)")
    druga = st.text_input("Drug A")
    drugb = st.text_input("Drug B")
 
    if st.button("Predict"):
 
        # Clean inputs
        condition_clean = clean_text(condition)
        druga_clean = clean_text(druga)
        drugb_clean = clean_text(drugb)
 
        try:
            input_data = pd.DataFrame({
                'age': [age],
                'gender': [1 if gender == 'male' else 0],
                'condition': [le_condition.transform([condition_clean])[0]],
                'druga': [le_druga.transform([druga_clean])[0]],
                'drugb': [le_drugb.transform([drugb_clean])[0]],
            })
 
            input_data['drug_interaction'] = input_data['druga'] * input_data['drugb']
            input_data['age_risk'] = input_data['age'] // 10
 
            prediction = model.predict(input_data)[0]
 
            if prediction == 1:
                st.error("⚠️ Side Effects Likely")
            else:
                st.success("✅ Safe Combination")
 
        except:
            st.warning("⚠️ Enter values present in dataset")