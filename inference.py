import streamlit as st
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

model = joblib.load("xgb_class.pkl")

class loanclassmodel:
    def __init__(self, model):
        self.model = model
        
    def make_prediction(self, input_data: pd.DataFrame):
        pred = self.model.predict(input_data)[0]
        prob = self.model.predict_proba(input_data)[0][1]
        return pred, prob


    
def main():
    st.title("Loan Approval Prediction Model")
    st.subheader("by Nicholas Ananda Heryanto - 2702213695")
    
    person_age = st.slider("Usia", 20, 70)
    person_income = st.number_input("Pendapatan Tahunan", min_value=8000,max_value=5500000)
    person_emp_exp = st.slider("Pengalaman Kerja (Tahun)", 0, 55, 5)
    loan_amount = st.number_input("Jumlah Pinjaman", min_value=500,max_value=35000)
    loan_int_rate = st.slider("Suku Bunga (%)", 5.0, 20.0, 1.0)
    credit_score = st.slider("Skor Kredit", 400, 850)
    cb_person_cred_hist_length = st.slider("Durasi Kredit (Tahun)", 2, 30, 2)

    loan_percent_income = loan_amount / person_income if person_income > 0 else 0.0
    previous_loan_defaults_on_file = st.selectbox("Ada tunggakan sebelumnya?", ["No", "Yes"])
    person_gender = st.selectbox("Gender", ["male", "female"])
    person_education = st.selectbox("Pendidikan", ["Master", "High School", "Bachelor", "Associate","Doctorate"])
    person_home_ownership = st.selectbox("Kepemilikan Rumah", ["RENT", "OWN","MORTGAGE","OTHER"])
    loan_intent = st.selectbox("Tujuan Pinjaman", ["PERSONAL","EDUCATION","MEDICAL","VENTURE" "HOMEIMPROVEMENT","DEBTCONSOLIDATION"])

    input_data = pd.DataFrame({
    'person_age': [person_age],
    'person_income': [person_income],
    'person_emp_exp': [person_emp_exp],
    'loan_amnt': [loan_amount],
    'loan_int_rate': [loan_int_rate],
    'loan_percent_income': [loan_percent_income],
    'cb_person_cred_hist_length': [cb_person_cred_hist_length],
    'credit_score': [credit_score],
    'previous_loan_defaults_on_file': [1 if previous_loan_defaults_on_file == "Yes" else 0],
    'person_gender': [1 if person_gender == "male" else 0],
    
    'person_education_Associate': [1 if person_education == 'Associate' else 0],
    'person_education_Bachelor': [1 if person_education == 'Bachelor' else 0],
    'person_education_Doctorate': [1 if person_education == 'Doctorate' else 0],
    'person_education_High School': [1 if person_education == 'High School' else 0],
    'person_education_Master': [1 if person_education == 'Master' else 0],

    'person_home_ownership_MORTGAGE': [1 if person_home_ownership == 'MORTGAGE' else 0],
    'person_home_ownership_OTHER': [1 if person_home_ownership == 'OTHER' else 0],
    'person_home_ownership_OWN': [1 if person_home_ownership == 'OWN' else 0],
    'person_home_ownership_RENT': [1 if person_home_ownership == 'RENT' else 0],

    'loan_intent_DEBTCONSOLIDATION': [1 if loan_intent == 'DEBTCONSOLIDATION' else 0],
    'loan_intent_EDUCATION': [1 if loan_intent == 'EDUCATION' else 0],
    'loan_intent_HOMEIMPROVEMENT': [1 if loan_intent == 'HOMEIMPROVEMENT' else 0],
    'loan_intent_MEDICAL': [1 if loan_intent == 'MEDICAL' else 0],
    'loan_intent_PERSONAL': [1 if loan_intent == 'PERSONAL' else 0],
    'loan_intent_VENTURE': [1 if loan_intent == 'VENTURE' else 0]
})
    scaler = MinMaxScaler()
    numerical_cols = [
        'person_age', 'person_emp_exp', 'person_income', 'loan_amnt', 
        'loan_int_rate', 'loan_percent_income', 'cb_person_cred_hist_length', 
        'credit_score'
    ]
    input_data[numerical_cols] = scaler.fit_transform(input_data[numerical_cols])
    
    input_data = input_data[model.feature_names_in_]
    
    predictor = loanclassmodel(model)

    if st.button("Prediksi"):
        prediction, probability = predictor.make_prediction(input_data)
        if prediction == 1:
            st.success(f"Disetujui (Probabilitas: {probability:.2f})")
        else:
            st.error(f"Ditolak (Probabilitas: {probability:.2f})")

            
            

    
if __name__ == "__main__":
    main()
            
