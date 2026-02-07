import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download
import joblib

# Download and load the model
model_path = hf_hub_download(repo_id="mkrish2025/Tourism-Customer-Prediction", filename="best_tourism_prediction_model_v1.joblib")
model = joblib.load(model_path)

# Download preprocessor model
preprocessor_path = hf_hub_download(repo_id="mkrish2025/Tourism-Customer-Prediction", filename="preprocessor.joblib")
preprocessor = joblib.load(preprocessor_path)

# Streamlit UI for Machine Failure Prediction
st.title("Customer Tour Package Prediction App")
st.write("""
This application predicts the likelihood of customers purchasing the Wellness Tourism Package.
Please enter the customer details below to get a prediction.
""")

Age = st.number_input("Age", min_value=18, max_value=100, value=30)
MonthlyIncome = st.number_input("Monthly Income", min_value=1000, max_value=100000, value=5000)
TypeofContact = st.selectbox("Type of Contact", ["Self Enquiry", "Company Invited"])
CityTier = st.number_input("City Tier", min_value=1, max_value=3, value=1)
Occupation = st.selectbox("Occupation", ["Salaried", "Small Business", "Large Business"])
Gender = st.selectbox("Gender", ["Male", "Female"])
ProductPitched = st.selectbox("Product Pitched", ["Basic", "Deluxe", "Standard", "Super Deluxe", "King"])
MaritalStatus = st.selectbox("Marital Status", ["Single", "Married", "Unmarried", "Divorced"])
Passport = st.number_input("Passport (0=No, 1=Yes)", min_value=0, max_value=1, value=0, step=1)
OwnCar = st.number_input("Own Car (0=No, 1=Yes)", min_value=0, max_value=1, value=0, step=1)
Designation = st.selectbox("Designation", ["Executive", "Manager", "Senior Manager", "AVP", "VP"])
PreferredPropertyStar = st.number_input("Preferred PropertyStar", min_value=3, max_value=5, value=3, step=1)
NumberOfTrips = st.number_input("Number Of Trips", min_value=0, max_value=100, value=2, step=1)
TotalVisiting = st.number_input("Total Visting (includes child below 5)", min_value=1, max_value=10, value=2, step=1)


# Assemble input into DataFrame
input_data = pd.DataFrame([{
    "Age": Age,
    "MonthlyIncome": MonthlyIncome,
    "Designation": Designation,
    "OwnCar": OwnCar,
    "Passport": Passport,
    "CityTier": CityTier,
    "MaritalStatus": MaritalStatus,
    "ProductPitched": ProductPitched,
    "Gender": Gender,
    "Occupation": Occupation,
    "TypeofContact": TypeofContact,
	  "PreferredPropertyStar": PreferredPropertyStar,
	  "NumberOfTrips": NumberOfTrips,
	  "TotalVisiting": TotalVisiting
	
}])

st.subheader("Raw Input Data")
st.dataframe(input_data)

# Prediction 	
if st.button("Predict"):
    try:
        

        prediction = model.predict(input_data)[0]
        prediction_proba = model.predict_proba(input_data)[0][1]

        if prediction == 1:
            st.success(f"✅ Customer is likely to take the product (Confidence: {prediction_proba:.2f})")
        else:
            st.warning(f"❌ Customer is unlikely to take the product (Confidence: {1 - prediction_proba:.2f})")

    except Exception as e:
        st.error(f"Prediction failed: {e}")	
	
	
