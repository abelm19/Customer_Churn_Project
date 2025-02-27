# Import needed libraries
#Import pickle for loading the saved model
import pickle
#Import Streamlit for creating easy webapp
import streamlit as st
#Import pandas for creating dataframe for prediction
import pandas as pd

#Open the loaded model
with open("customer_churn_model.pkl", "rb") as f:
    model_data = pickle.load(f)

with open("encoders.pkl", "rb") as f:
    encoders = pickle.load(f)

with open("scalers.pkl", "rb") as f:
    scalers = pickle.load(f)

loaded_model = model_data["model"]

# Function for Prediction
def churning_prediction(input_data):
   
    input_data_df = pd.DataFrame([input_data])

    # Encode categorical features using saved encoders
    for column, encoder in encoders.items():
        input_data_df[column] = encoder.transform(input_data_df[column])

    # Scale features using saved scalers
    for column, scaler in scalers.items():
        input_data_df[column] = scaler.transform(input_data_df[[column]])

    # Predict
    prediction = loaded_model.predict(input_data_df)
    
    if prediction[0] == 0:
        return 'Not Likely to Churn'
    else:
        return 'Likely to Churn'


def main():
    # Web app title and header
    st.title('Telecom Customer Churn Prediction')
    st.subheader("Please fill in the fields below to make a prediction!")

     # Convert the image to a base64 string
    import base64
    from io import BytesIO
    
    def img_to_base64(img_path):
     with open(img_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode("utf-8")

    # Local image path
    image_path = 'background1.jpg'  # Replace with your image path
    img_base64 = img_to_base64(image_path)

    # Add CSS to set the background image
    st.markdown(
     f"""
     <style>
        .stApp {{
        background-image: url('data:image/jpeg;base64,{img_base64}');
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        }}

       .custom-text-box {{
        background-color: rgba(0, 0, 0, 0.7);  /* Dark background */
        color: white;  /* White text */
        padding: 20px;
        border-radius: 10px;
        font-size: 18px;
       }}
        </style>
        """, 
     unsafe_allow_html=True
)


    # Collecting user input
    gender = st.radio("Select Gender: ", ('Male', 'Female'))
    SeniorCitizen_string = st.radio("Is a SeniorCitizen? : ", ('Yes', 'No'))
    SeniorCitizen = 1 if SeniorCitizen_string == 'Yes' else 0
    Partner = st.radio("Has a Partner? : ", ('Yes', 'No'))
    Dependents = st.radio("Has Dependents? : ", ('Yes', 'No'))
    tenure = st.slider("Select the number of months of subscription", 0, 72, step=1)
    PhoneService = st.radio("Has a PhoneService? : ", ('Yes', 'No'))
    MultipleLines = st.selectbox("Multiple Lines: ", ['No phone service', 'No', 'Yes'])
    InternetService = st.selectbox("Has an InternetService? : ", ['DSL', 'Fiber optic', 'No'])
    OnlineSecurity = st.selectbox("Has an Online Security? : ", ['No internet service', 'No', 'Yes'])
    OnlineBackup = st.selectbox("Has an Online Backup? : ", ['No internet service', 'No', 'Yes'])
    DeviceProtection = st.selectbox("Has a Device Protection? : ", ['No internet service', 'No', 'Yes'])
    TechSupport = st.selectbox("Has a TechSupport? : ", ['No internet service', 'No', 'Yes'])
    StreamingTV = st.selectbox("Has StreamingTV Access? : ", ['No internet service', 'No', 'Yes'])
    StreamingMovies = st.selectbox("Has StreamingMovies Access? : ", ['No internet service', 'No', 'Yes'])
    Contract = st.selectbox("Contract type? : ", ['Month-to-month', 'One year', 'Two Year'])
    PaperlessBilling = st.radio("Has PaperlessBilling? : ", ('Yes', 'No'))
    PaymentMethod = st.selectbox("Payment Method? : ", ['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'])
    MonthlyCharges = st.number_input("Monthly Charges: ", min_value=0.0, step=0.1)
    TotalCharges = st.number_input("Total Charges: ", min_value=0.0, step=0.1)

    # Prepare the input data as a dictionary
    input_data = {
        'gender': gender,
        'SeniorCitizen': SeniorCitizen,
        'Partner': Partner,
        'Dependents': Dependents,
        'tenure': tenure,
        'PhoneService': PhoneService,
        'MultipleLines': MultipleLines,
        'InternetService': InternetService,
        'OnlineSecurity': OnlineSecurity,
        'OnlineBackup': OnlineBackup,
        'DeviceProtection': DeviceProtection,
        'TechSupport': TechSupport,
        'StreamingTV': StreamingTV,
        'StreamingMovies': StreamingMovies,
        'Contract': Contract,
        'PaperlessBilling': PaperlessBilling,
        'PaymentMethod': PaymentMethod,
        'MonthlyCharges': MonthlyCharges,
        'TotalCharges': TotalCharges
    }

    # Prediction
    if st.button('Predict if this person will be likely to Churn or not!'):
        churns_or_not = churning_prediction(input_data)
        st.success(churns_or_not)


if __name__ == '__main__':
    main()
