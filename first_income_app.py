import streamlit as st
import pickle
import numpy as np
import pandas as pd
from sklearn import preprocessing

# Load the trained model
with open("model.pkl", "rb") as model_file:
    model = pickle.load(model_file)

# Load the encoder dictionary
with open("encoder.pkl", "rb") as encoder_file:
    encoder_dict = pickle.load(encoder_file)

# Function to preprocess and encode user input
def preprocess_input(data, encoder_dict):
    """Encodes categorical features and converts data to a feature list for prediction."""
    df = pd.DataFrame([data])
    
    category_col = ['workclass', 'education', 'maritalstatus', 'occupation', 'relationship', 'race', 'gender', 'nativecountry']
    
    for col in category_col:
        le = preprocessing.LabelEncoder()
        le.classes_ = np.array(encoder_dict[col], dtype=object)  # Convert classes to correct dtype
        
        # Replace unseen values with 'Unknown'
        df[col] = df[col].apply(lambda x: x if x in le.classes_ else 'Unknown')
        
        # Apply Label Encoding
        df[col] = le.transform(df[col])

    # Convert DataFrame to list for model prediction
    return df.iloc[0].tolist()

# Function to make predictions
def predict_income(model, features):
    """Predict income category using the trained model."""
    features = np.array(features).reshape(1, -1)  # Ensure correct shape
    prediction = model.predict(features)
    return ">50K" if prediction[0] == 1 else "<=50K"

# Streamlit app
def main():
    st.title("Income Prediction App")
    st.write("Enter the details below to predict whether your income is **<=50K or >50K**.")

    # Input fields
    age = st.number_input("Age", min_value=0, max_value=110, value=22, step=1)
    work_class = st.selectbox("Working Class", ["Private", "Self-emp-not-inc", "Self-emp-inc", "Federal-gov", "Local-gov", "State-gov", "Without-pay", "Never-worked"])
    education_status = st.selectbox("Education", ["HS-grad", "Some-college", "Bachelors", "Masters", "Doctorate", "JD"])
    marital_status = st.selectbox("Marital Status", ["Married", "Divorced", "Never-married", "Separated", "Widowed"])
    occupation = st.selectbox("Select Occupation", ["Tech", "Other-service", "Business", "Engineering", "Armed-Forces"])
    relationship = st.selectbox("Relationship", ["Wife", "Own-child", "Husband", "Not-in-family", "Other-relative", "Unmarried"])
    race = st.selectbox("Race", ["White", "Black", "Asian-Pac-Islander", "Amer-Indian-Eskimo", "Other"])
    gender = st.selectbox("Gender", ["Male", "Female"])
    capital_gain = st.number_input("Capital Gain:", min_value=0, max_value=100000, value=0, step=100)
    capital_loss = st.number_input("Capital Loss:", min_value=0, max_value=50000, value=0, step=100)
    hours_per_week = st.number_input("Hours Per Week:", min_value=1, max_value=100, value=40, step=1)
    native_country = st.selectbox(
        "Select Native Country:", 
        ["United-States", "Cambodia", "England", "Puerto-Rico", "Canada", "Germany", 
         "Outlying-US(Guam-USVI-etc)", "India", "Japan", "Greece", "South", "China",
         "Cuba", "Iran", "Honduras", "Philippines", "Italy", "Poland", "Jamaica",
         "Vietnam", "Mexico", "Portugal", "Ireland", "France", "Dominican-Republic",
         "Laos", "Ecuador", "Taiwan", "Haiti", "Columbia", "Hungary", "Guatemala",
         "Nicaragua", "Scotland", "Thailand", "Yugoslavia", "El-Salvador", "Trinidad&Tobago",
         "Peru", "Hong", "Holand-Netherlands"]
    )

    # Prepare input data
    input_data = {
        "age": age,
        "workclass": work_class,
        "education": education_status,
        "maritalstatus": marital_status,
        "occupation": occupation,
        "relationship": relationship,
        "race": race,
        "gender": gender,
        "capitalgain": capital_gain,
        "capitalloss": capital_loss,
        "hoursperweek": hours_per_week,
        "nativecountry": native_country
    }

    # Predict button
    if st.button("Predict Income"):
        # Preprocess and encode user input into a feature list
        features = preprocess_input(input_data, encoder_dict)
        
        # Make prediction
        result = predict_income(model, features)
        
        # Display result
        st.success(f"Predicted Income Category: **{result}**")

if __name__ == "__main__":
    main()
