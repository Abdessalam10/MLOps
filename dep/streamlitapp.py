import streamlit as st
import joblib
from sklearn.preprocessing import LabelEncoder
import pandas as pd

# Page configuration
st.set_page_config(
    page_title="Flight Price Prediction",
    page_icon="✈️",
    layout="centered"
)

st.title("Flight Price Prediction App")
st.markdown("Enter flight details to predict the price:")


# Load model and dataset
@st.cache_resource
def load_model():
    model_path = "flight_price_model.pkl"
    dataset_path = "../datasets/flight_dataset.csv"

    model = joblib.load(model_path)
    data = pd.read_csv(dataset_path)

    return model, data


model, data = load_model()

# Clean categorical columns
categorical_columns = ['Airline', 'Source', 'Destination']

for column in categorical_columns:
    data[column] = data[column].astype(str).str.strip().str.lower()

# Create label encoders
label_encoders = {}

for column in categorical_columns:
    le = LabelEncoder()
    le.fit(data[column])   # fit is enough
    label_encoders[column] = le


# User inputs
st.subheader("Flight Details")

airline_input = st.selectbox(
    "Airline",
    sorted(data['Airline'].unique())
)

source_input = st.selectbox(
    "Source",
    sorted(data['Source'].unique())
)

destination_input = st.selectbox(
    "Destination",
    sorted(data['Destination'].unique())
)

total_stops_input = st.number_input(
    "Total Stops",
    min_value=0,
    max_value=10,
    step=1
)

duration_hours_input = st.number_input(
    "Duration (hours)",
    min_value=0.0,
    step=0.5
)

month_input = st.number_input(
    "Month (1-12)",
    min_value=1,
    max_value=12,
    step=1
)


# Safe label encoding
def safe_transform(le, value):
    value = str(value).strip().lower()

    if value in le.classes_:
        return le.transform([value])[0]

    st.warning(f"'{value}' not found in training data.")
    return -1


# Prediction
if st.button("Predict Price"):

    airline_encoded = safe_transform(
        label_encoders['Airline'],
        airline_input
    )

    source_encoded = safe_transform(
        label_encoders['Source'],
        source_input
    )

    destination_encoded = safe_transform(
        label_encoders['Destination'],
        destination_input
    )

    # Create dataframe for prediction
    new_data = pd.DataFrame([{
        'Airline': airline_encoded,
        'Source': source_encoded,
        'Destination': destination_encoded,
        'Total_Stops': total_stops_input,
        'Duration_hours': duration_hours_input,
        'Month': month_input
    }])

    predicted_price = model.predict(new_data)[0]

    st.success(f"Predicted Flight Price: {predicted_price:.2f}")