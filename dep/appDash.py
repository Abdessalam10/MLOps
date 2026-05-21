import dash 
from dash import html, dcc, Input, Output, State
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder

app = dash.Dash(__name__)
app.title = "Flight Price Prediction"
# Load the trained model and label encoders
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

# Safe label encoding
def safe_transform(le, value):
    value = str(value).strip().lower()

    if value in le.classes_:
        return le.transform([value])[0]

    return -1


app.layout = html.Div(children=[
    html.H1("Flight Price Prediction App"),
    html.Div([
        html.Label("Airline"),
        dcc.Dropdown(
            id="airline-input",
            options=[{"label": airline, "value": airline} for airline in sorted(data['Airline'].unique())],
            placeholder="Select an airline"
        ),
    ], style={"margin-bottom": "20px"}),
    html.Div([
        html.Label("Source"),
        dcc.Dropdown(
            id="source-input",
            options=[{"label": source, "value": source} for source in sorted(data['Source'].unique())],
            placeholder="Select a source"
        ),
    ], style={"margin-bottom": "20px"}),
    html.Div([
        html.Label("Destination"),
        dcc.Dropdown(
            id="destination-input",
            options=[{"label": destination, "value": destination} for destination in sorted(data['Destination'].unique())],
            placeholder="Select a destination"
        ),
    ], style={"margin-bottom": "20px"}),
    html.Div([
        html.Label("Total Stops"),
        dcc.Input(
            id="total-stops-input",
            type="number",
            min=0,
            max=10,
            step=1,
            placeholder="Enter total stops"
        ),
    ], style={"margin-bottom": "20px"}),
    html.Div([
        html.Label("Duration (hours)"),
        dcc.Input(
            id="duration-hours-input",
            type="number",
            min=0,
            step=0.5,
            placeholder="Enter duration in hours"
        ),
    ], style={"margin-bottom": "20px"}),
    html.Div([
        html.Label("Month (1-12)"),
        dcc.Input(
            id="month-input",
            type="number",
            min=1,
            max=12,
            step=1,
            placeholder="Enter month (1-12)"
        ),
    ], style={"margin-bottom": "20px"}),
    html.Button("Predict Price", id="predict-button", n_clicks=0),
    html.Div(id="prediction-result", style={"margin-top": "20px", "font-size": "24px"})
])


@app.callback(
    Output("prediction-result", "children"),
    Input("predict-button", "n_clicks"),
    State("airline-input", "value"),
    State("source-input", "value"),
    State("destination-input", "value"),
    State("total-stops-input", "value"),
    State("duration-hours-input", "value"),
    State("month-input", "value")
)

def predict_price(n_clicks, airline_input, source_input, destination_input, total_stops_input, duration_hours_input, month_input):
    if n_clicks == 0:
        return ""

    if not all([airline_input, source_input, destination_input, total_stops_input is not None, duration_hours_input is not None, month_input is not None]):
        return "Please fill in all fields."

    # Safe label encoding
    airline_encoded = safe_transform(label_encoders['Airline'], airline_input)
    source_encoded = safe_transform(label_encoders['Source'], source_input)
    destination_encoded = safe_transform(label_encoders['Destination'], destination_input)

    if -1 in [airline_encoded, source_encoded, destination_encoded]:
        return "Invalid input for categorical fields. Please select from the dropdowns."

    # Create dataframe for prediction
    input_data = pd.DataFrame({
        'Airline': [airline_encoded],
        'Source': [source_encoded],
        'Destination': [destination_encoded],
        'Total_Stops': [total_stops_input],
        'Duration_hours': [duration_hours_input],
        'Month': [month_input]
    })

    # Predict price
    predicted_price = model.predict(input_data)[0]

    return f"Predicted Flight Price: {predicted_price:.2f}"

if __name__ == "__main__":
    app.run(debug=True)