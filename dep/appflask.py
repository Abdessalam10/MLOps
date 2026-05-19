import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import joblib
import os
from flask import Flask, request, render_template_string
app = Flask(__name__)

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

def safe_transform(le, value):
    if value in le.classes_:
        return le.transform([value])[0]
    else:
        print(f"Warning: '{value}' not found in training data. Using 'unknown' category.")
        return -1

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Flight Price Prediction</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 40px;
            background-color: #f4f4f4;
        }
        h1 {
            color: #333;
        }
        form {
            background-color: #fff;
            padding: 20px;
            border-radius: 5px;
            box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
        }
        label {
            display: block;
            margin-top: 10px;
            font-weight: bold;
        }
        input[type="text"], input[type="number"] {
            width: 100%;
            padding: 8px;
            margin-top: 5px;
            border: 1px solid #ccc;
            border-radius: 4px;
        }
        input[type="submit"] {
            margin-top: 20px;
            padding: 10px 15px;
            background-color: #28a745;
            color: white;
            border: none;
            border-radius: 4px;
            cursor: pointer;
        }
        input[type="submit"]:hover {
            background-color: #218838;
        }
        h2 {
            margin-top: 20px;
            color: #555;
        }
    </style>
</head>
<body>
    <h1>Flight Price Prediction</h1>
    <form method="post">
        <label for="airline">Airline:</label><br>
        <input type="text" id="airline" name="airline" required><br><br>
        
        <label for="source">Source:</label><br>
        <input type="text" id="source" name="source" required><br><br>
        
        <label for="destination">Destination:</label><br>
        <input type="text" id="destination" name="destination" required><br><br>
        
        <label for="total_stops">Total Stops:</label><br>
        <input type="number" id="total_stops" name="total_stops" required><br><br>
        
        <label for="duration_hours">Duration (hours):</label><br>
        <input type="number" step="0.1" id="duration_hours" name="duration_hours" required><br><br>
        
        <label for="month">Month (1-12):</label><br>
        <input type="number" id="month" name="month" required><br><br>
        
        <input type="submit" value="Predict Price">
    </form>

    {% if predicted_price is not none %}
    <h2>Predicted Flight Price: {{ predicted_price }}</h2>
    {% endif %}
</body>
</html>
"""

@app.route('/', methods=['GET', 'POST'])
def predict():
    predicted_price = None

    if request.method == 'POST':
        airline = request.form['airline'].strip().lower()
        source = request.form['source'].strip().lower()
        destination = request.form['destination'].strip().lower()
        total_stops = int(request.form['total_stops'])
        duration_hours = float(request.form['duration_hours'])
        month = int(request.form['month'])

        airline_encoded = safe_transform(label_encoders['Airline'], airline)
        source_encoded = safe_transform(label_encoders['Source'], source)
        destination_encoded = safe_transform(label_encoders['Destination'], destination)

        new_data = pd.DataFrame([{
            'Airline': airline_encoded,
            'Source': source_encoded,
            'Destination': destination_encoded,
            'Total_Stops': total_stops,
            'Duration_hours': duration_hours,
            'Month': month
        }])

        predicted_price = model.predict(new_data)[0]

    return render_template_string(HTML_TEMPLATE, predicted_price=predicted_price)

if __name__ == '__main__':
    app.run(debug=True)
    