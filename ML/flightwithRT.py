import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score



FP=pd.read_csv('../datasets/flight_dataset.csv')
for column in ['Airline', 'Source', 'Destination']:
    FP[column] = FP[column].astype('str').str.strip().str.lower()
    
features = ['Airline', 'Source', 'Destination', 'Total_Stops', 'Duration_hours','Month']

target = 'Price'

X = FP.loc[:, features].copy()
y = FP[target]
label_encoders = {}
for column in ['Airline', 'Source', 'Destination']:
    le = LabelEncoder()
    X[column] = le.fit_transform(X[column].astype(str))
    label_encoders[column] = le
print(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = DecisionTreeRegressor(max_depth=6, random_state=42)
model.fit(X_train, y_train)
print('Enter flight details to predict the price:')
airline = input('Airline: ').strip().lower()
source = input('Source: ').strip().lower()
destination = input('Destination: ').strip().lower()
total_stops = int(input('Total Stops: ').strip())
duration_hours = float(input('Duration (hours): ').strip())
month = int(input('Month (1-12): ').strip())

def safe_transform(le, value):
  if value in le.classes_:
    return le.transform([value])[0]
  else:
    print(f"Warning: '{value}' not found in training data. Using 'unknown' category.")
    return -1

airline_encoded = safe_transform(label_encoders['Airline'], airline)
source_encoded = safe_transform(label_encoders['Source'], source)
destination_encoded = safe_transform(label_encoders['Destination'], destination)
new_data = [[airline_encoded, source_encoded, destination_encoded, total_stops, duration_hours, month]]
predicted_price = model.predict(new_data)[0]
print(f"Predicted Flight Price: {predicted_price:.2f}")

import joblib 

joblib.dump(model, 'flight_price_model.pkl')


