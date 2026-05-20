import pandas as pd

from sklearn.preprocessing import LabelEncoder
import joblib
from fastapi import FastAPI, Request
from pydantic import BaseModel
import uvicorn

file_path ="earthquake_alert_model.pkl"
app = FastAPI(title="Earthquake Alert Prediction API")
@app.get("/")
async def root():
    return {"message": "Welcome to the Earthquake Alert Prediction API. Use the /predict endpoint to get predictions."}
class EarthquakeData(BaseModel):
    magnitude: float
    depth: float
    cdi: float
    mmi: float
    sig: float
@app.post("/predict")
async def predict(data: EarthquakeData):
    model = joblib.load(file_path)
    le = LabelEncoder()
    le.fit(['green', 'yellow', 'orange', 'red'])
    new_data = pd.DataFrame([[data.magnitude, data.depth, data.cdi, data.mmi, data.sig]], columns=['magnitude', 'depth', 'cdi', 'mmi', 'sig'])
    pred_encoded = model.predict(new_data)[0]
    new_prediction = le.inverse_transform([pred_encoded])[0]
    return {"predicted_alert": new_prediction}
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8089)
    