import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib

EQ=pd.read_csv('../datasets/earthquake_alert_balanced_dataset.csv')
FP=pd.read_csv('../datasets/flight_dataset.csv')

x = EQ[['magnitude','depth', 'cdi','mmi' , 'sig']]
y = EQ['alert']
le = LabelEncoder()
y = le.fit_transform(y)
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
rf = RandomForestClassifier(n_estimators=100, random_state=42,max_depth=8)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
print("Accuracy:", accuracy_score(y_test, y_pred))
joblib.dump(rf, 'earthquake_alert_model.pkl')
"""
magnitude = float(input("Enter the magnitude of the earthquake: "))
depth = float(input("Enter the depth of the earthquake: "))
cdi = float(input("Enter the Community Internet Intensity Map (cdi) value: "))
mmi = float(input("Enter the Modified Mercalli Intensity (mmi) value: "))
sig = float(input("Enter the significance (sig) value: "))"

new_data = pd.DataFrame([[magnitude, depth, cdi, mmi, sig]], columns=['magnitude', 'depth', 'cdi', 'mmi', 'sig'])
pred_encoded = rf.predict(new_data)[0]
new_prediction = le.inverse_transform([pred_encoded])[0]
print("New Prediction:", new_prediction)
"""
""""
import gradio as gr
def predict_earthquake_alert(magnitude, depth, cdi, mmi, sig):
    new_data = pd.DataFrame([[magnitude, depth, cdi, mmi, sig]], columns=['magnitude', 'depth', 'cdi', 'mmi', 'sig'])
    pred_encoded = rf.predict(new_data)[0]
    new_prediction = le.inverse_transform([pred_encoded])[0]
    return f"Predicted Alert: {new_prediction}"

interface = gr.Interface(
    fn=predict_earthquake_alert,
    inputs=[
        gr.Number(label="Magnitude"),
        gr.Number(label="Depth"),
        gr.Number(label="CDI"),
        gr.Number(label="MMI"),
        gr.Number(label="Sig")
    ],
    outputs=gr.Textbox(label="Predicted Alert"),
    title="Earthquake Alert Prediction",
    description="Enter earthquake parameters to predict the alert level."
)
interface.launch()

"""