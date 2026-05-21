import pandas as pd
import mlflow
import mlflow.sklearn
from pathlib import Path
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, classification_report, confusion_matrix,recall_score,precision_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

EQ=pd.read_csv('../datasets/earthquake_alert_balanced_dataset.csv')


x = EQ[['magnitude','depth', 'cdi','mmi' , 'sig']]
y = EQ['alert']
le = LabelEncoder()
y = le.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)



mlflow.set_experiment("Earthquake Alert Prediction Experiment")
with mlflow.start_run(run_name="Earthquake Alert Model Performance Tracking"):
    model = RandomForestClassifier(n_estimators=100, random_state=42,max_depth=8)
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)

    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    classification_rep = classification_report(y_test, y_pred)
    confusion_mat = confusion_matrix(y_test, y_pred)
    recall = recall_score(y_test, y_pred, average='macro')
    precision = precision_score(y_test, y_pred, average='macro')
    # Log param
    mlflow.log_param("n_estimators", 100)
    mlflow.log_param("max_depth", 8)
    # Log metrics to MLflow
    mlflow.log_metric("Accuracy", accuracy)
    mlflow.log_metric("Mean Squared Error", mse)
    mlflow.log_metric("R2 Score", r2)
    mlflow.log_metric("Recall", recall)
    mlflow.log_metric("Precision", precision)
    mlflow.log_text(classification_rep, "classification_report.txt")
    mlflow.log_text(str(confusion_mat), "confusion_matrix.txt")
    print(f"Mean Squared Error: {mse}")
    print(f"R^2 Score: {r2}")
    print(f"Accuracy: {accuracy}")
    print(f"Recall: {recall}")
    print(f"Precision: {precision}")
    print("Classification Report:\n", classification_rep)
    print("Confusion Matrix:\n", confusion_mat)

    from mlflow.server import app
    mlflow.sklearn.log_model(model, "model")
    app.run(debug=True)