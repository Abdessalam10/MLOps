from airflow import DAG
from airflow.providers.standard.operators.python import PythonOperator

from datetime import datetime
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import pickle


def data_augmentation():
    data = pd.read_csv('../datasets/augmented_earthquake_data.csv')
    return data.to_dict()  # Airflow XCom-compatible


def train_model(ti):
    # Get data from previous task
    data_dict = ti.xcom_pull(task_ids='data_augmentation')
    data = pd.DataFrame(data_dict)

    X = data[['magnitude', 'depth', 'latitude', 'longitude']]
    y = data['alert']

    le = LabelEncoder()
    y = le.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )

    model.fit(X_train, y_train)

    with open('../models/earthquake_model.pkl', 'wb') as f:
        pickle.dump(model, f)

    with open('../models/label_encoder.pkl', 'wb') as f:
        pickle.dump(le, f)

    print("Model training completed successfully.")


def test_model():
    with open('../models/earthquake_model.pkl', 'rb') as f:
        model = pickle.load(f)

    with open('../models/label_encoder.pkl', 'rb') as f:
        le = pickle.load(f)

    test_data = pd.DataFrame(
        [[7.0, 10.0, 34.0, -118.0]],
        columns=['magnitude', 'depth', 'latitude', 'longitude']
    )

    predictions = model.predict(test_data)
    predicted_label = le.inverse_transform(predictions)

    print(f'Predicted Alert: {predicted_label[0]}')

    return predicted_label[0]


default_args = {
    'owner': 'airflow',
    'retries': 1,
}


with DAG(
    dag_id='earthquake_alert_model_training',
    default_args=default_args,
    start_date=datetime(2024, 6, 1),
    schedule=None,
    catchup=False,
    description='DAG for training and testing earthquake alert model',
) as dag:

    data_augmentation_task = PythonOperator(
        task_id='data_augmentation',
        python_callable=data_augmentation,
    )

    train_task = PythonOperator(
        task_id='train_model',
        python_callable=train_model,
    )

    test_task = PythonOperator(
        task_id='test_model',
        python_callable=test_model,
    )

    data_augmentation_task >> train_task >> test_task