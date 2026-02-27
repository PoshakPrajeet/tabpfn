import os
from tabpfn import TabPFNRegressor
import sys
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error,mean_absolute_error
import joblib  # for saving model
from datetime import datetime
from azure.storage.blob import BlobServiceClient
from sqlalchemy import create_engine, text
from azure.cosmos import CosmosClient
from azure.identity import DefaultAzureCredential
from azure.keyvault.secrets import SecretClient

KEY_VAULT_URL = os.environ.get("KEY_VAULT_URL")

credential = DefaultAzureCredential()
secret_client = SecretClient(
    vault_url=KEY_VAULT_URL,
    credential=credential
)

def get_secret(name: str) -> str:
    return secret_client.get_secret(name).value


COSMOS_ENDPOINT = get_secret("COSMOS-DB-ENDPOINT")
COSMOS_KEY = get_secret("COSMOS-DB-KEY")
DATABASE_NAME = get_secret("COSMOS-DB-CONFIGS-DATABASE-NAME")
CONTAINER_NAME = get_secret("COSMOS-DB-CLIENT-CONFIG-CONTAINER-NAME")

cosmos_client = CosmosClient(COSMOS_ENDPOINT, COSMOS_KEY)
database = cosmos_client.get_database_client(DATABASE_NAME)
container = database.get_container_client(CONTAINER_NAME)

query = 'SELECT c.clientID FROM c WHERE c.clientID NOT IN ("STATIC_DEFAULT", "the_icehouse", "hemingways", "local_pawleys")'
items = list(container.query_items(query=query, enable_cross_partition_query=True))

for item in items:
    client_id = item["clientID"]
    print("Processing client:", client_id)


    def get_sql_engine():
        server = get_secret("SQL-DB-SERVER")
        database = get_secret("SQL-DB-DATABASE-SILVER")
        username = get_secret("SQL-DB-USER")
        password = get_secret("SQL-DB-PASSWORD")

        engine = create_engine(
            f"mssql+pymssql://{username}:{password}@{server}:1433/{database}",
            pool_pre_ping=True,
            pool_recycle=3600,
        )
        return engine

    def get_bills_data( client_id: str):
        engine = get_sql_engine()
        query = text(f"""
            SELECT
                bill_id,
                bill_created_by AS employee_id,
                bill_open_date_time,
                date_key
            FROM "{client_id}_bills"
            ORDER BY date_key
        """)
        with engine.connect() as connection:
            df = pd.read_sql_query(query, connection, params={"client_id": client_id})
            engine.dispose()
        return df


    # Step-1 Load Data
    df = get_bills_data(client_id)
    df["bill_open_date_time"] = pd.to_datetime(df["bill_open_date_time"], errors='coerce')

    df["date"] = df["bill_open_date_time"].dt.date
    df["hour"] = df["bill_open_date_time"].dt.hour
    df["day_of_week"] = df["bill_open_date_time"].dt.dayofweek


    # Step-2 Create Daypart
    def get_daypart(hour):
        if 6 <= hour < 11:
            return "Breakfast"
        elif 11 <= hour < 16:
            return "Lunch"
        elif 16 <= hour <= 23:
            return "Dinner"
        else:
            return "Other"


    df["daypart"] = df["hour"].apply(get_daypart)
    # # Remove non-operating hours if needed
    df = df[df["daypart"] != "Other"]

    # Step-3 Create Historical Demand Table
    daily_demand = (
        df.groupby(["date", "daypart"])
        .agg(total_bills=("bill_id", "count"))
        .reset_index()
    )

    # Add day_of_week again (for safety)
    daily_demand["date"] = pd.to_datetime(daily_demand["date"])
    daily_demand["day_of_week"] = daily_demand["date"].dt.dayofweek

    # Step-1 Load Data
    df = daily_demand
    # Step-5: Prepare features for ML model
    X = pd.get_dummies(df[["day_of_week", "daypart"]])
    y = df["total_bills"]

    # Use last 7 days as test set (optional, for evaluation)
    daily_demand_sorted = daily_demand.sort_values("date")
    test_idx = daily_demand_sorted["date"].isin(daily_demand_sorted["date"].unique()[-7:])
    X_train = X[~test_idx]
    X_test = X[test_idx]
    y_train = y[~test_idx]
    y_test = y[test_idx]

    # Train TabPFNRegressor
    model = TabPFNRegressor(ignore_pretraining_limits=True)
    model.fit(X_train.values, y_train.values)

    joblib.dump(model, f"{client_id}_tabpfn_model.joblib")
    print(f"Model trained and saved as '{client_id}_tabpfn_model.joblib'.")

    y_pred = model.predict(X_test.values)

    print(f"MAE: {round(mean_absolute_error(y_test, y_pred), 2)}")
    print(f"MSE: {mean_squared_error(y_test, y_pred):.2f}")
    print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.2f}")
    print(f"R2: {r2_score(y_test, y_pred):.2f}")
    print(f"MAPE: {mean_absolute_percentage_error(y_test, y_pred):.2%}")
    print(f"Confidence Score: {model.score(X_test.values, y_test.values):.2f}")


    # Save it to Datalake
    try:
        CONNECTION_STRING = get_secret("DATALAKE-CONNECTION-STRING")
        CONTAINER_NAME = get_secret("DATALAKE-INSIGHTS-CONTAINER-NAME")
    except KeyError as e:
        raise KeyError(f"Missing required environment variable: {e}")
    TARGET_DIR = f"{client_id}/"  # folder path inside container

    blob_service_client = BlobServiceClient.from_connection_string(CONNECTION_STRING)
    container_client = blob_service_client.get_container_client(CONTAINER_NAME)

    files_to_upload = [
        f"{client_id}_tabpfn_model.joblib",
    ]

    for file_name in files_to_upload:
        blob_path = f"{TARGET_DIR}{file_name}"
        blob_client = container_client.get_blob_client(blob_path)

        with open(file_name, "rb") as data:
            blob_client.upload_blob(data, overwrite=True)

        print(f"Uploaded: {blob_path}")