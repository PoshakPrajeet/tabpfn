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
from datetime import timedelta


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

# query = 'SELECT c.clientID FROM c WHERE c.clientID NOT IN ("STATIC_DEFAULT", "the_icehouse", "hemingways", "local_pawleys")'
query = 'SELECT c.clientID FROM c WHERE c.clientID = "wahoos_fresno"'
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


    def get_bills_data(client_id: str):
        engine = get_sql_engine()
        query = text(f"""
            SELECT
                bill_id,
                bill_created_by AS employee_id,
                bill_open_date_time,
                date_key,
                bill_total
            FROM "{client_id}_bills"
            ORDER BY date_key
        """)
        with engine.connect() as connection:
            df = pd.read_sql_query(query, connection)
            engine.dispose()
        return df


    # Step-1 Load Data
    df = get_bills_data(client_id)
    df["bill_open_date_time"] = pd.to_datetime(df["bill_open_date_time"], errors='coerce')

    # Create date column
    df["date"] = df["bill_open_date_time"].dt.date

    # Create daily sales aggregation
    daily_sales = (
        df.groupby("date")
        .agg(total_sales=("bill_total", "sum"))
        .reset_index()
    )

    # Convert back to datetime
    daily_sales["date"] = pd.to_datetime(daily_sales["date"])

    # Add time features
    daily_sales["day_of_week"] = daily_sales["date"].dt.dayofweek
    daily_sales["month"] = daily_sales["date"].dt.month
    daily_sales["day_of_month"] = daily_sales["date"].dt.day
    daily_sales["week_of_year"] = daily_sales["date"].dt.isocalendar().week.astype(int)
    daily_sales.to_csv(f"{client_id}_daily_sales.csv", index=False)

    # Create features
    daily_sales["lag_1"] = daily_sales["total_sales"].shift(1)
    daily_sales["lag_7"] = daily_sales["total_sales"].shift(7)
    daily_sales["rolling_mean_7"] = daily_sales["total_sales"].rolling(7).mean()
    daily_sales["rolling_std_7"] = daily_sales["total_sales"].rolling(7).std()
    daily_sales["lag_14"] = daily_sales["total_sales"].shift(14)
    daily_sales["rolling_mean_14"] = daily_sales["total_sales"].rolling(14).mean()
    daily_sales["rolling_mean_30"] = daily_sales["total_sales"].rolling(30).mean()
    daily_sales.to_csv(f"{client_id}_daily_sales_with_features.csv", index=False)

    daily_sales = daily_sales.dropna().reset_index(drop=True)

    # Step-1 Load Data
    df = daily_sales.sort_values("date")

    # Step-5: Prepare features for ML model
    X = df[[
    "day_of_week",
    "month",
    "lag_1",
    "lag_7",
    "lag_14",
    "rolling_mean_7",
    "rolling_mean_14",
    "rolling_mean_30",
    "rolling_std_7"
    ]]
    y = df["total_sales"]

    # Last 7 days as test
    test_idx = df["date"].isin(df["date"].unique()[-7:])
    X_train = X[~test_idx]
    X_test = X[test_idx]
    y_train = y[~test_idx]
    y_test = y[test_idx]

    # Train TabPFNRegressor
    model = TabPFNRegressor(ignore_pretraining_limits=True)
    model.fit(X_train.values, y_train.values)

    joblib.dump(model, f"{client_id}_sales_forecasting_training_tabpfn.joblib")
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