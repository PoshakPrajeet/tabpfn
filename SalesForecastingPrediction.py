import os
import sys
import pandas as pd
import numpy as np
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

    # Create features
    daily_sales["lag_1"] = daily_sales["total_sales"].shift(1)
    daily_sales["lag_7"] = daily_sales["total_sales"].shift(7)
    daily_sales["rolling_mean_7"] = daily_sales["total_sales"].rolling(7).mean()
    daily_sales["rolling_std_7"] = daily_sales["total_sales"].rolling(7).std()
    daily_sales["lag_14"] = daily_sales["total_sales"].shift(14)
    daily_sales["rolling_mean_14"] = daily_sales["total_sales"].rolling(14).mean()
    daily_sales["rolling_mean_30"] = daily_sales["total_sales"].rolling(30).mean()

    daily_sales = daily_sales.dropna().reset_index(drop=True)

    # Step-1 Load Data
    df = daily_sales.sort_values("date")

    future_df = daily_sales.copy().sort_values("date").reset_index(drop=True)

    last_date = future_df["date"].max()
    future_dates = [last_date + timedelta(days=i) for i in range(1, 8)]

    predictions = []

    for next_date in future_dates:
        # Create basic calendar features
        day_of_week = next_date.dayofweek
        month = next_date.month

        # Get required lag values from most recent data
        lag_1 = future_df.iloc[-1]["total_sales"]
        lag_7 = future_df.iloc[-7]["total_sales"]

        rolling_mean_7 = future_df["total_sales"].tail(7).mean()
        rolling_std_7 = future_df["total_sales"].tail(7).std()
        lag_14 = future_df.iloc[-14]["total_sales"]
        rolling_mean_14 = future_df["total_sales"].tail(14).mean()
        rolling_mean_30 = future_df["total_sales"].tail(30).mean()

        # Prepare input
        X_future = [[
            day_of_week,
            month,
            lag_1,
            lag_7,
            lag_14,
            rolling_mean_7,
            rolling_mean_14,
            rolling_mean_30,
            rolling_std_7
        ]]

        model = joblib.load(f"{client_id}_sales_forecasting_training_tabpfn.joblib")

        # Predict
        next_pred = model.predict(X_future)[0]

        # Store prediction
        predictions.append(next_pred)

        # Append prediction to dataframe (VERY IMPORTANT)
        new_row = {
            "date": next_date,
            "total_sales": next_pred
        }

        future_df = pd.concat(
            [future_df, pd.DataFrame([new_row])],
            ignore_index=True
        )

    forecast_df = pd.DataFrame({
        "date": future_dates,
        "predicted_sales": predictions
    })

    forecast_df.to_csv(f"{client_id}_7_day_sales_forecast.csv", index=False)