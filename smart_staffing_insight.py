import datetime
import os
import joblib
import pandas as pd
import numpy as np
from datetime import timedelta
from sqlalchemy import create_engine, text
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
            business_day,
            bill_created_by AS employee_id,
            bill_total,
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
df = get_bills_data(client_id= "wahoos_fresno")
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


# Step-3 Create Historical Demand Table (This is for Forecasting)
daily_demand = (
    df.groupby(["date", "daypart"])
      .agg(total_bills=("bill_id", "count"))
      .reset_index()
)

# Add day_of_week again (for safety)
daily_demand["date"] = pd.to_datetime(daily_demand["date"])
daily_demand["day_of_week"] = daily_demand["date"].dt.dayofweek


# Step-4 Calculate Employee Productivity

# Calculate the number of unique bills handled by each employee in each daypart
employee_shift = (
    df.groupby(["date", "daypart", "employee_id"])
      .agg(bills_per_employee=("bill_id", "nunique"))
      .reset_index()
)

# Calculate average bills per employee for each daypart
avg_productivity = (
    employee_shift
        .groupby("daypart")
        .agg(avg_bills_per_staff=("bills_per_employee", "mean"))
        .reset_index()
)

# Step-5: Prepare features for ML model
X = pd.get_dummies(daily_demand[["day_of_week", "daypart"]])
y = daily_demand["total_bills"]

model = joblib.load("wahoos_fresno_tabpfn_model.joblib")

# # Step-6: Generate next 7 days
# last_date = daily_demand["date"].max()
# future_df = pd.DataFrame({
#     "date": pd.date_range(last_date + pd.Timedelta(days=1), periods=7),
# })
# future_df["day_of_week"] = future_df["date"].dt.dayofweek
# future_dayparts = ["Breakfast", "Lunch", "Dinner"]
# future_pred = []
#
# for _, row in future_df.iterrows():
#     for dp in future_dayparts:
#         features = pd.get_dummies(pd.DataFrame([{"day_of_week": row["day_of_week"], "daypart": dp}]))
#         features = features.reindex(columns=X.columns, fill_value=0)
#         pred = model.predict(features.values)[0]
#         future_pred.append({"date": row["date"], "daypart": dp, "predicted_bills": pred})
#
# future_forecast = pd.DataFrame(future_pred)


# Step-6: Generate next 7 days (Vectorized version)
last_date = daily_demand["date"].max()

future_dates = pd.date_range(last_date + pd.Timedelta(days=1), periods=7)

future_df = pd.DataFrame({
    "date": np.repeat(future_dates, 3),
    "daypart": ["Breakfast", "Lunch", "Dinner"] * 7
})

future_df["day_of_week"] = future_df["date"].dt.dayofweek

# Create dummy features
future_features = pd.get_dummies(future_df[["day_of_week", "daypart"]])

# Align columns with training data
future_features = future_features.reindex(columns=X.columns, fill_value=0)

# Predict ALL at once
future_df["predicted_bills"] = model.predict(future_features.values)

future_forecast = future_df.copy()

# Step-7: Calculate Required Staff (as before)
future_forecast = future_forecast.merge(
    avg_productivity,
    on="daypart",
    how="left"
)

future_forecast["required_staff"] = (
    future_forecast["predicted_bills"] /
    future_forecast["avg_bills_per_staff"]
)

# Add buffer (10%) if needed
# future_forecast["required_staff"] *= 1.1

# Round up
future_forecast["required_staff"] = np.ceil(future_forecast["required_staff"])

# Apply constraints
MAX_STAFF = len(df["employee_id"].unique())
MIN_STAFF = 1

future_forecast["required_staff"] = (
    future_forecast["required_staff"]
        .clip(lower=MIN_STAFF, upper=MAX_STAFF)
)

final_plan = future_forecast[
    ["date", "daypart", "predicted_bills", "required_staff"]
]

final_plan.to_csv("final_staffing_plan.csv", index=False)
