import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import os
import io
import zipfile
import requests
from prophet import Prophet
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score

st.sidebar.title("RetailPulse Dashboard")
st.sidebar.write("Customer & Sales Analytics")

st.title("RetailPulse")
st.write(
    "RetailPulse is a retail sales analytics dashboard "
    "for customer analysis, sales trends, and demand forecasting."
)
file_path = "online_retail_II.xlsx"

if os.path.exists(file_path):
    df1 = pd.read_excel(file_path, sheet_name="Year 2009-2010")
    df2 = pd.read_excel(file_path, sheet_name="Year 2010-2011")
    df = pd.concat([df1, df2], ignore_index=True)

else:
    st.info("Downloading the Online Retail II dataset...")

    url = "https://archive.ics.uci.edu/static/public/502/online+retail+ii.zip"
    response = requests.get(url, timeout=180)
    response.raise_for_status()

    with zipfile.ZipFile(io.BytesIO(response.content)) as z:
        excel_name = next(
            name for name in z.namelist()
            if name.lower().endswith(".xlsx")
        )
        excel_bytes = z.read(excel_name)

    excel_data = pd.ExcelFile(io.BytesIO(excel_bytes))
    df1 = pd.read_excel(excel_data, sheet_name="Year 2009-2010")
    df2 = pd.read_excel(excel_data, sheet_name="Year 2010-2011")
    df = pd.concat([df1, df2], ignore_index=True)

df["StockCode"] = df["StockCode"].astype(str)

df = df.dropna(subset=["Customer ID"])

duplicate_count = df.duplicated().sum()

df = df.drop_duplicates()

df["TotalAmount"] = df["Quantity"] * df["Price"]

df["Year"] = df["InvoiceDate"].dt.year

df["Month"] = df["InvoiceDate"].dt.month

# Customer Segmentation using RFM Analysis

reference_date = df["InvoiceDate"].max() + pd.Timedelta(days=1)

rfm = df.groupby("Customer ID").agg(
    recency=("InvoiceDate", lambda x: (reference_date - x.max()).days),
    Frequency=("Invoice", "nunique"),
    Monetary=("TotalAmount", "sum")
).reset_index()

total_sales = df["TotalAmount"].sum()

total_transactions = len(df)
unique_customers = df["Customer ID"].nunique()

col1, col2, col3 = st.columns(3)

col1.metric("Total Sales", f"£{total_sales:,.2f}")
col2.metric("Total Transactions", f"{total_transactions:,}")
col3.metric("Unique Customers", f"{unique_customers:,}")

st.subheader("Sales by Year")

year_sales = df.groupby("Year")["TotalAmount"].sum()

st.bar_chart(year_sales)

st.subheader("Monthly Sales Trend")

month_sales = df.groupby("Month")["TotalAmount"].sum()

st.line_chart(month_sales)

st.subheader("Top 10 Products by Sales")

top_products = (
    df.groupby("Description")["TotalAmount"]
    .sum()
    .sort_values(ascending=False)
    .head(10)
)

st.bar_chart(top_products)

top_products = ()

st.subheader("Country Filter")

countries = sorted(df["Country"].dropna().unique())

selected_country = st.selectbox(
    "Select a country",
    ["All Countries"] + countries
)

if selected_country == "All Countries":
    filtered_df = df
else:
    filtered_df = df[df["Country"] == selected_country]

st.write("Selected Country:", selected_country)
st.write("Sales:", f"£{filtered_df['TotalAmount'].sum():,.2f}")

st.subheader("Customer Analysis")

customer_count = filtered_df["Customer ID"].nunique()
transaction_count = len(filtered_df)

col1, col2 = st.columns(2)

col1.metric("Customers", f"{customer_count:,}")
col2.metric("Transactions", f"{transaction_count:,}")

st.subheader("Top 10 Countries by Sales")

country_sales = (
    df.groupby("Country")["TotalAmount"]
    .sum()
    .sort_values(ascending=False)
    .head(10)
)

st.bar_chart(country_sales)

st.subheader("Sales Summary")

average_order_value = df["TotalAmount"].mean()

st.metric(
    "Average Transaction Value",
    f"£{average_order_value:,.2f}"
)

st.subheader("Top 10 Customers by Sales")

customer_sales = (
    df.groupby("Customer ID")
["TotalAmount"]
    .sum()
    .sort_values(ascending=False)
    .head(10)
)

st.bar_chart(customer_sales)

st.subheader("Top 10 Products by Quantity Sold")

product_quantity = (
    df.groupby("Description")
["Quantity"]
    .sum()
    .sort_values(ascending=False)
    .head(10)
)

st.bar_chart(product_quantity)

st.subheader("Demand Forecasting")

monthly_sales = (
    df.groupby("InvoiceDate")
["TotalAmount"]
    .sum()
    .reset_index()
)

monthly_sales.columns = ["date", "sales"]
st.write("Monthly sales columns:", monthly_sales.columns.tolist())
monthly_sales["date"] = pd.to_datetime(monthly_sales["date"])

st.write("Monthly sales data prepared for forecasting")
st.dataframe(monthly_sales.head(10))

st.subheader("Demand Forecasting")

monthly_sales = (
    df.set_index("InvoiceDate")["TotalAmount"]
    .resample("ME")
    .sum()
    .reset_index()
)

monthly_sales.columns = ["Date", "Sales"]

st.write("Monthly sales data prepared for forecasting")
st.dataframe(monthly_sales)

st.subheader("Historical Monthly Sales")

st.line_chart(
    monthly_sales.set_index("Date")["Sales"]
)

forecast_data = monthly_sales.rename(
    columns={"Date": "ds", "Sales": "y"}
)

model = Prophet()

model.fit(forecast_data)
# Prepare daily sales data for LSTM
daily_sales = monthly_sales.copy()
daily_sales.columns = ["date", "sales"]
daily_sales["date"] = pd.to_datetime(daily_sales["date"])
daily_sales = daily_sales.set_index("date").resample("D")["sales"].sum().reset_index()
st.write("Daily sales columns:", daily_sales.columns.tolist())

#Prepare sales values for LSTM training
daily_sales = daily_sales.sort_values("date")
daily_sales["sales"] = daily_sales["sales"].fillna(0)

scaler = MinMaxScaler()
scaled_sales = scaler.fit_transform(daily_sales[["sales"]])

sequence_length = 30
X_lstm = []
y_lstm = []

for i in range(sequence_length, len(scaled_sales)):
    X_lstm.append(scaled_sales[i-sequence_length:i, 0])
    y_lstm.append(scaled_sales[i, 0])

X_lstm = np.array(X_lstm).reshape(-1, sequence_length, 1)
y_lstm = np.array(y_lstm)

# Build and train the LSTM model
if len(X_lstm) > 0:
    lstm_model = Sequential([
        LSTM(50, activation="tanh", input_shape=(sequence_length, 1)),
        Dense(1)
    ])

    lstm_model.compile(
        optimizer="adam",
        loss="mean_squared_error"
    )

    lstm_model.fit(
        X_lstm,
        y_lstm,
        epochs=10,
        batch_size=32,
        verbose=0
    )

        # Generate LSTM predictions
    last_sequence = scaled_sales[-sequence_length:].reshape(1, sequence_length, 1)
    lstm_predictions = []

    for _ in range(30):
        next_prediction = lstm_model.predict(last_sequence, verbose=0)
        lstm_predictions.append(next_prediction[0, 0])
        last_sequence = np.append(
            last_sequence[:, 1:, :],
            next_prediction.reshape(1, 1, 1),
            axis=1
        )

    lstm_predictions = scaler.inverse_transform(
        np.array(lstm_predictions).reshape(-1, 1)
    ).flatten()

future = model.make_future_dataframe(periods=30, freq="D")

forecast = model.predict(future)

# Combine Prophet and LSTM forecasts
prophet_future = forecast.tail(30).reset_index(drop=True)

if len(prophet_future) == len(lstm_predictions):
    ensemble_predictions = (
        prophet_future["yhat"].to_numpy() + lstm_predictions
    ) / 2

    ensemble_forecast_df = pd.DataFrame({
        "Date": prophet_future["ds"],
        "Ensemble Predicted Sales": ensemble_predictions
    })

st.subheader("Forecast Accuracy")

historical_forecast = forecast[["ds", "yhat"]].merge(
    forecast_data,
    on="ds",
    how="inner"
)

historical_forecast = historical_forecast[
    historical_forecast["y"] != 0
]

if not historical_forecast.empty:
    historical_forecast["APE"] = (
        abs(
            historical_forecast["y"]
            - historical_forecast["yhat"]
        )
        / abs(historical_forecast["y"])
    ) * 100

    mape = historical_forecast["APE"].mean()
    st.metric("MAPE (%)", f"{mape:.2f}%")
else:
    st.write("Not enough valid data to calculate MAPE.")

st.subheader("Sales Forecast Results")

st.write("Predicted sales for the next six months:")

st.dataframe(
    forecast[["ds",
"yhat"]].tail(6)
)

st.subheader("Future Sales Forecast Chart")

forecast_chart = forecast[["ds", "yhat"]].tail(6)

forecast_chart = forecast_chart.rename(
    columns={"ds": "Date", "yhat": "Predicted Sales"}
)

st.line_chart(

forecast_chart.set_index("Date")
["Predicted Sales"]
)

st.subheader("Customer Segmentation - RFM Analysis")

st.write(
    "RFM analysis helps identify valuable customers "
    "based on their purchasing behaviour."
)

st.write("RFM Analysis Results")

st.dataframe(rfm.head(10))

rfm["Customer Segment"] = pd.qcut(

rfm["Monetary"].rank(method="first"),
    q=3,
    labels=["Low Value", "Medium Value", "High Value"]
)

st.subheader("Customer Segments")

st.write("Low Value: Customers with relatively low total spending.")
st.write("Medium Value: Customers with moderate level spending.")
st.write("High Value: Customers with relatively high total spending")

st.dataframe(rfm.head(10))

#Customer segmentation using K-Means 

st.subheader("K-Means Customer Segmentation")

st.write("RFM column names:", rfm.columns.tolist())
cluster_data = rfm[["recency", "Frequency", "Monetary"]].copy()

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

scaler = StandardScaler()
scaled_data = scaler.fit_transform(cluster_data)

kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
rfm["Cluster"] = kmeans.fit_predict(scaled_data)

st.write("Number of customer groups:", rfm["Cluster"].nunique())
st.dataframe(rfm[[
    "Customer ID", "recency", "Frequency", "Monetary", "Cluster"
]].head(10))

st.subheader("Customer Count by Cluster")
st.bar_chart(rfm["Cluster"].value_counts().sort_index())

#Understand customer behaviour in each cluster

st.subheader("Customer Cluster Summary")

cluster_summary = rfm.groupby("Cluster").agg(
    Customer_Count=("Customer ID", "count"),
    Average_Recency=("recency", "mean"),
    Average_Frequency=("Frequency", "mean"),
    Average_Monetary=("Monetary", "mean")
).round(2).reset_index()

st.dataframe(cluster_summary)

#Interpret customer clusters

st.subheader("Customer Cluster Insights")

for _, row in cluster_summary.iterrows():
    st.write(
        f"Cluster{int(row['Cluster'])}: "
        f"{int(row['Customer_Count'])} customers, "
        f"average spending = {row['Average_Monetary']:.2f}, "
        f"average purchase frequency = {row['Average_Frequency']:.2f}, "
        f"average recency = {row['Average_Recency']:.2f} days."
    )

# Inventory Reorder Recommendations

st.subheader("Inventory Reorder Recommendations")

product_demand = df.groupby("Description").agg(
    Total_Quantity_Sold=("Quantity", "sum"),
    Total_Sales=("TotalAmount", "sum")
).reset_index()

product_demand = product_demand[
    product_demand["Total_Quantity_Sold"] > 0
].copy()
product_demand = product_demand.sort_values(
    "Total_Quantity_Sold", ascending=False
)

product_demand["Average Monthly Demand"] = (
    product_demand["Total_Quantity_Sold"] / 24
).round(0)

product_demand["Suggested Reorder Quantity"] = (
    product_demand["Average Monthly Demand"] * 1
).astype(int)

st.write(
    "These estimates use historical sales as a simple demand guide"
    "Actual stock levels are not available in this dataset."
)

st.dataframe(product_demand.head(10))

rfm["Churn Risk"] = rfm["recency"].apply(
    lambda days: "High Risk" if days > 180
    else "Medium Risk" if days > 90
    else "Low Risk"
)

st.subheader("Customer Churn Risk")
st.dataframe(rfm.head(10))

st.subheader("Prophet + LSTM Ensemble Forecast")

if "ensemble_forecast_df" in locals():
    st.line_chart(
        ensemble_forecast_df.set_index("Date")
    )
else:
    st.warning("Ensemble forecast is not available.")

st.subheader("LSTM Sales Forecast")

if "lstm_predictions" in locals():
    lstm_forecast_df = pd.DataFrame({
        "Day": range(1, len(lstm_predictions) + 1),
        "Predicted Sales": lstm_predictions
    })

    st.line_chart(
        lstm_forecast_df.set_index("Day")
    )
else:
    st.warning("LSTM forecast is not available.")

st.subheader("Inventory Risk Analysis")

product_stock = df.groupby("Description").agg(

Total_Quantity_sold=("Quantity", "sum"),
    Total_Sales=("TotalAmount", "sum")
).reset_index()

product_stock = product_stock.sort_values(
    by="Total_Quantity_sold",
    ascending=True
)

st.write("Products with the lowest total quantity sold:")
st.dataframe(product_stock.head(10))

# Prepare data for churn prediction

st.subheader("Churn Prediction - Data Preparation")

data_end = df["InvoiceDate"].max()
cutoff_date = data_end - pd.Timedelta(days=90)

feature_data = df[df["InvoiceDate"] < cutoff_date]

future_data = df[
    (df["InvoiceDate"] >= cutoff_date)
    & (df["InvoiceDate"] <= data_end)
]

st.write("Historical transactions:", len(feature_data))
st.write("Transactions in the final 90 days:", len(future_data))
 
customer_history=feature_data.groupby("Customer ID").agg(
Recency=("InvoiceDate", lambda x:(cutoff_date-x.max()).days),
Frequency=("Invoice","nunique"),
Monetary=("TotalAmount","sum"),
Tenure=("InvoiceDate", lambda x:(cutoff_date - x.min()).days)
).reset_index()

st.subheader("Customer Churn Model-Customer Data")
st.dataframe(customer_history.head(10))

recent_customers=future_data["Customer ID"].dropna().unique()

customer_history["Churn"]=customer_history["Customer ID"].apply(
lambda customer: 0 if customer in recent_customers else 1
)

st.subheader("Customer Churn Labels")
st.write("0 = Purchased in the final 90 days")
st.write("1 = Did not purchase in the final 90 days")
st.dataframe(customer_history.head(10))

st.write("Churn label counts:")
st.write(customer_history["Churn"].value_counts())
features = ["Recency","Frequency","Monetary","Tenure"]
X = customer_history[features]
y = customer_history["Churn"]
st.write("Churn class counts:", y.value_counts().to_dict())

st.write("Number of customers:", len(customer_history))
st.write("Number of model features:", len(features))

st.write("Checking data for model training:")
st.write("Missing values:", X.isnull().sum().sum())
st.write("Churn labels available:", y.nunique())

# Train the customer churn prediction model

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    min_samples_split=10,
    min_samples_leaf=5,
    random_state=42,
    class_weight="balanced"
)
model.fit(X_train, y_train)
st.subheader("Churn Model Feature Importance")

feature_importance = pd.DataFrame({
    "Feature": X_train.columns,
    "Importance": model.feature_importances_
}).sort_values("Importance", ascending=False)

st.dataframe(feature_importance)
st.write("Total Feature Importance:", round(feature_importance["Importance"].sum(), 4))

predictions = model.predict(X_test)
auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
probabilities = model.predict_proba(X_test)[:, 1]

top_20_count = max(1, int(len(probabilities) * 0.20))

top_20_indices = probabilities.argsort()[::-1][:top_20_count]

precision_top_20 = y_test.iloc[top_20_indices].mean()

report = classification_report(
    y_test,
    predictions,
    target_names=["Active", "Potentially Inactive"],
    zero_division=0
)

st.subheader("Churn Model Performance Report")
st.text(report)

accuracy = accuracy_score(y_test, predictions)

st.subheader("Random Forest Churn Prediction Results")
st.write("Model accuracy:", round(accuracy * 100, 2), "%")
st.write("AUC-ROC Score:", round(auc, 3))
st.write("Precision@Top 20%:", round(precision_top_20, 3))
st.write("Training customers:", len(X_train))
st.write("Testing customers:", len(X_test))

st.subheader("Customer Churn Category Summary")

churn_counts = customer_history["Churn"].value_counts()
churn_counts.index = churn_counts.index.map({
    0: "Active",
    1: "Potentially Inactive"
})

st.bar_chart(churn_counts)