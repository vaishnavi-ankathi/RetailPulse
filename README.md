# RetailPulse – Customer Analytics and Demand Forecasting

## Project Overview
RetailPulse is a retail analytics project that uses historical online retail transactions to analyse sales, understand customer purchasing behaviour, identify potentially inactive customers, and estimate future demand.

## Dataset
The project uses the UCI Online Retail II dataset.

The dataset contains transaction information such as invoice number, product description, quantity, invoice date, price, customer ID, and country.

## Project Features
1. Data cleaning and preparation.
2. Sales analysis by year, month, product, and country.
3. Customer segmentation using Recency, Frequency, and Monetary (RFM) analysis.
4. Customer grouping using the K-Means clustering algorithm.
5. Potential customer inactivity prediction using a Random Forest classifier.
6. Sales demand forecasting using Prophet.
7. Inventory risk analysis based on historical sales and quantities.
8. Historical-sales-based reorder quantity estimates.
9. Interactive dashboard with a country filter.

## Technologies Used
- Python
- Pandas
- Streamlit
- Scikit-learn
- Prophet
- Power BI
- Excel

## How to Run the Streamlit Dashboard
1. Install Python.
2. Open the project folder in VS Code.
3. Install the required Python packages.
4. Ensure the dataset file is available at the path used by the application.
5. Open the terminal and run:

   `python -m streamlit run app.py`

6. Open the local address displayed in the terminal.

## Limitations

1. The analysis depends on the quality and completeness of the historical retail dataset.

2. Some records contain missing customer IDs and product descriptions.

3. Customer churn is estimated using purchase inactivity during the final 90 days of the dataset.

4. The churn prediction model may not identify every potentially inactive customer correctly.

5. Demand forecasting is based on historical sales patterns and may not capture unexpected changes in customer demand.

6. Forecast accuracy depends on the evaluation method and the quality of the historical data.

7. Inventory risk analysis uses historical sales and quantity information rather than actual warehouse stock levels.

8. Reorder quantities are estimates based on historical sales and do not represent actual stock replenishment requirements.

9. Customer purchasing behaviour and sales trends may change over time.

10. The results should be used as decision-support information rather than as guaranteed predictions.

## Conclusion
RetailPulse demonstrates how retail transaction data can be used to support sales analysis, customer segmentation, customer inactivity analysis, demand forecasting, and inventory-related decision-making.