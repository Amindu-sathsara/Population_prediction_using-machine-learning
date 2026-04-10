# Sri Lanka Population & Crop Consumption Prediction

This project is designed to predict the population of Sri Lanka on a monthly/weekly basis using annual data from MacroTrends, with the ultimate goal of predicting crop consumption (Beans, Beetroot, Cabbage, Carrot, Tomato, etc.).

## 1. Best Way to Create the Dataset

MacroTrends provides **yearly** data. Since you want to predict population **monthly or weekly**, you will face a common data science challenge: how to get monthly data from yearly data.

**The Solution: Interpolation**
1. **Download CSVs:** Download the CSV files for Population, Growth Rate, Birth Rate, Death Rate, Urban/Rural populations from MacroTrends pages.
2. **Clean Headers:** MacroTrends CSVs usually have 14 to 16 lines of text at the top before the actual data starts. You need to skip these rows when reading the data.
3. **Merge:** Merge all datasets on the `date` (Year) column.
4. **Resample & Interpolate:** Use Pandas to resample the yearly data into monthly (`MS` - Month Start) data, and use mathematical interpolation (like `spline` or `linear`) to estimate the months in between the years.

## 2. Best Input Features (Variables)

For a machine learning model, raw population numbers aren't always enough. You need to create "Features". Good input features for this project include:

*   **Temporal Features:** `Year`, `Month`, `Quarter` (Helps the model learn seasonal crop consumption trends later).
*   **Lagged Features (Crucial!):** You cannot predict *December's* population using *December's* birth rate because you won't know the birth rate yet! You must use `Population_Last_Month`, `Birth_Rate_Last_Month`, etc.
*   **Demographic Rates:** `Birth_Rate`, `Death_Rate`, `Growth_Rate`.
*   **Ratios:** `Urban_to_Total_Ratio`, `Rural_to_Total_Ratio`, `Population_Density`.

## 3. Best Practices as a Researcher / Undergraduate

To get top marks and establish a strong research methodology, adhere to the following:

1.  **Acknowledge Assumptions Explicitly:** In your thesis or report, explicitly state: *"Because demographic censuses are conducted annually, monthly population figures were synthetically derived using Spline Interpolation."* Honesty about data limitations is a hallmark of an excellent researcher.
2.  **Prevent Data Leakage:** When doing time-series prediction, NEVER use future information to predict past events, and do not randomly shuffle your Train/Test data. Always split it chronologically (e.g., Train on 1960-2015, Test on 2016-2023).
3.  **Start Simple (Baseline Model):** Do not jump straight to Deep Learning (LSTM) or XGBoost. Create a baseline using a simple Linear Regression or ARIMA. Then, show how much your advanced ML model improves over the baseline.
4.  **Connect to Crop Consumption:** Once your population model works, multiply the projected population by the "Per Capita Consumption Rate" of a specific crop (e.g., kg/person/month) to estimate total crop demand.

## Running the Code
1. Place your downloaded CSV files from MacroTrends into a folder named `data/raw/`.
2. Install dependencies: `pip install -r requirements.txt`
3. Run `python src/data_preparation.py` to merge and interpolate your data.
4. Run `python src/model_training.py` to train the Machine Learning model.
