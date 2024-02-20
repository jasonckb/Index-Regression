import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go
import requests
from io import BytesIO
import yfinance as yf
from datetime import datetime
import tensorflow as tf
from keras.models import load_model
import logging

# Main app setup
st.set_page_config(page_title="HSI and SPX Statistical Analysis", layout="wide")
st.title("HSI and SPX Statistical Analysis by Jason Chan")

# Function to apply numeric formatting for decimal numbers without decimal places
def format_decimal(value):
    if pd.isnull(value):
        return None
    try:
        return f"{value:,.0f}"  # Format numbers >= 1 with commas and no decimal places
    except ValueError:
        return value  # If conversion fails, return the original value

# Function to format percentages with one decimal place
def format_percentage(value):
    if pd.isnull(value):
        return None
    try:
        return f"{value:.1%}"  # Format numbers < 1 as percentages with one decimal place
    except ValueError:
        return value  # If conversion fails, return the original value

# Function to format date columns in "MMM-YY" format
def format_date_column(date_val):
    date = pd.to_datetime(date_val, errors='coerce')
    if date is not pd.NaT:
        return date.strftime('%b-%y')
    return date_val

# Function to fetch data from Yahoo Finance and format it
def fetch_and_format_data(ticker):
    # Fetch data
    data = yf.download(ticker, start="1970-01-01")
     
    # Ensure date is the index, then reset it to make it a column
    data.reset_index(inplace=True)

    # Format the Date column
    data['Date'] = data['Date'].dt.strftime('%Y-%m-%d')

    # Sort data by Date in descending order to show the latest data first
    data.sort_values(by='Date', ascending=False, inplace=True)

    return data

# Select ticker based on user choice
index_tickers = {
    "HSI": "^HSI",  # Hang Seng Index ticker symbol
    "SPX": "^GSPC"  # S&P 500 ticker symbol
}

# Sidebar for user inputs
index_choice = st.sidebar.selectbox("Select Market Index", ["HSI", "SPX"])
month_choice = st.sidebar.selectbox("Select Month for Prediction", ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"])
monthly_open = st.sidebar.number_input("Input Prediction Month's Open", min_value=0.0, format="%.2f")

# Fetch and format price history data
df_price_history = fetch_and_format_data(index_tickers[index_choice])

# Dropbox direct download link
dropbox_url = "https://www.dropbox.com/scl/fi/88utrb82zwzlbyo3ljkvl/HSI_SPX-Dashboard.xlsx?rlkey=jobmxd040dyhhs07k9gpmbq6j&dl=1"

# Function to load data from Dropbox
@st.cache_data(show_spinner=False)
def load_data_from_dropbox(url, sheet_name, nrows=None):
    try:
        response = requests.get(url)
        if response.status_code == 200:
            file_stream = BytesIO(response.content)
            df = pd.read_excel(file_stream, sheet_name=sheet_name, nrows=nrows)
            return df
        else:
            st.error(f"Failed to download the file. Status code: {response.status_code}")
            return None
    except Exception as e:
        st.error(f"An error occurred: {e}")
        return None
    
# Format 'Open', 'High', 'Low', and 'Close' columns to 0 decimal places
for column in ['Open', 'High', 'Low', 'Close']:
    df_price_history[column] = df_price_history[column].apply(lambda x: format_decimal(x))

# Display the formatted price history data
with st.expander(f"View {index_choice} Price History", expanded=True):
    st.dataframe(df_price_history[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']])

# Define the sheet names based on the index_choice
stats_sheet_name = 'HSI Stat' if index_choice == "HSI" else 'SPX Stat'
pred_sheet_name = 'HSI Pred' if index_choice == "HSI" else 'SPX Pred'  # This line defines pred_sheet_name

# Load the data
df_price = fetch_and_format_data(index_tickers[index_choice])
df_stats = load_data_from_dropbox(dropbox_url, sheet_name=stats_sheet_name, nrows=14).copy()
df_pred = load_data_from_dropbox(dropbox_url, sheet_name=pred_sheet_name).copy() # Ensure pred_sheet_name is defined before this line

# Verify data is loaded
if df_price is None or df_stats is None or df_pred is None:
    st.error("Failed to load data. Please check the Dropbox link and try again.")
    st.stop()

# Define the columns you want to keep
desired_columns = [
    'Month of Year', 'Average Range (5 years)', 'Average Range', 
    'No. of Rise', 'Avg Rise', 'No. of Fall', 'Avg Fall', 
    'Largest Rise', 'Largest Drop', 'Date of Largest Rise', 
    'Date of Largest Drop', 'Rise: Fall', 'Avg. Up Wick %', 
    'Avg Down Wick%', 'Body %'
]
# Select only the desired columns
df_stats = df_stats[desired_columns]

# Function to perform linear regression and plot results
def plot_index_regression(df, index_name):
    n = len(df)
    X = np.arange(1, n + 1).reshape(-1, 1)  # Independent variable: sequential numbers
    y = np.log(df['Close'].replace(',', '').astype(float)).values  # Logarithmic transformation of the Close price

    # Perform linear regression on the log-transformed data
    model = LinearRegression().fit(X, y)
    y_pred = model.predict(X)

    # Calculate R2 score
    r2 = model.score(X, y)  # Alternatively, you can use r2_score(y, y_pred)

    # Calculate the standard error of the estimate
    residuals = y - y_pred
    sse = np.sum(residuals ** 2)
    degrees_of_freedom = n - 2
    standard_error = np.sqrt(sse / degrees_of_freedom)

    # Confidence levels for the SE bands
    conf_levels = {
        1: "68.27%",
        2: "95.45%",
        3: "99.73%",
    }

    # Plotting the results using Plotly
    fig = go.Figure()

    # Hover font size adjustment
    hover_font_size = 16  # Base size for hover text, adjust as needed

    # Plot actual data points with log-transformed data
    fig.add_trace(go.Scatter(x=df['Date'], y=np.log(df['Close']), mode='lines', name=f'Actual {index_name} Level (Log)',
                             line=dict(color='black'),
                             hovertemplate='%{text}', text=[f'{y:.0f}' for y in df['Close']],
                             hoverlabel=dict(font=dict(size=hover_font_size))))

    # Plot regression line
    fig.add_trace(go.Scatter(x=df['Date'], y=y_pred, mode='lines', name='Regression Line (Log)',
                             line=dict(color='#1C1A1A'),
                             hovertemplate='%{text}', text=[f'{np.exp(y):.0f}' for y in y_pred],
                             hoverlabel=dict(font=dict(size=hover_font_size))))
    # Define colors for the confidence lines
    colors_above = ['#00C9F9', '#004FF9', '#032979']  # For lines above the regression
    colors_below = ['#BFB70F', '#DC810C', '#DC220C']  # For lines below the regression

    # Plot standard error bands
    for n, conf_level in conf_levels.items():
        line_width = 5 if n == 4 else 2  # Make the outermost line thicker
        # Calculate SE bands in log scale
        se_above = y_pred + n * standard_error
        se_below = y_pred - n * standard_error

        # Add SE band above regression line
        fig.add_trace(go.Scatter(x=df['Date'], y=se_above, mode='lines',
                                 name=f'+{n} SE (Log) ({conf_level})',
                                 line=dict(dash='dot', color=colors_above[n-1], width=line_width),
                                 hovertemplate='%{text}', text=[f'{np.exp(y):.0f}' for y in se_above],
                                 hoverlabel=dict(font=dict(size=hover_font_size))))

        # Add SE band below regression line
        fig.add_trace(go.Scatter(x=df['Date'], y=se_below, mode='lines',
                                 name=f'-{n} SE (Log) ({conf_level})',
                                 line=dict(dash='dot', color=colors_below[n-1], width=line_width),
                                 hovertemplate='%{text}', text=[f'{np.exp(y):.0f}' for y in se_below],
                                 hoverlabel=dict(font=dict(size=hover_font_size))))
        
    # Update plot layout with increased font size
    fig.update_layout(
        title={
            'text': f'{index_name} Linear Regression with Confidence Bands - Log Scale(R² = {r2:.2f})',
            'y':0.9,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': {'size': 19}  # Adjust title font size here
        },
        xaxis_title='Date',
        yaxis_title='Log(Close Price)',
        legend_title='Legend',
        xaxis=dict(
            titlefont=dict(size=15),  # Adjust x-axis title font size here
            tickfont=dict(size=15),   # Adjust x-axis tick labels font size here
        ),
        yaxis=dict(
            titlefont=dict(size=15),  # Adjust y-axis title font size here
            tickfont=dict(size=15),   # Adjust y-axis tick labels font size here
        ),
        legend=dict(
            font=dict(size=15)  # Adjust legend font size here
        ),
        yaxis_tickformat='.1f',  # Set tick format for y-axis
        height=800  # Custom height for the chart
    )

    return fig

# Process and format the price history DataFrame
df_price['Date'] = pd.to_datetime(df_price['Date']).dt.strftime('%Y-%m-%d')
df_price = df_price.sort_values(by='Date', ascending=False)

# Apply specific formatting to designated columns
decimal_columns = ['Average Range (5 years)', 'Average Range', 'No. of Rise', 'Avg Rise', 'No of Fall', 'Avg Fall', 'Largest Rise', 'Largest Drop']  # Specify decimal columns
percentage_columns = ['Rise: Fall','Avg. Up Wick %','Avg Down Wick%','Body %']  # Specify percentage columns
date_columns = ['Date of Largest Rise', 'Date of Largest Drop']  # Specify date columns

# Apply formatting
for col in df_stats.columns:
    if col in decimal_columns:
        df_stats[col] = df_stats[col].apply(format_decimal)        
    elif col in percentage_columns:
        df_stats[col] = df_stats[col].apply(format_percentage)
    elif col in date_columns:
        df_stats[col] = df_stats[col].apply(format_date_column)

# Call the regression plot function and display it
if index_choice == "HSI":
    regression_fig = plot_index_regression(df_price, "Hang Seng Index")
    st.plotly_chart(regression_fig, use_container_width=True)
elif index_choice == "SPX":
    regression_fig = plot_index_regression(df_price, "S&P 500")
    st.plotly_chart(regression_fig, use_container_width=True)

# Display the data in Streamlit for Statistics
with st.expander(f"View {index_choice} Statistics", expanded=True):
    st.dataframe(df_stats)

# Find the row with the selected month
month_row = df_pred.loc[df_pred['Month'] == month_choice]

# Sidebar section for Prediction of the Month
st.sidebar.subheader("Prediction of the Month")

# Display the prediction figures under their respective sub-headers
prediction_labels = [
    "Rise Prob", "Fall Prob", "Close @ Avg Rise", "Close @ Avg Fall",
    "Close @ Largest Rise", "Close @ Largest Fall", "Hi @ Largest Hi-Open",
    "Lo @ Largest Lo-Open", "If Bullish Bar, Lo @ (5 Yr. Av)",
    "If Bullish Bar, Hi @ (5 Yr. Av)", "If Bearish Bar, Lo @ (5 Yr. Av)",
    "If Bearish Bar, hi @ (5 Yr. Av)"
]

# Function to add monthly open to prediction values safely
def add_monthly_open(value, monthly_open):
    if pd.isnull(value) or value == '' or isinstance(value, str):
        return value  # Return the original value if it's not numeric
    return value + monthly_open

# If a matching month is found in the predictions, display the values
if not month_row.empty:
    month_row = month_row.iloc[0]  # Extract the matching row as a Series
    for label in prediction_labels:
        # Check if the column exists in the DataFrame to avoid KeyError
        if label in month_row:
            value = month_row[label]
            if pd.isnull(value):
                display_value = "N/A"
            elif "Prob" in label:  # Format probability values as percentages
                display_value = format_percentage(value)
            else:
                # Sum with Monthly Open where applicable and format as decimal
                adjusted_value = add_monthly_open(value, monthly_open)
                display_value = format_decimal(adjusted_value)
            st.sidebar.text(f"{label}: {display_value}")
        else:
            st.sidebar.text(f"{label}: Column not found")
else:
    st.sidebar.warning(f"No prediction data found for {month_choice}.")

# Deep learning Model and Data Processing and Plotting session
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logging.info("Starting to download and load models...")

# Sidebar inputs for deep learning prediction
st.sidebar.subheader("Prediction by Deep Learning")

model_weights = {
    "GRU": st.sidebar.number_input("Weight for GRU", value=2, min_value=0),
    "LSTM": st.sidebar.number_input("Weight for LSTM", value=1, min_value=0),
    "InceptionTime": st.sidebar.number_input("Weight for InceptionTime", value=1, min_value=0)
}
# Assuming a function to preprocess and prepare data for prediction
def preprocess_data(data, base_symbol, ticker2, ticker3):
    """
    Preprocesses the data by cleaning, feature weighting, and scaling.

    Parameters:
    - data: DataFrame containing the historical data.
    - base_symbol: The base ticker symbol for calculating log returns and other metrics.
    - ticker2, ticker3: Additional symbols used for feature weighting.

    Returns:
    - DataFrame after preprocessing (cleaning, feature weighting, and scaling).
    """
    # Handle NaN values by filling forward, then backward to cover all gaps
    data_filled = data.fillna(method='ffill').fillna(method='bfill')

    # Calculate log returns for the Close prices of the base symbol
    # Ensure there's a column named '{base_symbol}_Close' in your DataFrame
    log_return_col_name = f'{base_symbol}_Log_Return'
    close_col_name = f'{base_symbol}_Close'
    if close_col_name in data_filled.columns:
        data_filled[log_return_col_name] = np.log(data_filled[close_col_name] / data_filled[close_col_name].shift(1))
    
    # Applying feature weighting
    weights = {
        log_return_col_name: 1,
        # Adjust these keys based on the actual columns in your DataFrame
        f'{base_symbol}_Volume_Log': 0,
        f'{base_symbol}_OBV_Log_Diff': 0,
        f'{ticker2}_Log_Return': 0,
        f'{ticker3}_Log_Return': 0
    }

    for col, weight in weights.items():
        if col in data_filled.columns:
            data_filled[col] *= weight

    # Select only numeric columns for scaling
    numeric_columns = data_filled.select_dtypes(include=['float64', 'int64']).columns
    scaler = StandardScaler()
    data_filled[numeric_columns] = scaler.fit_transform(data_filled[numeric_columns])

    return data_filled

# Define tickers and the data range
base_symbol = index_choice
ticker2 = 'DX-Y.NYB'  # Example: US Dollar Index
ticker3 = '^VIX'  # Example: CBOE Volatility Index

# Fetch and format the data (assuming fetch_and_format_data is already defined)
df_price_history = fetch_and_format_data(index_tickers[base_symbol])

# Preprocess the data
preprocessed_data = preprocess_data(df_price_history, base_symbol, ticker2, ticker3)

@st.cache(allow_output_mutation=True, show_spinner=True)
def download_model(url):
    """Downloads and loads a Keras model from a specified URL."""
    try:
        response = requests.get(url)
        response.raise_for_status()  # This will raise an HTTPError if the request returned an unsuccessful status code.
        model_file = BytesIO(response.content)
        model = load_model(model_file)
        return model
    except requests.exceptions.RequestException as e:
        st.error(f"Request error: {e}")
        return None
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None
    
def load_models(index_choice):
    model_names = ['GRU', 'LSTM', 'InceptionTime']
    models = {}

    # Define the base URL for your GitHub repository's raw content.
    base_url = "https://raw.githubusercontent.com/jasonckb/Index-Regression/main/"
    
    for model_name in model_names:
        # Adjust the file naming pattern as necessary
        model_filename = f"{model_name}_{index_choice}.h5"
        model_url = f"{base_url}{model_filename}"
        
        print(f"Downloading {model_name} model for {index_choice}...")
        model = download_model(model_url)  # Attempt to download and load the model
        
        if model:  # Check if the model is successfully loaded
            models[model_name] = model
            logging.info(f"Successfully loaded {model_name} model for {index_choice}.")
        else:
            error_message = f"Failed to load {model_name} model for {index_choice}."
            st.error(error_message)
            logging.error(error_message)
    
    return models


# Function to predict with models and calculate weighted average of predictions
def predict_with_models(preprocessed_data, model_weights, models):
    predictions = []
    total_weight = sum(model_weights.values())
    for model_name, model in models.items():
        weight = model_weights[model_name]
        prediction = model.predict(preprocessed_data)  # Ensure preprocessed_data is correctly shaped for the model
        predictions.append(prediction * weight)
    weighted_prediction = np.sum(predictions, axis=0) / total_weight
    return weighted_prediction

# Fetch and format historical data
historical_data = fetch_and_format_data(index_tickers[index_choice])

# Function to plot historical and forecasted prices
def plot_predictions(historical_data, forecasted_data):
    fig = go.Figure()
    # Historical data plot
    fig.add_trace(go.Scatter(x=historical_data['Date'], y=historical_data['Close'], mode='lines', name='Historical Close'))
    # Forecasted data plot
    forecast_dates = [historical_data['Date'].iloc[-1] + timedelta(days=i) for i in range(1, len(forecasted_data) + 1)]
    fig.add_trace(go.Scatter(x=forecast_dates, y=forecasted_data, mode='lines+markers', name='Forecasted Close', line=dict(dash='dot')))
    # Update layout
    fig.update_layout(title='Historical and Forecast Closing Prices', xaxis_title='Date', yaxis_title='Price', legend_title='Legend')
    st.plotly_chart(fig)

# Now we check if historical_data is defined and proceed with processing
if historical_data is not None and not historical_data.empty:
    logging.info("Historical data is valid, proceeding with plotting...")
    try:
        # Assuming preprocess_data, predict_with_models, and plot_predictions are defined
        preprocessed_data = preprocess_data(historical_data)  # Ensure this matches your actual preprocessing logic
        models = load_models(index_choice)  # Load models based on index choice
        forecasted_data = predict_with_models(preprocessed_data, model_weights, models)
        plot_predictions(historical_data, forecasted_data)
    except Exception as e:
        error_message = f"Error during data processing or plotting: {e}"
        st.error(error_message)
        logging.error(error_message)
else:
    st.error("Historical data is missing or invalid. Cannot proceed with plotting.")
    logging.error("Historical data is missing or invalid.")




# Trigger prediction and plotting
if st.sidebar.button("Execute Prediction"):
    # Example of fetching and formatting data
    ticker = "^HSI" if index_choice == "HSI" else "^GSPC"
    historical_data = fetch_and_format_data(ticker)
    
   








