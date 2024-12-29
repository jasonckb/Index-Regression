import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import plotly.graph_objects as go
import requests
from io import BytesIO
import yfinance as yf
from datetime import datetime

# Main app setup
st.set_page_config(page_title="HSI and SPX Statistical Analysis", layout="wide")
st.title("HSI and SPX Statistical Analysis by Jason Chan")

# Function to apply numeric formatting for decimal numbers without decimal places
def format_decimal(value):
    if pd.isnull(value):
        return None
    try:
        if isinstance(value, str):
            value = float(value.replace(',', ''))  # Convert string to float if needed
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
@st.cache_data(show_spinner=False)
def fetch_and_format_data(ticker):
    try:
        # Fetch data with error handling
        data = yf.download(ticker, start="1970-01-01")
        if data.empty:
            st.error(f"No data received for {ticker}")
            return None
        
        # Convert to DataFrame and reset index
        df = pd.DataFrame(data).reset_index()
        
        # Format the Date column
        df['Date'] = pd.to_datetime(df['Date']).dt.strftime('%Y-%m-%d')
        
        # Sort data by Date in descending order
        df = df.sort_values(by='Date', ascending=False)
        
        # Ensure numeric columns are float type
        numeric_columns = ['Open', 'High', 'Low', 'Close']
        for col in numeric_columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            
        # Drop any rows with NaN values
        df = df.dropna(subset=numeric_columns)
        
        return df
    except Exception as e:
        st.error(f"Error fetching data: {str(e)}")
        return None

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
    
# Format and display the price history data
if df_price_history is not None:
    # Create a copy for display
    display_df = pd.DataFrame(df_price_history).copy()
    
    # Format numeric columns
    for column in ['Open', 'High', 'Low', 'Close']:
        display_df[column] = display_df[column].apply(
            lambda x: format_decimal(x) if pd.notnull(x) and isinstance(x, (int, float)) else x
        )
    
    # Display the formatted data
    with st.expander(f"View {index_choice} Price History", expanded=True):
        st.dataframe(display_df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']])

# Define the sheet names based on the index_choice
stats_sheet_name = 'HSI Stat' if index_choice == "HSI" else 'SPX Stat'
pred_sheet_name = 'HSI Pred' if index_choice == "HSI" else 'SPX Pred'  # This line defines pred_sheet_name

# Load the data with explicit DataFrame creation
try:
    # Load price data
    df_price = fetch_and_format_data(index_tickers[index_choice])
    if df_price is None:
        st.error("Failed to load price data.")
        st.stop()
    
    # Load stats data
    stats_data = load_data_from_dropbox(dropbox_url, sheet_name=stats_sheet_name, nrows=14)
    if stats_data is None:
        st.error("Failed to load statistics data.")
        st.stop()
    df_stats = pd.DataFrame(stats_data)
    
    # Load prediction data
    pred_data = load_data_from_dropbox(dropbox_url, sheet_name=pred_sheet_name)
    if pred_data is None:
        st.error("Failed to load prediction data.")
        st.stop()
    df_pred = pd.DataFrame(pred_data)
    
    # Convert stats columns to appropriate types
    for col in df_stats.columns:
        if col not in ['Month of Year', 'Date of Largest Rise', 'Date of Largest Drop']:
            df_stats[col] = pd.to_numeric(df_stats[col], errors='coerce')
    
    # Convert prediction columns to appropriate types
    for col in df_pred.columns:
        if col != 'Month':
            df_pred[col] = pd.to_numeric(df_pred[col], errors='coerce')
            
except Exception as e:
    st.error(f"Error loading data: {str(e)}")
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
    # Drop any NaN values
    df = df.dropna(subset=['Close'])
    
    # Sort by date in ascending order for regression
    df = df.sort_values('Date', ascending=True)
    
    n = len(df)
    X = np.arange(1, n + 1).reshape(-1, 1)  # Independent variable: sequential numbers
    y = np.log(df['Close']).values  # Logarithmic transformation of the Close price

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

    # Plot actual data points with actual values
    fig.add_trace(go.Scatter(x=df['Date'], y=df['Close'], mode='lines', name=f'Actual {index_name} Level',
                             line=dict(color='black'),
                             hovertemplate='%{y:,.0f}<extra></extra>',
                             hoverlabel=dict(font=dict(size=hover_font_size))))

    # Plot regression line with actual values
    fig.add_trace(go.Scatter(x=df['Date'], y=np.exp(y_pred), mode='lines', name='Regression Line',
                             line=dict(color='#1C1A1A'),
                             hovertemplate='%{y:,.0f}<extra></extra>',
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

        # Add SE band above regression line with actual values
        fig.add_trace(go.Scatter(x=df['Date'], y=np.exp(se_above), mode='lines',
                                name=f'+{n} SE ({conf_level})',
                                line=dict(dash='dot', color=colors_above[n-1], width=line_width),
                                hovertemplate='%{y:,.0f}<extra></extra>',
                                hoverlabel=dict(font=dict(size=hover_font_size))))

        # Add SE band below regression line with actual values
        fig.add_trace(go.Scatter(x=df['Date'], y=np.exp(se_below), mode='lines',
                                name=f'-{n} SE ({conf_level})',
                                line=dict(dash='dot', color=colors_below[n-1], width=line_width),
                                hovertemplate='%{y:,.0f}<extra></extra>',
                                hoverlabel=dict(font=dict(size=hover_font_size))))
        
    # Update plot layout with increased font size and log scale toggle
    fig.update_layout(
        title={
            'text': f'{index_name} Linear Regression with Confidence Bands (R² = {r2:.2f})',
            'y':0.9,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': {'size': 19}  # Adjust title font size here
        },
        updatemenus=[
            dict(
                type="buttons",
                direction="right",
                x=0.7,
                y=1.2,
                showactive=True,
                buttons=[
                    dict(
                        label="Linear Scale",
                        method="relayout",
                        args=[{"yaxis.type": "linear"}]
                    ),
                    dict(
                        label="Log Scale",
                        method="relayout",
                        args=[{"yaxis.type": "log"}]
                    )
                ]
            )
        ],
        xaxis_title='Date',
        yaxis_title='Close Price',
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
        yaxis_tickformat=',',  # Set tick format for y-axis to show commas for thousands
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
