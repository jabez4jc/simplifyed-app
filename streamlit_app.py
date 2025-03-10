import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import google.generativeai as genai
import tempfile
import os
import json
import numpy as np
import statsmodels.api as sm
import time
from datetime import datetime, timedelta
from fpdf import FPDF

# ------------------------------------------------------
# Data Provider Availability Checks
# ------------------------------------------------------
# Check if yahooquery package is available
try:
    from yahooquery import Ticker as YQTicker
    YQ_AVAILABLE = True
except ImportError:
    YQ_AVAILABLE = False

# ------------------------------------------------------
# Technical Analysis Functions
# ------------------------------------------------------
def compute_rsi(series, period=14):
    """Calculate Relative Strength Index (RSI) for a given price series"""
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def compute_heikin_ashi(df):
    """Convert OHLC data to Heikin-Ashi candlesticks"""
    ha = df.copy()
    ha['HA_Close'] = (df['Open'] + df['High'] + df['Low'] + df['Close']) / 4
    ha['HA_Open'] = 0.0
    ha['HA_Open'].iloc[0] = (df['Open'].iloc[0] + df['Close'].iloc[0]) / 2
    for i in range(1, len(df)):
        ha['HA_Open'].iloc[i] = (ha['HA_Open'].iloc[i-1] + ha['HA_Close'].iloc[i-1]) / 2
    ha['HA_High'] = ha[['HA_Open', 'HA_Close', df['High']]].max(axis=1)
    ha['HA_Low'] = ha[['HA_Open', 'HA_Close', df['Low']]].min(axis=1)
    return ha

def filter_market_hours(df, interval):
    """Since we removed intraday data, just return the dataframe"""
    return df

def generate_pdf(ticker, fig, justification):
    """Generate PDF report containing chart and analysis"""
    # Save chart image to a temporary file
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmpfile:
        fig.write_image(tmpfile.name)
        chart_image_path = tmpfile.name
    # Create PDF
    pdf = FPDF()
    pdf.add_page()
    # Insert chart image (adjust width to page width minus margins)
    pdf.image(chart_image_path, x=10, y=10, w=pdf.w - 20)
    os.remove(chart_image_path)
    # Move cursor below image
    pdf.set_xy(10, pdf.get_y() + 80)
    pdf.set_font("Arial", size=12)
    pdf.multi_cell(0, 10, justification)
    return pdf.output(dest="S").encode("latin1")

# ------------------------------------------------------
# AI Model Configuration
# ------------------------------------------------------
# Access API key from secrets
GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
genai.configure(api_key=GOOGLE_API_KEY)
MODEL_NAME = 'gemini-2.0-flash'
gen_model = genai.GenerativeModel(MODEL_NAME)

# ------------------------------------------------------
# Streamlit App Configuration and Layout
# ------------------------------------------------------
st.set_page_config(layout="wide")
st.title("Simplifyed.in - Advanced Financial Analysis Dashboard")

# ------------------------------------------------------
# Sidebar Configuration
# ------------------------------------------------------
# Help/About section
with st.sidebar.expander("**About**"):
    st.markdown("""
    **Advanced Financial Analysis Dashboard by Simplifyed.in**
    
    This web-based tool provides comprehensive technical analysis reports on stock data and generates AI-powered trading recommendations.
    
    **Key Features:**
    - Data Analysis
        - Multiple stock analysis
        - 5 timeframes (1d to 3mo)
        - 11 technical indicators
        - 3 chart types (Candlestick/Line/Heikin-Ashi)
    - AI Integration
        - AI-powered analysis
        - Trading recommendations
        - Detailed justifications
    - Export Options
        - Charts (PNG)
        - Analysis reports (PDF)
        - Raw data (CSV)

    **Data Sources:**
    - Yahoo Finance
    - Yahoo Query
    
    **Instructions:**
    1. Enter stock tickers as seen on Yahoo Finance.
    2. You can Enable Comparative Analysis if more than 1 ticker is entered.
    3. Select your data provider.
    4. Choose the data interval (default covers the last 1000 bars).
    5. Adjust start and end dates if needed.
    6. Select chart type and technical indicators.
    7. Click **Fetch Data** to load data and generate the report.
    8. In each ticker's tab, use the export buttons to download your chart image, analysis PDF, and raw data CSV.
    """, unsafe_allow_html=True)

# Input parameters
tickers_input = st.sidebar.text_input("Enter Stock Tickers (comma-separated):", "IOC.NS, BPCL.NS, TATAPOWER.NS, NTPC.NS, SBIN.NS, TITAGARH.NS")
tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]

# Enable comparative analysis for multiple tickers
enable_comparative = False
if len(tickers) > 1:
    enable_comparative = st.sidebar.checkbox("Enable Comparative Analysis", value=False)

# Data provider selection (removed TradingView option)
data_provider = st.sidebar.selectbox(
    "Select Data Provider", 
    options=["Yahoo Query (yahooquery)", "Yahoo Finance (yfinance)"],
    index=0
)

# Remove TradingView credentials section
tv_username, tv_password = None, None

# Time interval and date range selection
selected_interval = st.sidebar.selectbox(
    "Select Data Interval", 
    options=["1d", "5d", "1wk", "1mo", "3mo"], 
    index=0  # Default to "1d"
)
interval = selected_interval

# Calculate default date range based on selected interval
interval_deltas = {
    "1d": timedelta(days=1000),
    "5d": timedelta(days=5 * 1000),
    "1wk": timedelta(weeks=1000),
    "1mo": timedelta(days=30 * 1000),
    "3mo": timedelta(days=90 * 1000)
}

# Date range inputs
end_date_default = datetime.today()
start_date_default = end_date_default - interval_deltas[interval]
start_date = st.sidebar.date_input("Start Date", value=start_date_default.date())
end_date = st.sidebar.date_input("End Date", value=end_date_default.date())

# Chart type and indicator selection
chart_type = st.sidebar.selectbox("Select Chart Type", options=["Candlestick", "Line", "Heikin-Ashi"], index=0)

# Available technical indicators
all_indicators = [
    "20-Period SMA", "55-Period EMA", "20-Period Bollinger Bands", "VWAP", 
    "RSI", "MACD", "Linear Regression Channels", "Stochastic RSI", "Stochastic Momentum Index",
    "Support & Resistance", "Volume"
]
indicators = st.sidebar.multiselect("Select Technical Indicators", all_indicators, default=all_indicators)

# ------------------------------------------------------
# Data Fetching Function with Caching
# ------------------------------------------------------
@st.cache_data(show_spinner=True)
def get_stock_data(ticker, start_date, end_date, interval, provider, tv_user=None, tv_pass=None):
    data = None
    try:
        if provider == "Yahoo Query (yahooquery)":
            st.info(f"Fetching {ticker} data from yahooquery...")
            yq = YQTicker(ticker)
            data = yq.history(start=start_date.strftime("%Y-%m-%d"), end=end_date.strftime("%Y-%m-%d"), interval=interval)
            if not data.empty:
                data = data.reset_index().set_index("date")
                data.rename(columns={"open": "Open", "high": "High", "low": "Low", "close": "Close", "volume": "Volume"}, inplace=True)
        elif provider == "Yahoo Finance (yfinance)":
            st.info(f"Fetching {ticker} data from yfinance...")
            data = yf.download(ticker, start=start_date, end=end_date, interval=interval, progress=False)
        else:
            st.error(f"Data provider {provider} not implemented or available.")
    except Exception as e:
        st.error(f"Error fetching data for {ticker}: {e}")
    return data

# ------------------------------------------------------
# Fetch Data Button
# ------------------------------------------------------
if st.sidebar.button("Fetch Data"):
    stock_data = {}
    progress_bar = st.progress(0)
    with st.spinner("Fetching stock data..."):
        for i, ticker in enumerate(tickers):
            data = get_stock_data(ticker, start_date, end_date, interval, data_provider, tv_username, tv_password)
            if data is not None and not data.empty:
                stock_data[ticker] = data
                msg = st.success(f"{ticker} data loaded successfully.")
                time.sleep(3)
                msg.empty()
            else:
                msg = st.error(f"No data found for {ticker} using {data_provider}.")
                time.sleep(3)
                msg.empty()
            progress_bar.progress((i + 1) / len(tickers))
    st.session_state["stock_data"] = stock_data
    if stock_data:
        msg = st.success("All available stock data loaded.")
        time.sleep(3)
        msg.empty()

# ------------------------------------------------------
# Analysis Function
# ------------------------------------------------------
def analyze_ticker(ticker, data):
    start_msg = st.info(f"Starting analysis for {ticker}...")
    # Remove intraday filtering since we no longer have intraday data
    if chart_type == "Heikin-Ashi":
        data = compute_heikin_ashi(data)
    
    overlay_indicators = {"20-Period SMA", "55-Period EMA", "20-Period Bollinger Bands", "VWAP", "Linear Regression Channels", "Support & Resistance"}
    oscillator_indicators = {"RSI", "MACD", "Stochastic RSI", "Stochastic Momentum Index", "Volume"}
    selected_oscillators = [ind for ind in indicators if ind in oscillator_indicators]
    
    num_rows = 1 + len(selected_oscillators)
    row_heights = [0.6] + ([0.4 / len(selected_oscillators)] * len(selected_oscillators)) if num_rows > 1 else [1]
    
    fig = make_subplots(rows=num_rows, cols=1, shared_xaxes=True, vertical_spacing=0.02, row_heights=row_heights)
    
    try:
        if chart_type == "Candlestick":
            fig.add_trace(go.Candlestick(
                x=data.index,
                open=data['Open'],
                high=data['High'],
                low=data['Low'],
                close=data['Close'],
                name="Candlestick"
            ), row=1, col=1)
        elif chart_type == "Line":
            fig.add_trace(go.Scatter(
                x=data.index,
                y=data['Close'],
                mode='lines',
                name="Price"
            ), row=1, col=1)
        elif chart_type == "Heikin-Ashi":
            fig.add_trace(go.Candlestick(
                x=data.index,
                open=data['HA_Open'],
                high=data['HA_High'],
                low=data['HA_Low'],
                close=data['HA_Close'],
                name="Heikin-Ashi"
            ), row=1, col=1)
    except KeyError as e:
        st.error(f"Data format error for {ticker}: missing key {e}")
        start_msg.empty()
        return None, {"action": "Error", "justification": f"Missing key: {e}"}
    
    def add_overlay(indicator):
        try:
            if indicator == "20-Period SMA":
                sma = data['Close'].rolling(window=20).mean()
                fig.add_trace(go.Scatter(x=data.index, y=sma, mode='lines', name='SMA (20)'), row=1, col=1)
            elif indicator == "55-Period EMA":
                ema = data['Close'].ewm(span=55).mean()
                fig.add_trace(go.Scatter(x=data.index, y=ema, mode='lines', name='EMA (55)'), row=1, col=1)
            elif indicator == "20-Period Bollinger Bands":
                sma = data['Close'].rolling(window=20).mean()
                std_val = data['Close'].rolling(window=20).std()
                bb_upper = sma + 2 * std_val
                bb_lower = sma - 2 * std_val
                fig.add_trace(go.Scatter(x=data.index, y=bb_upper, mode='lines', name='BB Upper'), row=1, col=1)
                fig.add_trace(go.Scatter(x=data.index, y=bb_lower, mode='lines', name='BB Lower'), row=1, col=1)
            elif indicator == "VWAP":
                data['VWAP'] = (data['Close'] * data['Volume']).cumsum() / data['Volume'].cumsum()
                fig.add_trace(go.Scatter(x=data.index, y=data['VWAP'], mode='lines', name='VWAP'), row=1, col=1)
            elif indicator == "Linear Regression Channels":
                # Skip if not enough data points
                if len(data) < 2:
                    st.warning(f"Not enough data points for Linear Regression on {ticker}")
                    return
                
                # Calculate linear regression only on valid data
                valid_data = data['Close'].dropna()
                if len(valid_data) < 2:
                    st.warning(f"Not enough valid data points for Linear Regression on {ticker}")
                    return
                
                X = np.arange(len(valid_data)).reshape(-1, 1)
                X = sm.add_constant(X)
                y = valid_data.values
                
                try:
                    model = sm.OLS(y, X).fit()
                    data['LR'] = model.predict(X)
                    residuals = valid_data - data['LR']
                    std_val = residuals.std()
                    if not np.isnan(std_val):
                        data['LR_upper'] = data['LR'] + 2 * std_val
                        data['LR_lower'] = data['LR'] - 2 * std_val
                        
                        fig.add_trace(go.Scatter(x=data.index, y=data['LR'], mode='lines', name='LR Center'), row=1, col=1)
                        fig.add_trace(go.Scatter(x=data.index, y=data['LR_upper'], mode='lines', name='LR Upper'), row=1, col=1)
                        fig.add_trace(go.Scatter(x=data.index, y=data['LR_lower'], mode='lines', name='LR Lower'), row=1, col=1)
                except Exception as e:
                    st.warning(f"Could not calculate Linear Regression for {ticker}: {str(e)}")
                
            elif indicator == "Support & Resistance":
                # Use rolling windows for S&R levels in lower timeframes
                window = 20 if interval != "1d" else len(data)
                
                # Calculate rolling pivot points
                data['Rolling_High'] = data['High'].rolling(window=window).max()
                data['Rolling_Low'] = data['Low'].rolling(window=window).min()
                data['Rolling_Close'] = data['Close'].shift(1)  # Previous close
                
                # Calculate pivot points
                pivot = (data['Rolling_High'] + data['Rolling_Low'] + data['Rolling_Close']) / 3
                r1 = 2 * pivot - data['Rolling_Low']
                s1 = 2 * pivot - data['Rolling_High']
                
                # Add the most recent values as horizontal lines
                if not pivot.empty and not np.isnan(pivot.iloc[-1]):
                    fig.add_hline(y=pivot.iloc[-1], line_dash="dot", 
                                annotation_text="Pivot", row=1, col=1)
                    fig.add_hline(y=r1.iloc[-1], line_dash="dash", 
                                annotation_text="Resistance", row=1, col=1)
                    fig.add_hline(y=s1.iloc[-1], line_dash="dash", 
                                annotation_text="Support", row=1, col=1)
                
        except Exception as e:
            st.error(f"Error adding overlay {indicator} for {ticker}: {str(e)}")
    
    for ind in indicators:
        if ind in overlay_indicators:
            add_overlay(ind)
    
    def add_oscillator(indicator, row):
        try:
            if indicator == "RSI":
                rsi_values = compute_rsi(data['Close'], period=14)
                fig.add_trace(go.Scatter(x=data.index, y=rsi_values, mode='lines', name='RSI (14)'), row=row, col=1)
            elif indicator == "MACD":
                ema12 = data['Close'].ewm(span=12, adjust=False).mean()
                ema26 = data['Close'].ewm(span=26, adjust=False).mean()
                macd_line = ema12 - ema26
                signal_line = macd_line.ewm(span=9, adjust=False).mean()
                fig.add_trace(go.Scatter(x=data.index, y=macd_line, mode='lines', name='MACD Line'), row=row, col=1)
                fig.add_trace(go.Scatter(x=data.index, y=signal_line, mode='lines', name='Signal Line'), row=row, col=1)
            elif indicator == "Stochastic RSI":
                rsi_values = compute_rsi(data['Close'], period=14)
                stoch_period = 14
                min_rsi = rsi_values.rolling(stoch_period).min()
                max_rsi = rsi_values.rolling(stoch_period).max()
                stoch_rsi = (rsi_values - min_rsi) / (max_rsi - min_rsi)
                fig.add_trace(go.Scatter(x=data.index, y=stoch_rsi, mode='lines', name='Stoch RSI'), row=row, col=1)
            elif indicator == "Stochastic Momentum Index":
                n = 14
                k = 3
                data['HH'] = data['High'].rolling(n).max()
                data['LL'] = data['Low'].rolling(n).min()
                data['M'] = data['Close'] - (data['HH'] + data['LL']) / 2
                data['HL'] = (data['HH'] - data['LL']) / 2
                data['SMI_M'] = data['M'].ewm(span=k, adjust=False).mean().ewm(span=k, adjust=False).mean()
                data['SMI_HL'] = data['HL'].ewm(span=k, adjust=False).mean().ewm(span=k, adjust=False).mean()
                data['SMI'] = 100 * (data['SMI_M'] / data['SMI_HL'])
                fig.add_trace(go.Scatter(x=data.index, y=data['SMI'], mode='lines', name='SMI'), row=row, col=1)
            elif indicator == "Volume":
                fig.add_trace(go.Bar(x=data.index, y=data['Volume'], name='Volume'), row=row, col=1)
        except Exception as e:
            st.error(f"Error adding oscillator {indicator} for {ticker}: {e}")
    
    for i, osc in enumerate([ind for ind in indicators if ind in oscillator_indicators]):
        add_oscillator(osc, row=i+2)
        axis_num = i + 2
        fig.layout[f'yaxis{axis_num}'].title = dict(text=osc)
    
    fig.layout.yaxis.title = dict(text="Price")
    
    fig.update_layout(
        title_text=f"{ticker} Analysis | {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')} | Interval: {interval} | Chart Type: {chart_type} | Indicators: {', '.join(indicators)}",
        title_x=0.5,
        xaxis_rangeslider_visible=False,
        template="plotly_dark",
        margin=dict(t=120, b=50, l=50, r=50),
        height=600 + 150 * len(selected_oscillators)
    )
    
    try:
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmpfile:
            fig.write_image(tmpfile.name)
            tmpfile_path = tmpfile.name
        with open(tmpfile_path, "rb") as f:
            image_bytes = f.read()
        os.remove(tmpfile_path)
    except Exception as e:
        st.error(f"Error generating chart image for {ticker}: {e}")
        start_msg.empty()
        return fig, {"action": "Error", "justification": f"Chart image generation failed: {e}"}
    
    image_part = {"data": image_bytes, "mime_type": "image/png"}
    
    analysis_prompt = (
        f"You are a Stock Trader specializing in Technical Analysis at a top financial institution. "
        f"Analyze the {chart_type} chart for {ticker} (with overlays and separate oscillator panels if present). "
        f"This chart covers {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')} with a time frame of {interval} "
        f"and shows the following indicators: {', '.join(indicators)}. "
        f"Provide a detailed justification of your analysis, explaining observed patterns, signals, and trends. "
        f"Then, based solely on the chart, provide a recommendation from: 'Strong Buy', 'Buy', 'Weak Buy', 'Hold', 'Weak Sell', 'Sell', or 'Strong Sell'. "
        f"Return your output as a JSON object with keys: 'action' and 'justification'."
    )
    
    contents = [
        {"role": "user", "parts": [analysis_prompt]},
        {"role": "user", "parts": [image_part]}
    ]
    
    try:
        with st.spinner(f"Calling Gemini API for {ticker} analysis..."):
            response = gen_model.generate_content(contents=contents)
        result_text = response.text
        json_start_index = result_text.find('{')
        json_end_index = result_text.rfind('}') + 1
        if json_start_index != -1 and json_end_index > json_start_index:
            json_string = result_text[json_start_index:json_end_index]
            result = json.loads(json_string)
        else:
            raise ValueError("No valid JSON object found in the response")
    except json.JSONDecodeError as e:
        st.error(f"JSON Parsing error for {ticker}: {e}")
        result = {"action": "Error", "justification": f"JSON Parsing error: {e}. Raw response: {response.text}"}
    except Exception as e:
        st.error(f"Error during Gemini API call for {ticker}: {e}")
        result = {"action": "Error", "justification": f"General error: {e}. Raw response: {response.text}"}
    
    time.sleep(3)
    start_msg.empty()
    
    return fig, result

# ------------------------------------------------------
# Comparative Analysis Section
# ------------------------------------------------------
if enable_comparative and len(tickers) > 1 and "stock_data" in st.session_state and st.session_state["stock_data"]:
    st.subheader("Comparative Analysis")
    comp_fig = make_subplots(rows=1, cols=len(tickers), shared_yaxes=True)
    for i, ticker in enumerate(tickers):
        if ticker in st.session_state["stock_data"]:
            data = st.session_state["stock_data"][ticker]
            try:
                if chart_type == "Candlestick":
                    comp_fig.add_trace(go.Candlestick(
                        x=data.index, open=data['Open'], high=data['High'], low=data['Low'], close=data['Close'], name=ticker
                    ), row=1, col=i+1)
                elif chart_type == "Line":
                    comp_fig.add_trace(go.Scatter(
                        x=data.index, y=data['Close'], mode='lines', name=ticker
                    ), row=1, col=i+1)
                elif chart_type == "Heikin-Ashi":
                    data = compute_heikin_ashi(data)
                    comp_fig.add_trace(go.Candlestick(
                        x=data.index, open=data['HA_Open'], high=data['HA_High'], low=data['HA_Low'], close=data['HA_Close'], name=ticker
                    ), row=1, col=i+1)
            except Exception as e:
                st.error(f"Error adding comparative chart for {ticker}: {e}")
    comp_fig.update_layout(
        title_text=f"Comparative Analysis | {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')} | Interval: {interval} | Chart Type: {chart_type}",
        title_x=0.5,
        template="plotly_dark"
    )
    st.plotly_chart(comp_fig)

# ------------------------------------------------------
# Overall Analysis Tabs
# ------------------------------------------------------
if "stock_data" in st.session_state and st.session_state["stock_data"]:
    tab_names = ["Overall Summary"] + list(st.session_state["stock_data"].keys())
    tabs = st.tabs(tab_names)
    
    overall_results = []
    # The Overall Summary tab remains export-free.
    for i, ticker in enumerate(st.session_state["stock_data"]):
        with st.spinner(f"Analyzing {ticker}..."):
            data = st.session_state["stock_data"][ticker]
            fig, result = analyze_ticker(ticker, data)
            if fig is None:
                overall_results.append({"Stock": ticker, "Recommendation": "Error"})
                continue
            overall_results.append({"Stock": ticker, "Recommendation": result.get("action", "N/A")})
        with tabs[i+1]:
            st.subheader(f"Analysis for {ticker}")
            st.plotly_chart(fig)
            st.write("**Detailed Justification:**")
            st.write(result.get("justification", "No justification provided."))
            st.markdown("### Export Options")
            # Export Chart Image
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmpfile:
                fig.write_image(tmpfile.name)
                tmpfile.seek(0)
                png_data = tmpfile.read()
            st.download_button("Download Chart Image (PNG)", data=png_data, file_name=f"{ticker}_chart.png", mime="image/png")
            # Export PDF (chart and justification)
            pdf_data = generate_pdf(ticker, fig, result.get("justification", "No justification provided."))
            st.download_button("Download Analysis as PDF", data=pdf_data, file_name=f"{ticker}_analysis.pdf", mime="application/pdf")
            # Export Data as CSV
            csv_data = st.session_state["stock_data"][ticker].to_csv(index=True).encode('utf-8')
            st.download_button("Download Data (CSV)", data=csv_data, file_name=f"{ticker}_data.csv", mime="text/csv")
    with tabs[0]:
        st.subheader("Overall Structured Recommendations")
        df_summary = pd.DataFrame(overall_results)
        st.table(df_summary)
    st.session_state["report_generated"] = True
else:
    st.info("Please fetch stock data using the sidebar.")

# ------------------------------------------------------
# Footer Configuration
# ------------------------------------------------------
footer = """
<style>
    .footer {
        position: fixed;
        bottom: 0;
        left: 0;
        width: 100%;
        background-color: #333;
        color: #f1f1f1;
        text-align: center;
        padding: 10px;
        font-size: 14px;
        border-top: 1px solid #444;
    }
</style>
<div class="footer">
    <p>© 2025 Simplifyed.in | Jabez Vettriselvan | All Rights Reserved.</p>
</div>
"""
st.markdown(footer, unsafe_allow_html=True)
