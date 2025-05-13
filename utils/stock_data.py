import pandas as pd
import requests
from datetime import datetime
import os
import socket
import yfinance as yf  # Fallback option

# API configuration
ALPHA_VANTAGE_API_KEY = os.getenv('ALPHA_VANTAGE_API_KEY', 'ILQY756OQ7BWF6J1')
MONEYCONTROL_API = "https://www.moneycontrol.com/mc/widget/basicchart/get_chart_value"

def check_connection():
    try:
        socket.create_connection(("www.google.com", 80), timeout=5)
        return True
    except:
        return False

def fetch_indian_stock_data(ticker, start_date, end_date):
    if not check_connection():
        raise ConnectionError("No internet connection")
    
    # Try Alpha Vantage first
    data = fetch_alpha_vantage(ticker, start_date, end_date)
    if data is not None:
        return data
    
    # Fallback to MoneyControl
    data = fetch_moneycontrol(ticker, start_date, end_date)
    if data is not None:
        return data
    
    # Final fallback to Yahoo Finance
    data = fetch_yfinance(ticker, start_date, end_date)
    if data is not None:
        return data
    
    raise ValueError(f"Could not fetch data for {ticker} from any source")

def fetch_alpha_vantage(ticker, start_date, end_date):
    try:
        symbol = ticker.split('.')[0]  # Extract base symbol
        url = f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={symbol}&outputsize=full&apikey={ALPHA_VANTAGE_API_KEY}"
        
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        if "Time Series (Daily)" not in data:
            return None
        
        df = pd.DataFrame.from_dict(data["Time Series (Daily)"], orient='index')
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()
        df = df.loc[start_date:end_date]
        
        df = df.rename(columns={
            '1. open': 'Open',
            '2. high': 'High',
            '3. low': 'Low',
            '4. close': 'Close',
            '5. volume': 'Volume'
        })
        
        return df.apply(pd.to_numeric)
    except:
        return None

def fetch_moneycontrol(ticker, start_date, end_date):
    try:
        # MoneyControl uses different ticker formats
        mc_ticker_map = {
            'RELIANCE.BSE': 'RELI',
            'TCS.BSE': 'TCS',
            'HDFCBANK.BSE': 'HDFC',
            # Add more mappings as needed
        }
        
        if ticker not in mc_ticker_map:
            return None
            
        params = {
            'classic': 'true',
            'sc_dm': mc_ticker_map[ticker],
            'sc_int': 'day',
            'start_date': start_date.strftime('%d-%m-%Y'),
            'end_date': end_date.strftime('%d-%m-%Y')
        }
        
        response = requests.get(MONEYCONTROL_API, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        if not data.get('g1'):
            return None
            
        df = pd.DataFrame(data['g1'])
        df['Date'] = pd.to_datetime(df['x'], unit='ms')
        df = df.set_index('Date')
        df = df.rename(columns={
            'y1': 'Open',
            'y2': 'High',
            'y3': 'Low',
            'y4': 'Close'
        })
        
        return df[['Open', 'High', 'Low', 'Close']].apply(pd.to_numeric)
    except:
        return None

def fetch_yfinance(ticker, start_date, end_date):
    try:
        # For Yahoo Finance, we need to add .NS for NSE or .BO for BSE
        if ticker.endswith('.BSE'):
            yf_ticker = ticker.replace('.BSE', '.BO')
        else:
            yf_ticker = ticker + '.NS'
            
        data = yf.download(yf_ticker, start=start_date, end=end_date)
        return data if not data.empty else None
    except:
        return None