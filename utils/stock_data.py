import yfinance as yf
import pandas as pd

def fetch_stock_data(ticker: str, start_date: str, end_date: str, interval: str):
    try:
        
        data = yf.download(
            ticker,
            start=start_date,
            end=end_date,
            interval=interval,
            auto_adjust=True,  
            prepost=True,     
            threads=True      
        )
        
        
        expected_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        for col in expected_columns:
            if col not in data.columns:
                raise ValueError(f"Missing expected column: {col}")
        
        # Clean the data
        data = data[expected_columns]  # Keep only essential columns
        data.columns.name = None  # remove any multi-index name
        
        # Forward fill any missing values (but not NaN at start)
        data.ffill(inplace=True)
        data.dropna(inplace=True)
        
        return data
    
    except Exception as e:
        print(f"Error fetching stock data: {e}")
        return None