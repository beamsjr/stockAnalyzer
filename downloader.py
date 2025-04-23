import yfinance as yf
import pandas as pd
import argparse
from datetime import datetime
import os

def download_stock_data(ticker, start_date, end_date, output_file=None):
    """
    Download daily stock data for a specified ticker and date range.
    
    Parameters:
    ----------
    ticker : str
        Stock ticker symbol
    start_date : str
        Start date in YYYY-MM-DD format
    end_date : str
        End date in YYYY-MM-DD format
    output_file : str, optional
        Path to save the CSV file. If None, will use ticker_start_end.csv
    
    Returns:
    -------
    pandas.DataFrame
        DataFrame containing the historical stock data
    """
    # Create a Ticker object
    ticker_obj = yf.Ticker(ticker)
    
    # Download the historical data
    data = ticker_obj.history(start=start_date, end=end_date, interval="1d")
    
    # Check if data was retrieved
    if data.empty:
        print(f"No data found for {ticker} between {start_date} and {end_date}")
        return None
    
    # Reset index to make Date a column
    data = data.reset_index()
    
    # Save to CSV if specified
    if output_file is None:
        output_file = f"stock_data/{ticker}.csv"
    
    data.to_csv(output_file, index=False)
    print(f"Data saved to {output_file}")
    
    return data

def validate_date(date_str):
    """Validate if the date string is in YYYY-MM-DD format"""
    try:
        datetime.strptime(date_str, '%Y-%m-%d')
        return date_str
    except ValueError:
        raise argparse.ArgumentTypeError(f"Invalid date format: {date_str}. Please use YYYY-MM-DD")

def main():
    # Set up command-line arguments
    parser = argparse.ArgumentParser(description='Download daily stock data for a specified ticker and date range')
    parser.add_argument('ticker', type=str, help='Stock ticker symbol (e.g., AAPL)')
    parser.add_argument('--start', '-s', type=validate_date, required=True, help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end', '-e', type=validate_date, required=True, help='End date (YYYY-MM-DD)')
    parser.add_argument('--output', '-o', type=str, help='Output file path (defaults to ticker_start_end.csv)')
    
    args = parser.parse_args()
    
    # Download data
    data = download_stock_data(args.ticker, args.start, args.end, args.output)
    
    # Display summary
    if data is not None:
        print(f"\nSummary for {args.ticker}:")
        print(f"Date range: {args.start} to {args.end}")
        print(f"Number of trading days: {len(data)}")
        print(f"Columns: {', '.join(data.columns)}")
        print("\nFirst few rows:")
        print(data.head())

if __name__ == "__main__":
    main()
