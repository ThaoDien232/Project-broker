import streamlit as st
import pandas as pd
import os
import requests
from datetime import datetime

# Load the prop book data with error handling
@st.cache_data
def load_data():
    try:
        if os.path.exists("Prop book.xlsx"):
            return pd.read_excel("Prop book.xlsx")
        else:
            st.error("Excel file 'Prop book.xlsx' not found. Please ensure it's in the same folder as this app.")
            return pd.DataFrame()
    except Exception as e:
        st.error(f"Error loading Excel file: {str(e)}")
        return pd.DataFrame()

df_book = load_data()

def sort_quarters_by_date(quarters):
    """Sort quarters in chronological order (e.g., 4Q24, 1Q25, 2Q25)"""
    def quarter_sort_key(quarter):
        try:
            # Extract quarter number and year from format like "1Q25", "4Q24"
            q_num = int(quarter[0])  # First character is quarter number
            year = int(quarter[2:])  # Characters after 'Q' are year
            
            # Convert 2-digit year to 4-digit year (assuming 20xx)
            if year < 50:  # Assume years 00-49 are 2000-2049
                full_year = 2000 + year
            else:  # Assume years 50-99 are 1950-1999
                full_year = 1900 + year
            
            # Convert to sortable format: year * 100 + quarter
            return full_year * 100 + q_num
        except:
            # If format doesn't match, return original for regular sorting
            return quarter
    
    return sorted(quarters, key=quarter_sort_key)

def get_data(df, ticker, quarters):
    return df[(df['Ticker'] == ticker) & (df['Quarter'].isin(quarters))].copy()

def formatted_table(df, latest_quarter):
    if df.empty:
        return pd.DataFrame()
    
    # Get numeric columns (excluding Ticker, Broker, Quarter)
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    
    if 'Ticker' in df.columns and 'Quarter' in df.columns and len(numeric_cols) > 0:
        # Let user select which column to display
        if len(numeric_cols) > 1:
            value_col = st.multiselect("Select value column:", numeric_cols)
        else:
            value_col = numeric_cols[0]
        
        # Create pivot table with Ticker as index and Quarter as columns
        pivot_table = df.pivot_table(
            index='Ticker',
            columns='Quarter',
            values=value_col,
            aggfunc='sum',
            fill_value=0
        )
        
        # Handle multi-level columns if they exist
        if isinstance(pivot_table.columns, pd.MultiIndex):
            # Extract just the quarter part (second level) for sorting
            quarter_parts = [col[1] for col in pivot_table.columns]
            sorted_quarters = sort_quarters_by_date(quarter_parts)
            # Reconstruct the multi-level column tuples
            quarter_columns = [(col[0], q) for q in sorted_quarters for col in pivot_table.columns if col[1] == q]
        else:
            # Single level columns
            quarter_columns = sort_quarters_by_date(list(pivot_table.columns))
        
        pivot_table = pivot_table[quarter_columns]
        
        # Get prices for calculations if latest quarter exists
        # Check if we have multi-level columns or single level
        if isinstance(pivot_table.columns, pd.MultiIndex):
            # For multi-level columns, check if latest quarter exists in the second level
            quarter_exists = any(col[1] == latest_quarter for col in pivot_table.columns)
            latest_quarter_col = None
            for col in pivot_table.columns:
                if col[1] == latest_quarter:
                    latest_quarter_col = col
                    break
        else:
            # For single level columns
            quarter_exists = latest_quarter in pivot_table.columns
            latest_quarter_col = latest_quarter
        
        if quarter_exists and latest_quarter_col is not None:
            # Get unique tickers (excluding Others)
            tickers = [t for t in pivot_table.index.tolist() if t.upper() != 'OTHERS']
            
            if tickers:
                try:
                    st.write(f"Debug: Fetching prices for {len(tickers)} tickers for quarter {latest_quarter}")
                    # Fetch quarter-end and current prices
                    quarter_prices = get_quarter_end_prices(tickers, latest_quarter)
                    current_prices = get_current_prices(tickers)
                    
                    st.write(f"Debug: Quarter prices: {quarter_prices}")
                    st.write(f"Debug: Current prices: {current_prices}")
                    
                    # Calculate price change percentage
                    price_changes = {}
                    profit_loss = {}
                    
                    for ticker in pivot_table.index:
                        if ticker.upper() == 'OTHERS':
                            price_changes[ticker] = 0
                            profit_loss[ticker] = 0
                        else:
                            quarter_price = quarter_prices.get(ticker, 0)
                            current_price = current_prices.get(ticker, 0)
                            
                            if quarter_price and current_price and quarter_price != 0:
                                # Price change percentage
                                price_change_pct = ((current_price - quarter_price) / quarter_price) * 100
                                price_changes[ticker] = price_change_pct
                                
                                # Profit/loss calculation
                                latest_value = pivot_table.loc[ticker, latest_quarter_col]
                                if latest_value and latest_value != 0:
                                    volume = latest_value / quarter_price
                                    quarter_market_value = volume * quarter_price
                                    current_market_value = volume * current_price
                                    profit_loss[ticker] = current_market_value - quarter_market_value
                                else:
                                    profit_loss[ticker] = 0
                            else:
                                price_changes[ticker] = 0
                                profit_loss[ticker] = 0
                    
                    st.write(f"Debug: Price changes: {price_changes}")
                    st.write(f"Debug: Profit/loss: {profit_loss}")
                    
                    # Add the new columns - use tuple format for multi-level columns
                    if isinstance(pivot_table.columns, pd.MultiIndex):
                        pivot_table[(f'{latest_quarter}_Price_Change_%', '')] = pivot_table.index.map(price_changes)
                        pivot_table[(f'{latest_quarter}_Profit_Loss', '')] = pivot_table.index.map(profit_loss)
                    else:
                        pivot_table[f'{latest_quarter}_Price_Change_%'] = pivot_table.index.map(price_changes)
                        pivot_table[f'{latest_quarter}_Profit_Loss'] = pivot_table.index.map(profit_loss)
                except Exception as e:
                    st.error(f"Error fetching prices: {str(e)}")
                    # Add empty columns if price fetching fails
                    if isinstance(pivot_table.columns, pd.MultiIndex):
                        pivot_table[(f'{latest_quarter}_Price_Change_%', '')] = 0
                        pivot_table[(f'{latest_quarter}_Profit_Loss', '')] = 0
                    else:
                        pivot_table[f'{latest_quarter}_Price_Change_%'] = 0
                        pivot_table[f'{latest_quarter}_Profit_Loss'] = 0
            else:
                st.write("Debug: No tickers found to fetch prices for")
        else:
            st.write(f"Debug: Latest quarter {latest_quarter} not found in columns: {list(pivot_table.columns)}")
        
        # Sort to put "Others" at the bottom
        if 'Others' in pivot_table.index:
            # Separate "Others" from the rest
            others_row = pivot_table.loc[['Others']]
            other_rows = pivot_table.drop('Others')
            # Sort other rows alphabetically
            other_rows = other_rows.sort_index()
            # Combine with "Others" at the bottom
            pivot_table = pd.concat([other_rows, others_row])
        else:
            # Just sort alphabetically if no "Others"
            pivot_table = pivot_table.sort_index()
        
        # Format numbers with commas (except percentage columns)
        formatted_table = pivot_table.copy()
        for col in formatted_table.columns:
            if '_Price_Change_%' in str(col):
                formatted_table[col] = formatted_table[col].map(lambda x: f'{x:.2f}%' if pd.notnull(x) else '0.00%')
            else:
                formatted_table[col] = formatted_table[col].map(lambda x: '{:,.0f}'.format(x) if pd.notnull(x) and x != 0 else '0')
        
        return formatted_table
    else:
        return df

def fetch_historical_price(ticker: str, end_date: str = None) -> pd.DataFrame:
    """Fetch stock historical price and volume data from TCBS API"""
    
    # Skip "Others" ticker
    if ticker.upper() == "OTHERS":
        return pd.DataFrame()
    
    # TCBS API endpoint for historical data
    url = "https://apipubaws.tcbs.com.vn/stock-insight/v1/stock/bars-long-term"
    
    # Parameters for stock data
    params = {
        "ticker": ticker,
        "type": "stock",
        "resolution": "D",  # Daily data
    }
    
    # Add end date if provided
    if end_date:
        params["to"] = end_date
    
    st.write(f"Debug: API call for {ticker} with params: {params}")
    
    try:
        response = requests.get(url, params=params, timeout=10)
        st.write(f"Debug: API response status for {ticker}: {response.status_code}")
        response.raise_for_status()
        data = response.json()
        
        st.write(f"Debug: API response keys for {ticker}: {list(data.keys()) if isinstance(data, dict) else 'Not a dict'}")
        
        if 'data' in data and data['data']:
            df = pd.DataFrame(data['data'])
            df['tradingDate'] = pd.to_datetime(df['tradingDate'])
            st.write(f"Debug: Got {len(df)} price records for {ticker}")
            return df
        else:
            st.write(f"Debug: No data in API response for {ticker}")
            return pd.DataFrame()
    except Exception as e:
        st.warning(f"Could not fetch price data for {ticker}: {str(e)}")
        return pd.DataFrame()


@st.cache_data
def get_quarter_end_prices(tickers, quarter):
    """Get prices at the end of a specific quarter"""
    prices = {}
    
    # Convert quarter to end date (approximate)
    quarter_end_dates = {
        "1Q": "-03-31",
        "2Q": "-06-30", 
        "3Q": "-09-30",
        "4Q": "-12-31"
    }
    
    # Extract year and quarter from format like "1Q25", "4Q24"
    q_part = quarter[:2]  # "1Q", "2Q", etc.
    year_part = quarter[2:]  # "25", "24", etc. 
    # Convert 2-digit year to 4-digit year
    full_year = 2000 + int(year_part)
    end_date = str(full_year) + quarter_end_dates.get(q_part, "-12-31")
    
    st.write(f"Debug: Looking for prices at quarter end: {end_date}")
    
    for ticker in tickers:
        if ticker.upper() == "OTHERS":
            prices[ticker] = 0  # Set Others to 0 as requested
            continue
            
        st.write(f"Debug: Fetching quarter-end price for {ticker}")
        price_data = fetch_historical_price(ticker, end_date)
        if not price_data.empty:
            # Get the closest price to the quarter end date
            price_data = price_data.sort_values('tradingDate')
            latest_price = price_data.iloc[-1]['close']
            prices[ticker] = latest_price
            st.write(f"Debug: {ticker} quarter-end price: {latest_price}")
        else:
            prices[ticker] = None
            st.write(f"Debug: No quarter-end price data for {ticker}")
    return prices

@st.cache_data
def get_current_prices(tickers):
    """Get current/latest prices for multiple tickers"""
    prices = {}
    for ticker in tickers:
        if ticker.upper() == "OTHERS":
            prices[ticker] = 0  # Set Others to 0 as requested
            continue
            
        st.write(f"Debug: Fetching current price for {ticker}")
        price_data = fetch_historical_price(ticker)  # No end_date = latest data
        if not price_data.empty:
            # Get the most recent closing price
            latest_price = price_data.iloc[-1]['close']
            prices[ticker] = latest_price
            st.write(f"Debug: {ticker} current price: {latest_price}")
        else:
            prices[ticker] = None
            st.write(f"Debug: No current price data for {ticker}")
    return prices

def calculate_profit_loss(df, quarter_prices, current_prices, quarter):
    """Calculate profit/loss from quarter-end to current prices"""
    df_calc = df.copy()
    
    # Add quarter-end price column
    df_calc['Quarter_End_Price'] = df_calc['Ticker'].map(quarter_prices)
    df_calc['Current_Price'] = df_calc['Ticker'].map(current_prices)
    
    # Calculate volume using quarter-end prices (volume at quarter end)
    df_calc['Volume'] = df_calc.apply(lambda row: 
        0 if row['Ticker'].upper() == 'OTHERS' or pd.isna(row['Quarter_End_Price']) or row['Quarter_End_Price'] == 0
        else row['FVTPL value'] / row['Quarter_End_Price'], axis=1)
    
    # Calculate quarter-end market value
    df_calc['Quarter_End_Market_Value'] = df_calc['Volume'] * df_calc['Quarter_End_Price'].fillna(0)
    
    # Calculate current market value using the same volume but current prices
    df_calc['Current_Market_Value'] = df_calc['Volume'] * df_calc['Current_Price'].fillna(0)
    
    # Calculate profit/loss from quarter-end to current (not vs FVTPL value)
    df_calc['Profit_Loss'] = df_calc['Current_Market_Value'] - df_calc['Quarter_End_Market_Value']
    df_calc['Profit_Loss_Pct'] = df_calc.apply(lambda row:
        0 if row['Quarter_End_Market_Value'] == 0 else (row['Profit_Loss'] / row['Quarter_End_Market_Value'] * 100), axis=1).round(2)
    
    return df_calc

st.title("Prop Book Dashboard")

# Add refresh button
col1, col2 = st.columns([3, 1])
with col2:
    if st.button("🔄 Refresh Data"):
        st.cache_data.clear()
        st.rerun()

def display_prop_book_table():
    """Display prop book data by broker and quarter"""
    
    brokers = sorted(df_book['Broker'].unique())
    quarters = sort_quarters_by_date(df_book['Quarter'].unique())
    
    # Broker selection
    selected_brokers = st.selectbox(
        "Select Brokers:",
        options=brokers,
        index=0
    )
    
    # Quarter selection
    selected_quarters = st.multiselect(
        "Select Quarters:",
        options=quarters,
        default=quarters
    )
    
    # Filter the data
    filtered_df = df_book.copy()
    
    if selected_brokers and 'Broker' in df_book.columns:
        filtered_df = filtered_df[filtered_df['Broker'] == (selected_brokers)]
        
    if selected_quarters and 'Quarter' in df_book.columns:
        filtered_df = filtered_df[filtered_df['Quarter'].isin(selected_quarters)]

    # Get the latest quarter for additional calculations
    latest_quarter = max(selected_quarters) if selected_quarters else quarters[-1]
    
    # Display the prop book table with additional columns
    st.subheader(f"{selected_brokers} Prop Book")
    
    with st.spinner("Loading data and calculating price changes..."):
        formatted_df = formatted_table(filtered_df, latest_quarter)
        st.dataframe(formatted_df, use_container_width=True)

# Main application
def main():
    display_prop_book_table()

if __name__ == "__main__":
    main()
