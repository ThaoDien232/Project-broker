import streamlit as st
import pandas as pd
import requests
from datetime import datetime

# Load the prop book data with error handling
@st.cache_data
def load_data():
    try:
        return pd.read_excel("Prop book.xlsx")
    except Exception as e:
        st.error(f"Error loading Excel file: {e}")
        return pd.DataFrame()
df_book = load_data()
def sort_quarters_by_date(quarters):
    def quarter_sort_key(quarter):
        try:
            # Extract quarter number and year from format like "1Q25", "4Q24"
            q_num = int(quarter[0]) 
            year = int(quarter[2:])  # First character is quarter number # Characters after 'Q' are year
            full_year = 2000 + year if year < 50 else 1900 + year
            return full_year * 100 + q_num # Convert to sortable format: year * 100 + quarter
        except:
            return quarter # If format doesn't match, return original for regular sorting
    return sorted(quarters, key=quarter_sort_key)

def get_data(df, ticker, quarters):
    return df[(df['Ticker'] == ticker) & (df['Quarter'].isin(quarters))].copy()

def fetch_historical_price(ticker: str) -> pd.DataFrame:
    """
    Fetch daily stock prices from TCBS API for the given ticker.
    Returns DataFrame with 'tradingDate', 'open', 'high', 'low', 'close', 'volume'.
    """
    url = "https://apipubaws.tcbs.com.vn/stock-insight/v1/stock/bars-long-term"
    params = {
        "ticker": ticker,
        "type": "stock",
        "resolution": "D",
        "from": "0",
        "to": str(int(datetime.now().timestamp()))
    }
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "Accept": "application/json"
    }
    try:
        response = requests.get(url, params=params, headers=headers, timeout=10)
        response.raise_for_status()
        data = response.json()
        if 'data' in data and data['data']:
            df = pd.DataFrame(data['data'])
            # Convert tradingDate to datetime (ISO or ms)
            if 'tradingDate' in df.columns:
                if df['tradingDate'].dtype == 'object' and df['tradingDate'].str.contains('T').any():
                    df['tradingDate'] = pd.to_datetime(df['tradingDate'])
                else:
                    df['tradingDate'] = pd.to_datetime(df['tradingDate'], unit='ms')
            # Only keep relevant columns
            keep = ['tradingDate', 'open', 'high', 'low', 'close', 'volume']
            return df[[col for col in keep if col in df.columns]]
        else:
            print("No data found in response")
            return pd.DataFrame()
    except Exception as e:
        print(f"Error fetching data: {e}")
        return pd.DataFrame()
    
def get_close_price(df: pd.DataFrame, target_date: str = None):
    """
    Get closing price on or before target_date.
    If target_date is None, get the latest available price.
    """
    if df.empty:
        return None
    if target_date:
        target = pd.to_datetime(target_date)
        df2 = df[df['tradingDate'] <= target]
        if df2.empty:
            return None
        return df2.iloc[-1]['close']
    else:
        return df.iloc[-1]['close']

def get_quarter_end_prices(tickers, quarter):
    q_map = {"1Q":"-03-31", "2Q":"-06-30", "3Q":"-09-30", "4Q":"-12-31"}
    q_part, y_part = quarter[:2], quarter[2:]
    date_str = f"{2000+int(y_part)}{q_map.get(q_part, '-12-31')}"
    prices = {}
    for ticker in tickers:
        if ticker.upper() == "OTHERS":
            prices[ticker] = None
        else:
            price_df = fetch_historical_price(ticker)
            prices[ticker] = get_close_price(price_df, date_str)
    return prices

def get_current_prices(tickers):
    prices = {}
    for ticker in tickers:
        if ticker.upper() == "OTHERS":
            prices[ticker] = 0
        else:
            price_df = fetch_historical_price(ticker)
            prices[ticker] = get_close_price(price_df)
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
    main ()
