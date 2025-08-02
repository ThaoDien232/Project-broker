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

def formatted_table(df):
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
        
        # Format numbers with commas
        formatted_table = pivot_table.map(lambda x: '{:,.0f}'.format(x) if pd.notnull(x) and x != 0 else '0')
        
        return formatted_table
    else:
        return df

def fetch_historical_price(ticker: str) -> pd.DataFrame:
    """Fetch stock historical price and volume data from TCBS API"""
    
    # TCBS API endpoint for historical data
    url = "https://apipubaws.tcbs.com.vn/stock-insight/v1/stock/bars-long-term"
    
    # Parameters for stock data
    params = {
        "ticker": ticker,
        "type": "stock",
        "resolution": "D",  # Daily data
    }
    
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        
        if 'data' in data and data['data']:
            df = pd.DataFrame(data['data'])
            df['tradingDate'] = pd.to_datetime(df['tradingDate'])
            return df
        else:
            return pd.DataFrame()
    except Exception as e:
        st.error(f"Error fetching price data for {ticker}: {str(e)}")
        return pd.DataFrame()

@st.cache_data
def get_current_prices(tickers):
    """Get current prices for multiple tickers"""
    prices = {}
    for ticker in tickers:
        price_data = fetch_historical_price(ticker)
        if not price_data.empty:
            # Get the most recent closing price
            latest_price = price_data.iloc[-1]['close']
            prices[ticker] = latest_price
        else:
            prices[ticker] = None
    return prices

def calculate_profit_loss(df, current_prices):
    """Calculate profit/loss for each position"""
    df_calc = df.copy()
    
    # Add current price column
    df_calc['Current_Price'] = df_calc['Ticker'].map(current_prices)
    
    # Calculate total market value using current prices
    if 'Volume' in df_calc.columns and 'Current_Price' in df_calc.columns:
        df_calc['Current_Market_Value'] = df_calc['Volume'] * df_calc['Current_Price']
    
    # Calculate profit/loss vs FVTPL value (assuming this is the book value)
    if 'FVTPL value' in df_calc.columns and 'Current_Market_Value' in df_calc.columns:
        df_calc['Profit_Loss'] = df_calc['Current_Market_Value'] - df_calc['FVTPL value']
        df_calc['Profit_Loss_Pct'] = (df_calc['Profit_Loss'] / df_calc['FVTPL value'] * 100).round(2)
    
    return df_calc

st.title("Prop Book")

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
    
    # Add view selection
    view_type = st.radio(
        "Select View:",
        ["Historical Data", "Current Profit/Loss Analysis"],
        horizontal=True
    )
    
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

    if view_type == "Current Profit/Loss Analysis":
        # Get unique tickers for price fetching
        unique_tickers = filtered_df['Ticker'].unique().tolist()
        
        with st.spinner("Fetching current market prices..."):
            current_prices = get_current_prices(unique_tickers)
        
        # Calculate profit/loss for the latest quarter data
        latest_quarter = max(selected_quarters) if selected_quarters else quarters[-1]
        latest_data = filtered_df[filtered_df['Quarter'] == latest_quarter]
        
        if not latest_data.empty:
            profit_df = calculate_profit_loss(latest_data, current_prices)
            
            st.subheader(f"{selected_brokers} - Profit/Loss Analysis ({latest_quarter})")
            
            # Display summary metrics
            if not profit_df['Profit_Loss'].isna().all():
                col1, col2, col3 = st.columns(3)
                
                total_book_value = profit_df['FVTPL value'].sum()
                total_market_value = profit_df['Current_Market_Value'].sum()
                total_profit_loss = profit_df['Profit_Loss'].sum()
                
                with col1:
                    st.metric("Total Book Value", f"{total_book_value:,.0f}")
                with col2:
                    st.metric("Total Market Value", f"{total_market_value:,.0f}")
                with col3:
                    profit_pct = (total_profit_loss / total_book_value * 100) if total_book_value > 0 else 0
                    st.metric("Total P&L", f"{total_profit_loss:,.0f}", f"{profit_pct:.2f}%")
            
            # Display detailed table
            display_columns = ['Ticker', 'FVTPL value', 'Volume', 'Current_Price', 
                             'Current_Market_Value', 'Profit_Loss', 'Profit_Loss_Pct']
            available_columns = [col for col in display_columns if col in profit_df.columns]
            
            st.dataframe(
                profit_df[available_columns].style.format({
                    'FVTPL value': '{:,.0f}',
                    'Current_Price': '{:,.0f}',
                    'Current_Market_Value': '{:,.0f}',
                    'Profit_Loss': '{:,.0f}',
                    'Profit_Loss_Pct': '{:.2f}%'
                }),
                use_container_width=True
            )
        else:
            st.warning("No data available for the selected criteria.")
    
    else:
        # Display the historical filtered table
        st.subheader(f"{selected_brokers} Prop Book")
        st.dataframe(formatted_table(filtered_df), use_container_width=True)

# Main application
def main():
    display_prop_book_table()

if __name__ == "__main__":
    main()
