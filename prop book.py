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
    def key(q):
        try:
            q_num = int(q[0])
            year = int(q[2:])
            full_year = 2000 + year if year < 50 else 1900 + year
            return full_year * 10 + q_num
        except Exception:
            return q
    return sorted(quarters, key=key)

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
        target = pd.to_datetime(target_date).tz_localize('UTC')
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
    df_calc['Total_Profit_Loss'] = df_calc['Profit_Loss'].sum()
    df_calc['Profit_Loss_Pct'] = df_calc.apply(lambda row:
        0 if row['Quarter_End_Market_Value'] == 0 else (row['Profit_Loss'] / row['Quarter_End_Market_Value'] * 100), axis=1).round(1)
    return df_calc
    
def formatted_table(df, selected_quarters=None):
    if df.empty:
        return pd.DataFrame()
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    value_col = numeric_cols[0] if len(numeric_cols) == 1 else st.selectbox("Select value column:", numeric_cols)
    if selected_quarters is None:
        all_quarters = sort_quarters_by_date(df['Quarter'].unique())
    else:
        all_quarters = sort_quarters_by_date(selected_quarters)
    pivot_table = df.pivot_table(
        index='Ticker',
        columns='Quarter',
        values=value_col,
        aggfunc='sum',
        fill_value=0
    )
    pivot_table = pivot_table.reindex(columns=all_quarters, fill_value=0)
    tickers = [t for t in pivot_table.index if t.upper() not in ['OTHERS', 'PBT']]
    # --- PBT debug
    st.write("Pivot table tickers (should include PBT):", list(pivot_table.index))
    # --- Calculate P/L for each ticker's latest quarter
    profit_dict, pct_dict, quarter_dict = {}, {}, {}
    for t in tickers:
        q_list = [q for q in all_quarters if pivot_table.at[t, q] != 0]
        if not q_list:
            profit_dict[t] = ''
            pct_dict[t] = ''
            quarter_dict[t] = ''
            continue
        q = q_list[-1]
        quarter_dict[t] = q
        q_price = get_quarter_end_prices([t], q)[t]
        c_price = get_current_prices([t])[t]
        val = pivot_table.at[t, q]
        if q_price and c_price and q_price != 0 and val != 0:
            vol = val / q_price
            p_start = vol * q_price
            p_now = vol * c_price
            profit = p_now - p_start
            pct = 0 if p_start == 0 else (profit / p_start * 100)
            profit_dict[t] = profit
            pct_dict[t] = pct
        else:
            profit_dict[t] = ''
            pct_dict[t] = ''
    profit_col = "Profit/Loss (since latest q)"
    pct_col = "% Profit/Loss (since latest q)"
    qtr_col = "Quarter Used for P/L"
    pivot_table[profit_col] = pivot_table.index.map(lambda t: profit_dict.get(t, ''))
    pivot_table[pct_col] = pivot_table.index.map(lambda t: pct_dict.get(t, ''))
    pivot_table[qtr_col] = pivot_table.index.map(lambda t: quarter_dict.get(t, ''))
    # --- Compose table: main, others, pbt
    rows = pivot_table.index.tolist()
    main_rows = pivot_table.drop([r for r in ['Others', 'PBT'] if r in rows])
    concat_list = [main_rows]
    if 'Others' in rows:
        concat_list.append(pivot_table.loc[['Others']])
    if 'PBT' in rows:
        concat_list.append(pivot_table.loc[['PBT']])
    pivot_table = pd.concat(concat_list)
    # --- Total row, excluding PBT/Total
    rows_for_total = [idx for idx in pivot_table.index if idx not in ['PBT', 'Total']]
    total_row = {}
    for col in pivot_table.columns:
        if col in [profit_col, pct_col, qtr_col]:
            total_row[col] = pivot_table.loc[rows_for_total, col].sum() if col == profit_col else ''
        else:
            total_row[col] = pivot_table.loc[rows_for_total, col].sum()
    total_df = pd.DataFrame([total_row], index=["Total"])
    pivot_table = pd.concat([pivot_table, total_df])
    # --- Formatting
    formatted_table = pivot_table.copy()
    for col in formatted_table.columns:
        if "%" in str(col):
            formatted_table[col] = formatted_table[col].apply(lambda x: f"{x:,.1f}%" if pd.notnull(x) and x != '' else "")
        elif "Profit/Loss" in str(col):
            formatted_table[col] = formatted_table[col].apply(lambda x: f"{x:,.1f}" if pd.notnull(x) and x != '' else "")
        else:
            formatted_table[col] = formatted_table[col].apply(lambda x: f"{x:,.1f}" if pd.notnull(x) and x != '' else "")
    return formatted_table


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

    selected_brokers = st.selectbox(
        "Select Brokers:",
        options=brokers,
        index=0
    )
    selected_quarters = st.multiselect(
        "Select Quarters:",
        options=quarters,
        default=quarters
    )
    
    filtered_df = df_book.copy()
    # Always include PBT in the filtered DataFrame
    if selected_brokers and 'Broker' in df_book.columns:
        filtered_df = filtered_df[(filtered_df['Broker'] == selected_brokers) | (filtered_df['Ticker'] == 'PBT')]
    if selected_quarters and 'Quarter' in df_book.columns:
        filtered_df = filtered_df[filtered_df['Quarter'].isin(selected_quarters)]
    st.write("Filtered tickers:", filtered_df['Ticker'].unique())


    # Get the latest quarter chronologically with data for the selected broker or PBT ---
    available_quarters = sort_quarters_by_date(filtered_df['Quarter'].unique())
    latest_quarter = available_quarters[-1] if available_quarters else None
    
    # Display the prop book table with additional columns
    st.subheader(f"{selected_brokers} Prop Book")
    
    with st.spinner("Loading data and calculating price changes..."):
        # Use available_quarters for both display and calculation
        formatted_df = formatted_table(filtered_df, available_quarters)
        st.dataframe(formatted_df, use_container_width=True)

# Main application
def main():
    display_prop_book_table()

if __name__ == "__main__":
    main ()
