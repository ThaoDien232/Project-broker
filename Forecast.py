import streamlit as st
import pandas as pd
import sys
import os

if st.button("Reload Data"):
    st.cache_data.clear()

st.set_page_config(layout="wide")

@st.cache_data
def load_data():
    try:
        df_index = pd.read_csv("sql/INDEX.csv")
        df_bs = pd.read_csv("sql/BS_security.csv")
        df_is = pd.read_csv("sql/IS_security.csv")
        df_note = pd.read_csv("sql/Note_security.csv")
        df_forecast = pd.read_csv("sql/FORECAST.csv")
        df_turnover = pd.read_excel("turnover.xlsx")
        return df_index, df_bs, df_is, df_note, df_forecast, df_turnover
    except Exception as e:
        st.error(f"Error loading CSV file: {e}")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

df_index, df_bs, df_is, df_note, df_forecast, df_turnover = load_data()

if not df_index.empty:
    # Filter to use only COMGROUPCODE == 'VNINDEX'
    df_index = df_index[df_index['COMGROUPCODE'] == 'VNINDEX']
    brokers = ['SSI', 'HCM', 'VND', 'VCI', 'MBS', 'SHS', 'BSI', 'VIX']
    df_forecast = df_forecast[df_forecast['TICKER'].isin(brokers)]

    selected_broker = st.selectbox(
        "Select Brokers:",
        options=brokers,
        index=0
    )

    # Filter for selected broker and years 2020-2025
    df_broker_hist = df_forecast[(df_forecast['TICKER'] == selected_broker) & (df_forecast['DATE'] >= 2020) & (df_forecast['DATE'] < 2025)]
    df_broker_forecast = df_forecast[(df_forecast['TICKER'] == selected_broker) & (df_forecast['DATE'] == 2025)]


    # Calculate average daily market turnover and trading days for each year from INDEX file
    df_index['Year'] = pd.to_datetime(df_index['TRADINGDATE']).dt.year
    market_turnover_by_year = df_index.groupby('Year').agg(
        avg_daily_turnover=('TOTALVALUE', 'mean'),
        trading_days=('TRADINGDATE', 'nunique')
    ).reset_index()

    # Get trading days for 2025 (or fallback typical value)
    trading_days_2025 = market_turnover_by_year.loc[market_turnover_by_year['Year'] == 2025, 'trading_days']
    if not trading_days_2025.empty:
        trading_days_2025 = int(trading_days_2025.iloc[0])
    else:
        trading_days_2025 = 252  # fallback typical value

    # Sidebar: adjust market turnover per day (VND bn)
    market_share_adj = st.sidebar.slider("Adjust Company Market Share (%)", min_value=0.0, max_value=50.0, value=10.0, step=0.1)
    market_turnover_daily_adj = st.sidebar.slider("Adjust Market Turnover per Day (VND bn)", min_value=0.0, max_value=100000.0, value=50000.0, step=100.0)
    net_fee_adj = st.sidebar.slider("Adjust Net Brokerage Fee (%)", min_value=0.0, max_value=0.1, value=0.0001, step=0.0001)

    # For 2025, use sidebar values to set market share and market turnover
    market_share_2025 = market_share_adj / 100.0
    market_turnover_2025 = market_turnover_daily_adj * trading_days_2025 * 1e9 # annual market turnover in bn VND
    company_turnover_2025 = market_share_2025 * market_turnover_2025 * 2 / 1e9
    net_fee_2025 = net_fee_adj / 100.0
    brokerage_income = company_turnover_2025 * net_fee_2025

    # You can add more calculation functions below and use more sliders for other metrics

    # Ensure TRADINGDATE is datetime
    df_index['TRADINGDATE'] = pd.to_datetime(df_index['TRADINGDATE'])

    # Create annual and quarter columns
    df_index['Annual'] = df_index['TRADINGDATE'].dt.to_period('Y').astype(str)
    df_index['Quarter'] = df_index['TRADINGDATE'].dt.to_period('Q').astype(str)

    # Calculate net foreign trading value (buy - sell, matched)
    df_index['Net_Foreign_Trading'] = df_index['FOREIGNBUYVALUEMATCHED'] - df_index['FOREIGNSELLVALUEMATCHED']

    # Group by month and calculate total and average
    summary_annual = df_index.groupby('Annual').agg(
        Total_Value=('TOTALVALUE', 'sum'),
        Avg_Daily_Value=('TOTALVALUE', 'mean'),
        Total_Net_Foreign_Trading=('Net_Foreign_Trading', 'sum'),
        Avg_Daily_Net_Foreign_Trading=('Net_Foreign_Trading', 'mean'),
        Days=('TRADINGDATE', lambda x: x.nunique())
    ).reset_index()

    summary_annual = summary_annual.sort_values('Annual')

    # Format numbers into billions with thousand separators
    for col in ['Total_Value', 'Avg_Daily_Value', 'Total_Net_Foreign_Trading', 'Avg_Daily_Net_Foreign_Trading']:
        summary_annual[col] = summary_annual[col].apply(lambda x: f"{x/1e9:,.0f}")


    # Group by quarter and calculate total and average
    summary_quarter = df_index.groupby('Quarter').agg(
        Total_Value=('TOTALVALUE', 'sum'),
        Avg_Daily_Value=('TOTALVALUE', 'mean'),
        Total_Net_Foreign_Trading=('Net_Foreign_Trading', 'sum'),
        Avg_Daily_Net_Foreign_Trading=('Net_Foreign_Trading', 'mean'),
        Days=('TRADINGDATE', lambda x: x.nunique())
    ).reset_index()

    summary_quarter = summary_quarter.sort_values('Quarter')

    # Format numbers into billions with thousand separators
    for col in ['Total_Value', 'Avg_Daily_Value', 'Total_Net_Foreign_Trading', 'Avg_Daily_Net_Foreign_Trading']:
        summary_quarter[col] = summary_quarter[col].apply(lambda x: f"{x/1e9:,.0f}")


    # Universal function to build financial statement items
    def build_financial_statement(df, items):
        result = pd.DataFrame({'DATE': df['DATE'].unique()})
        for item_name, filter_func in items.items():
            if item_name == 'Net Fees Income':
                # Sum Net Brokerage Income and Net IB Income for each date
                brokerage_func = items['Net Brokerage Income']
                ib_func = items['Net IB Income']
                result[item_name] = result['DATE'].apply(
                    lambda d: brokerage_func(df[df['DATE'] == d]) + ib_func(df[df['DATE'] == d])
                )
            elif item_name == 'Net Capital Income':
                # Example: sum Net Investment Income and Margin Lending Income
                invest_func = items.get('Net Investment Income', lambda d: 0)
                margin_func = items.get('Margin Lending Income', lambda d: 0)
                result[item_name] = result['DATE'].apply(
                    lambda d: invest_func(df[df['DATE'] == d]) + margin_func(df[df['DATE'] == d])
                )
            else:
                result[item_name] = result['DATE'].apply(lambda d: filter_func(df[df['DATE'] == d]))
        return result

    # Example: Add items as you go
    items = {
    'Net Brokerage Income': lambda d: d.loc[d['KEYCODENAME'] == 'Net Brokerage Income', 'VALUE'].sum(),
    'Market Turnover': None,  # Will be calculated in the function
    'Company Turnover': None,  # Will be calculated in the function
    'Brokerage Market Share': None,  # Will be calculated in the function
    'Net Brokerage Fee': None, # Will be calculated in the function
        'Net IB Income': lambda d: d.loc[d['KEYCODENAME'] == 'Net IB Income', 'VALUE'].sum(),
        'Net other operating income': lambda d: d.loc[d['KEYCODENAME'] == 'Net other operating income', 'VALUE'].sum(),
        'Net Fees Income': None,     # Will be calculated in the function
        'Net Investment Income': lambda d: d.loc[d['KEYCODENAME'] == 'Net Investment Income', 'VALUE'].sum(),
        'Margin Lending Income': lambda d: d.loc[d['KEYCODENAME'] == 'Margin Lending Income', 'VALUE'].sum(),
        'Net Capital Income': None,  # Will be calculated in the function
        'Total Operating Income': None,  # Will be calculated in the function
        'SG&A': lambda d: d.loc[d['KEYCODENAME'] == 'SG&A', 'VALUE'].sum(),
        'FX gain/loss': lambda d: d.loc[d['KEYCODENAME'] == 'FX gain/(loss)', 'VALUE'].sum(),
        'Deposit income': lambda d: d.loc[d['KEYCODENAME'] == 'Deposit income', 'VALUE'].sum(),
        'Gain/loss from affiliates divestment': lambda d: d.loc[d['KEYCODENAME'] == 'Gain/(loss) in affliates divestments', 'VALUE'].sum(),
        'Net other income': lambda d: d.loc[d['KEYCODENAME'] == 'Net other income', 'VALUE'].sum(),
        'Other incomes': None,  # Will be calculated in the function
        'Interest expense': lambda d: d.loc[d['KEYCODENAME'] == 'Interest Expense', 'VALUE'].sum(),
        'PBT': None,  # Will be calculated in the function
        'Tax expense': lambda d: d.loc[d['KEYCODENAME'] == 'Tax expense', 'VALUE'].sum(),
        'Minority expense': lambda d: d.loc[d['KEYCODENAME'] == 'Minority expense', 'VALUE'].sum(),
        'Net Profit': None, # Will be calculated in the function
        # Add more items as needed
    }

    # Universal function to build financial statement items (pivoted)
    def build_financial_statement_pivot(df, items):
        years = sorted(df['DATE'].unique())
        data = {}
        for item_name, filter_func in items.items():
            if item_name == 'Market Turnover':
                # Use INDEX file for historical, sidebar for 2025
                values = []
                for year in years:
                    if year == 2025:
                        values.append(market_turnover_2025)
                    else:
                        mt_row = market_turnover_by_year[market_turnover_by_year['Year'] == year]
                        if not mt_row.empty:
                            avg_daily = mt_row['avg_daily_turnover'].iloc[0]
                            trading_days = mt_row['trading_days'].iloc[0]
                            annual_turnover = avg_daily * trading_days 
                            values.append(annual_turnover)
                        else:
                            values.append('')
                data[item_name] = values
            elif item_name == 'Net Brokerage Income':
                values = []
                for year in years:
                    if year == 2025:
                        values.append(net_fee_2025 * company_turnover_2025 * 2 * 1e9)
                    else:
                        # Use the sum from the DataFrame for historical years
                        values.append(df[df['DATE'] == year].loc[df['KEYCODENAME'] == 'Net Brokerage Income', 'VALUE'].sum())
                data[item_name] = values
            elif item_name == 'Company Turnover':
                broker_col = 'Ticker'
                company_col = 'Company turnover'
                values = []
                debug_rows = []
                for year in years:
                    if year == 2025:
                        values.append(company_turnover_2025)
                        debug_rows.append({'Year': year, 'Value': company_turnover_2025, 'Source': 'Sidebar'})
                    else:
                        row = df_turnover[(df_turnover['Year'] == year) & (df_turnover[broker_col] == selected_broker)]
                        if not row.empty and company_col in row.columns:
                            val = row.iloc[0][company_col]
                        else:
                            val = ''
                        values.append(val)
                        debug_rows.append({'Year': year, 'Value': val, 'Source': 'Excel'})
                data[item_name] = values
            elif item_name == 'Brokerage Market Share':
                # Update these to match your actual Excel column names:
                broker_col = 'Ticker'  # actual column name in your Excel file
                company_col = 'Company turnover'  # e.g. 'CompanyTurnover', 'BrokerTurnover', 'Turnover', 'Value'
                market_col = 'Market turnover'  # e.g. 'MarketTurnover', 'TotalTurnover', 'Turnover', 'Value'

                shares = []
                for year in years:
                    if year == 2025:
                        shares.append(market_share_2025)
                    else:
                        row = df_turnover[(df_turnover['Year'] == year) & (df_turnover[broker_col] == selected_broker)]
                        company_turnover = row[company_col].values[0] if not row.empty else 0
                        market_turnover = row[market_col].values[0] if not row.empty else 0
                        if market_turnover == 0:
                            share = 0
                        else:
                            share = company_turnover / market_turnover / 2
                        shares.append(share)
                data[item_name] = shares
            elif item_name == 'Net Brokerage Fee':
                fee = []
                for i, year in enumerate(years):
                    if year == 2025:
                        fee.append(net_fee_2025)
                    else:
                        # Use already computed values for Net Brokerage Income and Company Turnover
                        nbi = data.get('Net Brokerage Income', [None]*len(years))[i]
                        ct = data.get('Company Turnover', [None]*len(years))[i]
                        try:
                            nbi_val = float(nbi)
                            ct_val = float(ct)
                            if ct_val != 0:
                                fee_val = nbi_val / ct_val / 1e9
                            else:
                                fee_val = None
                        except (TypeError, ValueError):
                            fee_val = None
                        fee.append(fee_val)
                data[item_name] = fee
            elif item_name == 'Net Fees Income':
                brokerage_func = items['Net Brokerage Income']
                ib_func = items['Net IB Income']
                other_func = items['Net other operating income']
                data[item_name] = [
                    brokerage_func(df[df['DATE'] == year]) + ib_func(df[df['DATE'] == year]) + other_func(df[df['DATE'] == year])
                    for year in years
                ]
            elif item_name == 'Net Capital Income':
                invest_func = items.get('Net Investment Income', lambda d: 0)
                margin_func = items.get('Margin Lending Income', lambda d: 0)
                data[item_name] = [
                    invest_func(df[df['DATE'] == year]) + margin_func(df[df['DATE'] == year])
                    for year in years
                ]
            elif item_name == 'Total Operating Income':
                capital = data.get('Net Capital Income', [0]*len(years))
                fees = data.get('Net Fees Income', [0]*len(years))
                nbi = data.get('Net Brokerage Income', [0]*len(years))
                data[item_name] = []
                for i, year in enumerate(years):
                    if year == 2025:
                        # Use net brokerage income forecast for 2025, plus extracted data for other lines
                        toi_2025 = (nbi[i] + (capital[i] if pd.notnull(capital[i]) else 0) + (fees[i] if pd.notnull(fees[i]) else 0))
                        data[item_name].append(toi_2025)
                    else:
                        toi = (capital[i] if pd.notnull(capital[i]) else 0) + (fees[i] if pd.notnull(fees[i]) else 0)
                        data[item_name].append(toi)
            elif item_name == 'Other incomes':
                fx_func = items['FX gain/loss']
                deposit_func = items['Deposit income']
                gain_loss_func = items['Gain/loss from affiliates divestment']
                net_other_func = items['Net other income']
                data[item_name] = [
                    fx_func(df[df['DATE'] == year]) + deposit_func(df[df['DATE'] == year]) +
                    gain_loss_func(df[df['DATE'] == year]) + net_other_func(df[df['DATE'] == year])
                    for year in years
                ]
            elif item_name == 'PBT':
                TOI_vals = data.get('Total Operating Income', [0]*len(years))
                others_vals = data.get('Other incomes', [0]*len(years))
                SGA_func = items['SG&A']
                interest_func = items['Interest expense']
                data[item_name] = []
                for i, year in enumerate(years):
                    if year == 2025:
                        # For 2025, use TOI calculated above
                        pbt_2025 = TOI_vals[i] - SGA_func(df[df['DATE'] == year]) - others_vals[i] - interest_func(df[df['DATE'] == year])
                        data[item_name].append(pbt_2025)
                    else:
                        pbt = TOI_vals[i] - SGA_func(df[df['DATE'] == year]) - others_vals[i] - interest_func(df[df['DATE'] == year])
                        data[item_name].append(pbt)
            elif item_name == 'Net Profit':
                PBT_vals = data.get('PBT', [0]*len(years))
                tax_func = items['Tax expense']
                minority_func = items['Minority expense']
                data[item_name] = [
                    PBT_vals[i]
                    - tax_func(df[df['DATE'] == years[i]])
                    - minority_func(df[df['DATE'] == years[i]])
                    for i in range(len(years))
                ]
            else:
                if filter_func is not None:
                    data[item_name] = [filter_func(df[df['DATE'] == year]) for year in years]
                else:
                    # For custom-calculated items, skip or fill with blanks
                    data[item_name] = ['' for _ in years]
        result = pd.DataFrame(data, index=years).T
        result.columns = [str(y) for y in years]
        result.index.name = "Item"

        # Format all values as billions with thousand separators, except market share (show as percent)
        for item in result.index:
            if item == 'Market Turnover':
                for col in result.columns:
                    val = result.at[item, col]
                    if isinstance(val, str):
                        try:
                            val_num = float(val)
                        except ValueError:
                            val_num = None
                    else:
                        val_num = val
                    if pd.notnull(val_num) and pd.api.types.is_number(val_num):
                        result.at[item, col] = f"{val_num/1e9:,.0f}"
                    else:
                        result.at[item, col] = "-"
            elif item == 'Brokerage Market Share':
                for col in result.columns:
                    val = result.at[item, col]
                    result.at[item, col] = f"{val:.2%}" if pd.notnull(val) and pd.api.types.is_number(val) else "-"
            elif item == 'Net Brokerage Fee':
                for col in result.columns:
                    val = result.at[item, col]
                    if pd.notnull(val) and pd.api.types.is_number(val):
                        result.at[item, col] = f"{val:.3%}"
                    else:
                        result.at[item, col] = "-"
            elif item == 'Company Turnover' or item == 'Market Turnover':
                for i, col in enumerate(result.columns):
                    val = result.at[item, col]
                    # Try to convert string numbers to float
                    if isinstance(val, str):
                        try:
                            val_num = float(val)
                        except ValueError:
                            val_num = None
                    else:
                        val_num = val
                    # Only divide by 1e9 for 2025 (last column)
                    if pd.notnull(val_num) and pd.api.types.is_number(val_num):
                        if col == '2025':
                            result.at[item, col] = f"{val_num:,.0f}"
                        else:
                            result.at[item, col] = f"{val_num:,.0f}"
                    else:
                        result.at[item, col] = "-"
            else:
                for col in result.columns:
                    val = result.at[item, col]
                    if isinstance(val, str):
                        try:
                            val_num = float(val)
                        except ValueError:
                            val_num = None
                    else:
                        val_num = val
                    if pd.notnull(val_num) and pd.api.types.is_number(val_num):
                        result.at[item, col] = f"{val_num/1e9:,.0f}"
                    else:
                        result.at[item, col] = "-"
        return result

    # Merge historical and forecast data
    df_broker_all = pd.concat([df_broker_hist, df_broker_forecast])

    # --- Market Turnover Summary ---
    today = pd.Timestamp.today()
    current_year = today.year
    current_quarter = today.quarter
    # Filter for current year
    df_index_current_year = df_index[df_index['Year'] == current_year]
    # Previous quarters (exclude current quarter)
    prev_quarters = [f"{current_year}Q{q}" for q in range(1, current_quarter)]
    avg_daily_prev_quarters = {}
    for q in prev_quarters:
        df_q = df_index_current_year[df_index_current_year['Quarter'] == q]
        if not df_q.empty:
            avg_daily_prev_quarters[q] = df_q['TOTALVALUE'].mean() / 1e9
    # Current quarter MTD
    current_q_str = f"{current_year}Q{current_quarter}"
    df_current_q = df_index_current_year[df_index_current_year['Quarter'] == current_q_str]
    if not df_current_q.empty:
        mtd = df_current_q[df_current_q['TRADINGDATE'].dt.month == today.month]
        avg_daily_mtd = mtd['TOTALVALUE'].mean() / 1e9 if not mtd.empty else None
    else:
        avg_daily_mtd = None
    # YTD
    avg_daily_ytd = df_index_current_year['TOTALVALUE'].mean() / 1e9 if not df_index_current_year.empty else None

    # Build pivoted market turnover summary table
    turnover_dict = {}
    for q, val in avg_daily_prev_quarters.items():
        # Format with thousand separators
        turnover_dict[q] = f"{val:,.0f}"
    if avg_daily_mtd is not None:
        turnover_dict[f"{current_q_str} MTD"] = f"{avg_daily_mtd:,.0f}"
    if avg_daily_ytd is not None:
        turnover_dict[f"YTD {current_year}"] = f"{avg_daily_ytd:,.0f}"
    # Ensure 2025Q1 is present and aligned
    if f"{current_year}Q1" not in turnover_dict:
        df_q1 = df_index_current_year[df_index_current_year['Quarter'] == f"{current_year}Q1"]
        if not df_q1.empty:
            val_q1 = df_q1['TOTALVALUE'].mean() / 1e9
            turnover_dict[f"{current_year}Q1"] = f"{val_q1:,.0f}"
    # Sort keys for display
    sorted_periods = sorted(turnover_dict.keys(), key=lambda x: (x.split()[0], x))
    turnover_df = pd.DataFrame([turnover_dict])
    turnover_df = turnover_df.rename_axis('Metric').T
    turnover_df.columns = ['Avg Daily Market Turnover (VNDbn)']
    turnover_df.index.name = 'Period'
    turnover_df = turnover_df.loc[sorted_periods]
    st.subheader("Market Turnover Summary")
    st.dataframe(turnover_df.T)

    # --- Financial Statement ---
    st.header(f"{selected_broker} Financial Statement")
    statement_pivot = build_financial_statement_pivot(df_broker_all, items)
    # Hide specified rows from display
    rows_to_hide = [
        'FX gain/loss',
        'Deposit income',
        'Gain/loss from affiliates divestment',
        'Net other income',
        'Other incomes',
        'Tax expense',
        'Minority expense'
    ]
    statement_pivot_display = statement_pivot.drop(rows_to_hide, errors='ignore')

    # --- YoY Growth for PBT ---
    pbt_row = statement_pivot.loc['PBT'] if 'PBT' in statement_pivot.index else None
    yoy_growth_row = None
    if pbt_row is not None:
        years = [str(c) for c in statement_pivot.columns if c.isdigit()]
        yoy_growth_row = []
        yoy_growth_row.append('YoY Growth for PBT')
        for i in range(len(years)):
            if i == 0:
                yoy_growth_row.append('-')
            else:
                prev = pbt_row[years[i-1]]
                curr = pbt_row[years[i]]
                try:
                    prev_val = float(str(prev).replace(",", ""))
                    curr_val = float(str(curr).replace(",", ""))
                    if prev_val != 0:
                        growth = (curr_val - prev_val) / abs(prev_val)
                        yoy_growth_row.append(f"{growth*100:.1f}%")
                    else:
                        yoy_growth_row.append('-')
                except:
                    yoy_growth_row.append('-')
        # Insert YoY row below PBT
        idx = list(statement_pivot_display.index)
        if 'PBT' in idx:
            pbt_idx = idx.index('PBT')
            # Create a new DataFrame with YoY row inserted
            statement_pivot_display = pd.concat([
                statement_pivot_display.iloc[:pbt_idx+1],
                pd.DataFrame([yoy_growth_row[1:]], columns=statement_pivot_display.columns, index=[yoy_growth_row[0]]),
                statement_pivot_display.iloc[pbt_idx+1:]
            ])

    st.dataframe(statement_pivot_display, height=600)

else:
    st.warning("No data available for the selected period.")