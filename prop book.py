import streamlit as st
import pandas as pd

# Load the prop book data
df_book = pd.read_excel("Prop book.xlsx")

def get_data(df, ticker, quarters):
    return df[(df['Ticker'] == ticker) & (df['Quarter'].isin(quarters))].copy()

st.title("Prop Book Dashboard")
def display_prop_book_table():
    """Display prop book data by broker and quarter"""
    
brokers = sorted(df_book['Broker'].unique())
quarters = sorted(df_book['Quarter'].unique())
    
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
    
    # Display the filtered table
st.subheader(f"{selected_brokers} Prop Book")
st.dataframe(formatted_table(filtered_df), use_container_width=True)

# Main application
def main():
    display_prop_book_table()

if __name__ == "__main__":
    main()