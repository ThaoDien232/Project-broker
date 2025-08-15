import streamlit as st
from utils.db import run_query

@st.cache_data(ttl=600)  # cache for 10 minutes
def get_broker_history(broker: str, start_date: str):
    sql = """
        SELECT trade_date, metric, value
        FROM broker_history
        WHERE broker = :broker AND trade_date >= :start_date
        ORDER BY trade_date;
    """
    return run_query(sql, {"broker": broker, "start_date": start_date})
