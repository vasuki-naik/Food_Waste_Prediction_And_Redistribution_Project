# ============================================================
# GLOBAL FOOD WASTE DASHBOARD
# ============================================================

import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="Global Food Waste Dashboard", layout="wide")

# ------------------------------------------------------------
# LOAD DATA
# ------------------------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("processed_waste_data_final.csv")
    kpi = pd.read_csv("global_kpi_summary.csv")
    return df, kpi

df, kpi = load_data()

st.title("Global Food Waste Analysis & Redistribution Potential")

# ------------------------------------------------------------
# SIDEBAR FILTERS
# ------------------------------------------------------------
countries = df["country"].unique()
selected_country = st.sidebar.selectbox("Select Country", countries)

rescue_rate = st.sidebar.slider(
    "Food Recovery Rate (%)",
    min_value=10,
    max_value=80,
    value=35
) / 100

# ------------------------------------------------------------
# GLOBAL KPIs
# ------------------------------------------------------------
st.header("Global Impact Overview")

col1, col2, col3, col4, col5 = st.columns(5)

col1.metric("Total Waste (tons)", f"{kpi.loc[0,'value']:,.0f}")
col2.metric("Recoverable Waste (tons)", f"{kpi.loc[1,'value']:,.0f}")
col3.metric("Meals Possible", f"{kpi.loc[2,'value']:,.0f}")
col4.metric("People Fed / Year", f"{kpi.loc[3,'value']:,.0f}")
col5.metric("CO₂ Saved (tons)", f"{kpi.loc[4,'value']:,.0f}")

# ------------------------------------------------------------
# COUNTRY FORECAST TREND
# ------------------------------------------------------------
st.header("Waste Trend & Forecast")

country_df = df[df["country"] == selected_country]

fig_trend = px.line(
    country_df,
    x="year",
    y="total_waste_tons",
    color="data_type",
    markers=True,
    title=f"{selected_country} Waste Trend"
)
st.plotly_chart(fig_trend, use_container_width=True)

# ------------------------------------------------------------
# COUNTRY SEGMENT COMPARISON
# ------------------------------------------------------------
st.header("Country Segmentation")

segment_df = (
    df[df["data_type"] == "Actual"]
    .groupby(["country","segment"])["total_waste_tons"]
    .mean()
    .reset_index()
)

fig_segment = px.scatter(
    segment_df,
    x="country",
    y="total_waste_tons",
    color="segment",
    size="total_waste_tons",
    title="Country Waste Segments"
)
st.plotly_chart(fig_segment, use_container_width=True)

# ------------------------------------------------------------
# IMPACT SIMULATOR
# ------------------------------------------------------------
st.header("Food Redistribution Impact Simulator")

country_total_waste = country_df[country_df["data_type"]=="Actual"]["total_waste_tons"].sum()
recoverable = country_total_waste * rescue_rate
meals = recoverable * 1000 / 0.5
people_fed = meals / (365 * 3)

c1, c2, c3 = st.columns(3)
c1.metric("Recoverable Waste (tons)", f"{recoverable:,.0f}")
c2.metric("Meals Generated", f"{meals:,.0f}")
c3.metric("People Fed / Year", f"{people_fed:,.0f}")

# ------------------------------------------------------------
# TOP WASTE COUNTRIES
# ------------------------------------------------------------
st.header("Top Waste Generating Countries")

top_df = (
    df[df["data_type"] == "Actual"]
    .groupby("country")["total_waste_tons"]
    .sum()
    .sort_values(ascending=False)
    .head(10)
    .reset_index()
)

fig_top = px.bar(
    top_df,
    x="country",
    y="total_waste_tons",
    title="Top 10 Countries by Food Waste"
)
st.plotly_chart(fig_top, use_container_width=True)

st.write("Dashboard built using Streamlit.")
