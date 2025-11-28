import pandas as pd
import numpy as np
import streamlit as st

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

# BASIC PAGE CONFIG
st.set_page_config(
    page_title="COVID-19 Global Mortality by Age",
    layout="wide"
)

st.title("COVID-19 Global Mortality by Age")
st.markdown(
    """
This dashboard explores **monthly COVID-19 deaths by age group, country and region**,  
based on WHO data. It is designed for **public-health decision makers** who need to
quickly understand **where and for whom** the burden of mortality is highest.
"""
)

# -----------------------------
# LOAD DATA
# -----------------------------
@st.cache_data
def load_data():
    # Make sure this path matches your folder structure:
    data_path = "WHO-COVID-19-global-monthly-death-by-age-data.csv"
    df = pd.read_csv(data_path)

    # Clean and enrich
    df["Deaths"] = df["Deaths"].fillna(0)
    df["Date"] = pd.to_datetime(
        df["Year"].astype(str) + "-" + df["Month"].astype(str) + "-01",
        errors="coerce"
    )

    # Nice labels for age groups if you want them later
    age_map = {
        "0_4": "0–4",
        "5_14": "5–14",
        "15_64": "15–64",
        "65+": "65+"
    }
    df["Agegroup_pretty"] = df["Agegroup"].map(age_map).fillna(df["Agegroup"])

    return df


df = load_data()

# -----------------------------
# SIDEBAR – ROLE & FILTERS
# -----------------------------
st.sidebar.title("⚙️ Controls")

role = st.sidebar.selectbox(
    "View as:",
    [
        "WHO regional analyst",
        "National health minister",
        "Hospital system planner",
        "General policy analyst"
    ]
)

st.sidebar.markdown("---")

regions = ["All regions"] + sorted(df["Who_region"].dropna().unique().tolist())
region_selected = st.sidebar.selectbox("WHO Region", regions)

income_levels = ["All income levels"] + sorted(df["Wb_income"].dropna().unique().tolist())
income_selected = st.sidebar.selectbox("World Bank Income Group", income_levels)

countries_all = sorted(df["Country"].dropna().unique().tolist())
countries_selected = st.sidebar.multiselect(
    "Countries (optional)",
    options=countries_all,
    default=[]
)

agegroups_all = sorted(df["Agegroup_pretty"].unique().tolist())
agegroups_selected = st.sidebar.multiselect(
    "Age groups",
    options=agegroups_all,
    default=agegroups_all
)

year_min, year_max = int(df["Year"].min()), int(df["Year"].max())
year_range = st.sidebar.slider(
    "Year range",
    min_value=year_min,
    max_value=year_max,
    value=(year_min, year_max),
    step=1
)

st.sidebar.markdown("---")
show_raw = st.sidebar.checkbox("Show filtered data table", value=False)

# -----------------------------
# APPLY FILTERS
# -----------------------------
filtered = df.copy()

# Region filter
if region_selected != "All regions":
    filtered = filtered[filtered["Who_region"] == region_selected]

# Income filter
if income_selected != "All income levels":
    filtered = filtered[filtered["Wb_income"] == income_selected]

# Country filter
if countries_selected:
    filtered = filtered[filtered["Country"].isin(countries_selected)]

# Age filter
if agegroups_selected:
    filtered = filtered[filtered["Agegroup_pretty"].isin(agegroups_selected)]

# Year filter
filtered = filtered[(filtered["Year"] >= year_range[0]) & (filtered["Year"] <= year_range[1])]

# Safety check
if filtered.empty:
    st.error("No data for the current filter selection. Try widening your filters.")
    st.stop()

# -----------------------------
# TOP KPI METRICS
# -----------------------------
total_deaths = filtered["Deaths"].sum()

max_date = filtered["Date"].max()
last_12_months_start = max_date - pd.DateOffset(months=12)
recent = filtered[filtered["Date"] >= last_12_months_start]
recent_deaths = recent["Deaths"].sum()

age_summary = filtered.groupby("Agegroup_pretty")["Deaths"].sum().sort_values(ascending=False)
top_agegroup = age_summary.index[0]
top_age_deaths = age_summary.iloc[0]

col1, col2, col3 = st.columns(3)

col1.metric(
    "Total deaths (selected filters)",
    f"{int(total_deaths):,}"
)

col2.metric(
    "Deaths in last 12 months (filters)",
    f"{int(recent_deaths):,}"
)

col3.metric(
    "Most affected age group",
    f"{top_agegroup}",
    f"{int(top_age_deaths):,} deaths"
)

st.markdown("---")

# -----------------------------
# ROW 1 – PLOTLY (TIME SERIES)
# -----------------------------
st.subheader("Monthly deaths over time by age group (Plotly)")

ts = (
    filtered
    .groupby(["Date", "Agegroup_pretty"], as_index=False)["Deaths"]
    .sum()
    .sort_values("Date")
)

fig_ts = px.line(
    ts,
    x="Date",
    y="Deaths",
    color="Agegroup_pretty",
    markers=True,
    labels={
        "Date": "Month",
        "Deaths": "Deaths",
        "Agegroup_pretty": "Age group"
    },
    title="Monthly COVID-19 deaths by age group"
)

fig_ts.update_layout(legend_title_text="Age group")
st.plotly_chart(fig_ts, use_container_width=True)

st.markdown(
    """
**Story:** This view helps your stakeholder see **when** the major waves of deaths occurred
and **which age groups** carried most of the mortality burden over time.
"""
)

# -----------------------------
# ROW 2 – SEABORN + MATPLOTLIB (HEATMAP)
# -----------------------------
st.subheader("Deaths by age group and year (Seaborn + Matplotlib)")

heat = (
    filtered
    .groupby(["Year", "Agegroup_pretty"], as_index=False)["Deaths"]
    .sum()
)

pivot = heat.pivot(index="Agegroup_pretty", columns="Year", values="Deaths")

fig_hm, ax_hm = plt.subplots(figsize=(8, 4))
sns.heatmap(
    pivot,
    annot=False,
    fmt=".0f",
    ax=ax_hm
)
ax_hm.set_title("Total deaths by age group and year")
ax_hm.set_xlabel("Year")
ax_hm.set_ylabel("Age group")

st.pyplot(fig_hm, use_container_width=True)

st.markdown(
    """
**Story:** The heatmap lets users quickly compare **which years and age groups**
were most affected, highlighting whether mortality remained concentrated in the
oldest population or spread into younger groups.
"""
)

# -----------------------------
# ROW 3 – PURE MATPLOTLIB BAR CHART
# -----------------------------
st.subheader("Total deaths by WHO region (Matplotlib)")

region_deaths = (
    filtered
    .groupby("Who_region", as_index=False)["Deaths"]
    .sum()
    .sort_values("Deaths", ascending=False)
)

fig_bar, ax_bar = plt.subplots(figsize=(8, 4))
ax_bar.bar(region_deaths["Who_region"], region_deaths["Deaths"])
ax_bar.set_title("Total COVID-19 deaths by WHO region (filtered)")
ax_bar.set_xlabel("WHO region")
ax_bar.set_ylabel("Deaths")
plt.xticks(rotation=30, ha="right")

st.pyplot(fig_bar, use_container_width=True)

st.markdown(
    """
**Story:** This bar chart compares **regions** under the current filters, helping
decision-makers prioritize where resources, vaccination campaigns, or health-system
support might be most urgently needed.
"""
)

# -----------------------------
# OPTIONAL – SHOW FILTERED TABLE
# -----------------------------
if show_raw:
    st.subheader("streamlit run covid.py Filtered data")
    st.dataframe(
        filtered[[
            "Country", "Who_region", "Wb_income",
            "Year", "Month", "Agegroup_pretty", "Deaths"
        ]].sort_values(["Year", "Month"])
    )

# -----------------------------
# FOOTER TEXT
# -----------------------------
st.markdown("---")
st.caption(
    f"""
Role selected: **{role}**.  
Use the filters on the left to explore how COVID-19 mortality by age changes
across time, countries, regions and income levels.
"""
)
