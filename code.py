import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go 

#beginning page
st.set_page_config(
    page_title="COVID-19 Global Mortality by Age )",
    layout="wide"
)

st.title("COVID-19 Global Mortality by Age")

st.markdown(
    """
This dashboard explores **monthly COVID-19 deaths by age group, country and region**,based on WHO data. It is designed for a **public-health analyst working for WHO** who needs to
quickly understand **where and for whom** the burden of mortality is highest.
"""
)


#loading the data 
@st.cache_data
def load_data():
    data_path = "WHO-COVID-19-global-monthly-death-by-age-data.csv"
    df = pd.read_csv(data_path)

    df["Deaths"] = df["Deaths"].fillna(0)
    df["Date"] = pd.to_datetime(
        df["Year"].astype(str) + "-" + df["Month"].astype(str) + "-01",
        errors="coerce"
    )

    age_map = {
        "0_4": "0–4",
        "5_14": "5–14",
        "15_64": "15–64",
        "65+": "65+"
    }
    df["Agegroup_pretty"] = df["Agegroup"].map(age_map).fillna(df["Agegroup"])

    region_map = {
        "AFR": "African Region",
        "AMR": "Region of the Americas",
        "EMR": "Eastern Mediterranean Region",
        "EUR": "European Region",
        "SEAR": "South-East Asia Region",
        "WPR": "Western Pacific Region"
    }
    df["Who_region_pretty"] = df["Who_region"].map(region_map).fillna(df["Who_region"])

    return df


df = load_data()


#sidebar and filters for dashboard
st.sidebar.title("Controls")

role = "WHO regional analyst"
st.sidebar.markdown(f"**Role:** {role}")

st.sidebar.markdown("---")

regions = ["All regions"] + sorted(df["Who_region_pretty"].dropna().unique().tolist())
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

filtered = df.copy()

if region_selected != "All regions":
    filtered = filtered[filtered["Who_region_pretty"] == region_selected]

if income_selected != "All income levels":
    filtered = filtered[filtered["Wb_income"] == income_selected]

if countries_selected:
    filtered = filtered[filtered["Country"].isin(countries_selected)]

if agegroups_selected:
    filtered = filtered[filtered["Agegroup_pretty"].isin(agegroups_selected)]

filtered = filtered[
    (filtered["Year"] >= year_range[0]) &
    (filtered["Year"] <= year_range[1])
]

if filtered.empty:
    st.error("No data for the current filter selection. Try widening your filters.")
    st.stop()

#kpi metrics
total_deaths = filtered["Deaths"].sum()

max_date = filtered["Date"].max()
last_12_months_start = max_date - pd.DateOffset(months=12)
recent = filtered[filtered["Date"] >= last_12_months_start]
recent_deaths = recent["Deaths"].sum()

age_summary = (
    filtered
    .groupby("Agegroup_pretty")["Deaths"]
    .sum()
    .sort_values(ascending=False)
)

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

#first graph
st.subheader("Monthly deaths over time by age group (with median & mode)")

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

median_deaths = ts["Deaths"].median()
mode_values = ts["Deaths"].mode()
mode_deaths = mode_values.iloc[0] if not mode_values.empty else None

x_min = ts["Date"].min()
x_max = ts["Date"].max()

fig_ts.add_trace(
    go.Scatter(
        x=[x_min, x_max],
        y=[median_deaths, median_deaths],
        mode="lines",
        name=f"Median deaths ({median_deaths:,.0f})",
        line=dict(dash="dash")
    )
)

if mode_deaths is not None:
    fig_ts.add_trace(
        go.Scatter(
            x=[x_min, x_max],
            y=[mode_deaths, mode_deaths],
            mode="lines",
            name=f"Mode deaths ({mode_deaths:,.0f})",
            line=dict(dash="dot")
        )
    )

fig_ts.update_layout(legend_title_text="Age group / Statistics")
st.plotly_chart(fig_ts, use_container_width=True)

st.markdown(
    f"""
###Monthly deaths statistics (current filters)
- **Median monthly deaths:** {median_deaths:,.0f}  
- **Mode monthly deaths:** {mode_deaths:,.0f}
"""
)



#second graph
st.subheader("Deaths by age group and year (Heatmap)")

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
**Story:** The heatmap lets us quickly compare **which years and age groups**
were most affected, highlighting whether mortality remained concentrated in the
oldest population or spread into younger groups.
"""
)

#third graph
st.subheader("Global Map with COVID-19 deaths by country")
st.write("Filtered rows:", len(filtered))
st.write("Unique countries in filtered data:", filtered["Country"].nunique())

map_data = (
    filtered
    .groupby("Country", as_index=False)["Deaths"]
    .sum()
)

map_data = map_data.dropna(subset=["Country"])
map_data = map_data[map_data["Country"] != "Unknown"]

if map_data.empty:
    st.warning("No country-level data available for the selected filters.")
else:
    try:
        fig_map = px.choropleth(
            map_data,
            locations="Country",
            locationmode="country names",
            color="Deaths",
            color_continuous_scale="Reds",
            title="Total COVID-19 deaths by country (filtered)",
        )

        fig_map.update_geos(
            projection_type="natural earth",
            showcountries=True,
            showcoastlines=True,
            coastlinecolor="gray",
            showland=True,
            landcolor="white",
            showocean=True,
            oceancolor="#d0e7f7"
        )

        st.plotly_chart(fig_map, use_container_width=True)

    except Exception as e:
        st.error("❌ Error while building the choropleth map.")
        st.exception(e)

#fourth graph
st.subheader("Total deaths by WHO region")

region_deaths = (
    filtered
    .groupby("Who_region_pretty", as_index=False)["Deaths"]
    .sum()
    .sort_values("Deaths", ascending=False)
)

fig_bar, ax_bar = plt.subplots(figsize=(8, 4))
ax_bar.bar(region_deaths["Who_region_pretty"], region_deaths["Deaths"])
ax_bar.set_title("Total COVID-19 deaths by WHO region (filtered)")
ax_bar.set_xlabel("WHO region")
ax_bar.set_ylabel("Deaths")
plt.xticks(rotation=30, ha="right")

st.pyplot(fig_bar, use_container_width=True)

st.markdown(
    """
**Story:** This bar chart compares **regions** under the current filters, helping
us prioritize where resources, vaccination campaigns, or health-system
support might be most urgently needed.
"""
)

if show_raw:
    st.subheader("Filtered data")
    st.dataframe(
        filtered[[
            "Country", "Who_region_pretty", "Wb_income",
            "Year", "Month", "Agegroup_pretty", "Deaths"
        ]].sort_values(["Year", "Month"])
    )
    
#final text
st.markdown("---")
st.caption(
    f"""
Role selected: **{role}**.  
The filters help us explore how COVID-19 mortality by age changes
across time, countries, regions and income levels.
"""
)

