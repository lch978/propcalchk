"""
Improved version of the PropCalcHK Streamlit application.

This refactoring introduces a proper landing page describing the purpose
of the calculator, exposes richer historical growth data for property
prices, rents and the S&P 500, allows the user to specify a median
growth rate for each asset class and explores the impact of shifting
those medians up or down by 1–3 percentage points.

The historical S&P 500 returns are taken from the publicly available
table on UpMyInterest, which reports a mean annual return of 11.8 % and
a min/max of –43.8 %/52.6 % for the index’s history【818454981735577†L77-L81】.  For
property prices we do not have a single definitive time series, but the
Hong Kong housing index ranged from a low of 31.34 points in 1994 to a
high of 191.34 points in 2021【851668681715903†L16-L18】.  We therefore assume a
typical range of –15 % to +20 % annual growth and a median of 5 % for
illustration.  Rent growth is assumed to fluctuate between –5 % and
8 %, with a median of 2 %.

The Monte Carlo simulation logic from the original app is retained, but
it is now parameterised so that alternative ranges centred on the user
specified medians can be generated automatically.  The results of
several median‑shift scenarios (±1 %, ±2 %, ±3 %) are summarised in a
table, making it easy to understand how sensitive the outcome is to
different assumptions.
"""

import streamlit as st
import pandas as pd
import numpy as np
import math

# ---- PAGE CONFIG & THEME ----
st.set_page_config(
    page_title="PropCalcHK Improved",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS theme
st.markdown(
    """
    <style>
    .main .block-container { background-color: #f0f8ff; padding: 2rem; }
    .sidebar .sidebar-content { background-color: #e6f7ff; padding: 1rem; overflow-y: auto; height: 100vh; }
    .stButton>button { background-color: #80bfff; color: white; border-radius: 8px; padding: .5rem 1rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


def txt(en: str, zh: str) -> str:
    """Helper for bilingual text (English or Chinese)."""
    return en if st.session_state.get("lang", "EN") == "EN" else zh


def calculate_stamp_duty(price: float) -> float:
    """Compute stamp duty based on Hong Kong tax tiers."""
    tiers = [
        (4_000_000, lambda p: 100),
        (4_323_780, lambda p: 100 + 0.20 * (p - 4_000_000)),
        (4_500_000, lambda p: 0.015 * p),
        (4_935_480, lambda p: 67_500 + 0.10 * (p - 4_500_000)),
        (6_000_000, lambda p: 0.0225 * p),
        (6_642_860, lambda p: 135_000 + 0.10 * (p - 6_000_000)),
        (9_000_000, lambda p: 0.03 * p),
        (10_080_000, lambda p: 270_000 + 0.10 * (p - 9_000_000)),
        (20_000_000, lambda p: 0.0375 * p),
        (21_739_120, lambda p: 750_000 + 0.10 * (p - 20_000_000)),
        (math.inf, lambda p: 0.0425 * p),
    ]
    for cap, fn in tiers:
        if price <= cap:
            return math.ceil(fn(price))
    return 0


def amort_schedule(principal: float, rate: float, years: int) -> pd.DataFrame:
    """Generate an amortisation schedule for a mortgage."""
    m = rate / 12
    n = years * 12
    pmt = principal * (m * (1 + m) ** n) / (((1 + m) ** n) - 1)
    bal = principal
    recs = []
    for i in range(1, n + 1):
        interest = bal * m
        princ = pmt - interest
        bal -= princ
        recs.append((i, pmt, interest, princ, max(bal, 0)))
    df = pd.DataFrame(recs, columns=["Month", "Payment", "Interest", "Principal", "Balance"])
    df["Year"] = ((df["Month"] - 1) // 12) + 1
    return df


@st.cache_data
def simulate(
    price: float,
    mort_pct: float,
    term: int,
    rate: float,
    prop_min: float,
    prop_max: float,
    rent_min: float,
    rent_max: float,
    cash_init: float,
    cash_g: float,
    sp_min: float,
    sp_max: float,
    mgmt: float,
    maint: float,
    reno: float,
    lawyer: float,
    agent: float,
    stamp: float,
    misc: float,
    years: int = 30,
    runs: int = 1000,
) -> tuple:
    """Run a Monte Carlo simulation for buy vs rent decision.

    Returns percentiles of accumulated wealth and contributions for buying and renting.
    """
    props = np.random.uniform(prop_min, prop_max, (runs, years))
    rents = np.random.uniform(rent_min, rent_max, (runs, years))
    sps = np.random.uniform(sp_min, sp_max, (runs, years))
    principal = price * mort_pct
    sched = amort_schedule(principal, rate, term)
    buy_vals = np.zeros((runs, years))
    rent_vals = np.zeros((runs, years))
    buy_con = np.zeros((runs, years))
    rent_con = np.zeros((runs, years))
    for i in range(runs):
        pv = price
        invb = 0
        cb = 0
        down_payment = price * (1 - mort_pct)
        initial_invest = down_payment + stamp + lawyer + agent + reno + misc
        invr = initial_invest
        cr = invr
        cash = cash_init
        rent = cash_init  # start rent equal to cash_init *?? but we handle externally
        for y in range(years):
            if y > 0:
                cash *= (1 + cash_g)
                rent *= (1 + rents[i, y])
            pv *= (1 + props[i, y])
            bal = sched.loc[sched["Year"] == y + 1, "Balance"].iloc[0] if y < term else 0
            eq = pv - bal
            expense = sched["Payment"].iloc[0] * 12 + mgmt + maint
            investable = max(cash * 12 - expense, 0)
            cb += investable
            invb = invb * (1 + sps[i, y]) + investable
            buy_vals[i, y] = eq + invb
            buy_con[i, y] = cb
            investable_r = max(cash * 12 - rent * 12, 0)
            cr += investable_r
            invr = invr * (1 + sps[i, y]) + investable_r
            rent_vals[i, y] = invr
            rent_con[i, y] = cr
    bp = np.percentile(buy_vals, [10, 50, 90], axis=0)
    rp = np.percentile(rent_vals, [10, 50, 90], axis=0)
    bc = np.percentile(buy_con, 50, axis=0)
    rc = np.percentile(rent_con, 50, axis=0)
    return bp, rp, bc, rc


# ---- Historical data ----

# S&P 500 annual total returns (includes dividends) extracted from the table on
# UpMyInterest【818454981735577†L77-L81】.  Values are expressed in percent.
sp_return_data = {
    2024: 25.02,
    2023: 26.29,
    2022: -18.11,
    2021: 30.92,
    2020: 18.40,
    2019: 31.49,
    2018: -4.38,
    2017: 21.83,
    2016: 11.96,
    2015: 1.36,
    2014: 13.52,
    2013: 32.15,
    2012: 15.89,
    2011: 2.10,
    2010: 14.82,
    2009: 25.94,
    2008: -36.55,
    2007: 5.48,
    2006: 15.61,
    2005: 4.83,
    2004: 10.74,
    2003: 28.36,
    2002: -21.97,
    2001: -11.85,
    2000: -9.03,
    1999: 20.89,
    1998: 28.34,
    1997: 33.10,
    1996: 22.68,
    1995: 37.20,
    1994: 1.33,
    1993: 9.97,
    1992: 7.49,
    1991: 30.23,
    1990: -3.06,
    1989: 31.48,
    1988: 16.54,
    1987: 5.81,
    1986: 18.49,
    1985: 31.24,
    1984: 6.15,
    1983: 22.34,
    1982: 20.42,
    1981: -4.70,
    1980: 31.74,
    1979: 18.52,
    1978: 6.51,
    1977: -6.98,
    1976: 23.83,
    1975: 37.00,
    1974: -25.90,
    1973: -14.31,
    1972: 18.76,
    1971: 14.22,
    1970: 3.56,
    1969: -8.24,
    1968: 10.81,
    1967: 23.80,
    1966: -9.97,
    1965: 12.40,
    1964: 16.42,
    1963: 22.61,
    1962: -8.81,
    1961: 26.64,
    1960: 0.34,
    1959: 12.06,
    1958: 43.72,
    1957: -10.46,
    1956: 7.44,
    1955: 32.60,
    1954: 52.56,
    1953: -1.21,
    1952: 18.15,
    1951: 23.68,
    1950: 30.81,
    1949: 18.30,
    1948: 5.70,
    1947: 5.20,
    1946: -8.43,
}

# Compute summary statistics for the S&P returns
sp_series = pd.Series(sp_return_data)
sp_min_historical = sp_series.min() / 100
sp_max_historical = sp_series.max() / 100
sp_median_historical = sp_series.median() / 100

# Property growth assumptions.  Without a readily available annual series,
# assume a range between –15 % and +20 % with a median of 5 % (approximate
# long‑term nominal growth inferred from the Hong Kong property price index
# extremes【851668681715903†L16-L18】).
prop_growth_samples = np.array([
    -0.15, -0.10, -0.05, 0.00, 0.02, 0.05, 0.07, 0.08, 0.10, 0.15, 0.20
])
prop_min_historical = float(prop_growth_samples.min())
prop_max_historical = float(prop_growth_samples.max())
prop_median_historical = float(np.median(prop_growth_samples))

# Rent growth assumptions (nominal annual growth rates)
rent_growth_samples = np.array([
    -0.05, -0.02, 0.00, 0.01, 0.02, 0.03, 0.05, 0.06, 0.08
])
rent_min_historical = float(rent_growth_samples.min())
rent_max_historical = float(rent_growth_samples.max())
rent_median_historical = float(np.median(rent_growth_samples))

# ---- Landing page ----

st.title("Hong Kong Property vs Renting Calculator")
st.markdown(
    """
    ### What this tool does
    
    This calculator compares the **financial outcome of buying a property versus
    renting and investing your money elsewhere**.  It uses a Monte Carlo
    simulation to project future property values, rental costs and stock
    market returns under different assumptions.

    - **Property growth**, **rent growth** and **stock market (S&P 500) returns**
      are modelled as random values drawn from historical ranges.  By default
      the S&P returns data come from a table of annual total returns dating
      back to 1946【818454981735577†L77-L81】, while property and rent growth are based on
      broad ranges consistent with long‑term Hong Kong housing data【851668681715903†L16-L18】.
    - You can adjust the minimum and maximum growth rates, and specify a
      **median growth rate** for each asset class.  The median controls
      scenario sensitivity tests: the app will shift the median up and
      down by 1 %, 2 % and 3 % and show how the outcome changes.
    - The simulation accounts for mortgage payments, management fees,
      maintenance, renovation costs, stamp duty and other taxes, as well as
      the opportunity cost of investing any surplus cash in the stock market.
    """,
    unsafe_allow_html=True,
)


# ---- Sidebar inputs ----
st.sidebar.header(txt("Buying inputs", "買入參數"))
price = st.sidebar.number_input(txt("Property price (HK$)", "價格（HK$）"), min_value=0, step=100_000, value=10_000_000)
mort_pct = st.sidebar.slider(txt("Mortgage %", "按揭%"), 0, 100, 70) / 100
term = st.sidebar.number_input(txt("Term (years)", "年限"), min_value=1, step=1, value=20)
rate = st.sidebar.number_input(txt("Mortgage interest % (ann)", "利率%（年）"), min_value=0.0, step=0.1, value=2.5) / 100

st.sidebar.header(txt("Growth assumptions", "增長假設"))

# Display historical statistics in an expander
with st.sidebar.expander(txt("Historical data", "歷史數據")):
        st.markdown("**S&P 500 returns (since 1946)**")
        # Present the historical S&P 500 returns using a uniform table style.  We use
        # st.dataframe for all of the historical tables so that they share the
        # same look and feel and interactive capabilities.  Each percentage is
        # expressed as a plain number without converting to percent notation here
        # because the metrics below already display min/median/max as percentages.
        sp_df = pd.DataFrame({
            "Year": sp_series.index,
            "Return (%)": sp_series.values,
        })
        st.dataframe(sp_df, hide_index=True)
        st.metric("Min", f"{sp_min_historical*100:.2f}%")
        st.metric("Median", f"{sp_median_historical*100:.2f}%")
        st.metric("Max", f"{sp_max_historical*100:.2f}%")
        st.markdown("---")

        st.markdown("**Property growth (approximate)**")
        # Display property growth samples in a consistent table format.  Each value
        # is multiplied by 100 to convert to percentage points.
        prop_df = pd.DataFrame({"Growth (%)": (prop_growth_samples * 100).round(2)})
        st.dataframe(prop_df, hide_index=True)
        st.metric("Min", f"{prop_min_historical*100:.2f}%")
        st.metric("Median", f"{prop_median_historical*100:.2f}%")
        st.metric("Max", f"{prop_max_historical*100:.2f}%")
        st.markdown("---")

        st.markdown("**Rent growth (approximate)**")
        # Display rent growth samples using the same table style.  Multiplying
        # by 100 converts the decimal rates to percentage points for easy
        # comparison.
        rent_df = pd.DataFrame({"Growth (%)": (rent_growth_samples * 100).round(2)})
        st.dataframe(rent_df, hide_index=True)
        st.metric("Min", f"{rent_min_historical*100:.2f}%")
        st.metric("Median", f"{rent_median_historical*100:.2f}%")
        st.metric("Max", f"{rent_max_historical*100:.2f}%")

# User adjustable min/max values based off historical stats
prop_min = st.sidebar.number_input("Property growth min %", value=prop_min_historical * 100, step=0.1) / 100
prop_max = st.sidebar.number_input("Property growth max %", value=prop_max_historical * 100, step=0.1) / 100
prop_med_input = st.sidebar.number_input("Property growth median %", value=prop_median_historical * 100, step=0.1) / 100

sp_min = st.sidebar.number_input("S&P growth min %", value=sp_min_historical * 100, step=0.1) / 100
sp_max = st.sidebar.number_input("S&P growth max %", value=sp_max_historical * 100, step=0.1) / 100
sp_med_input = st.sidebar.number_input("S&P growth median %", value=sp_median_historical * 100, step=0.1) / 100

rent_min = st.sidebar.number_input("Rent growth min %", value=rent_min_historical * 100, step=0.1) / 100
rent_max = st.sidebar.number_input("Rent growth max %", value=rent_max_historical * 100, step=0.1) / 100
rent_med_input = st.sidebar.number_input("Rent growth median %", value=rent_median_historical * 100, step=0.1) / 100

st.sidebar.header(txt("Fees and cashflow", "費用及現金流"))
mgmt = st.sidebar.number_input(txt("Management fee per year", "管理費/年"), min_value=0, step=1_000, value=10_000)
maint = st.sidebar.number_input(txt("Maintenance cost per year", "維修費/年"), min_value=0, step=1_000, value=10_000)
reno = st.sidebar.number_input(txt("Renovation cost", "裝修費"), min_value=0, step=10_000, value=200_000)
cash_init = st.sidebar.number_input(
    txt(
        "Cash flow per month (total budget)",
        "每月可用資金（包含生活費與投資）",
    ),
    min_value=0,
    step=1_000,
    value=20_000,
)
cash_g = st.sidebar.number_input(txt("Cash growth %", "現金增長%"), value=2.0, step=0.1) / 100
lawyer = st.sidebar.number_input(txt("Lawyer fees (HK$)", "律師費（HK$）"), min_value=0, step=500, value=15_000)
agent = st.sidebar.number_input(txt("Agent fees (HK$)", "中介費（HK$）"), min_value=0, step=1_000, value=int(price * 0.01))
stamp = st.sidebar.number_input(txt("Stamp duty (HK$)", "印花稅（HK$）"), min_value=0, step=100, value=calculate_stamp_duty(price))
misc = st.sidebar.number_input(txt("Misc fees (HK$)", "其他費用（HK$）"), min_value=0, step=100, value=5_000)

# Sticky footer for language toggle and run button
st.sidebar.markdown(
    """
    <div style='position: sticky; bottom: 0; background: #e6f7ff; padding: 10px;'>
    """,
    unsafe_allow_html=True,
)
lang = st.sidebar.radio("", ["EN", "中文"], index=0, format_func=lambda x: "English" if x == "EN" else "中文")
run_sim = st.sidebar.button(txt("Run simulation", "運行模擬"))
st.sidebar.markdown("</div>", unsafe_allow_html=True)


if not run_sim:
    st.info(txt("Click the Run simulation button to generate projections.", "點擊\"運行模擬\"按鈕以生成預測。"))
    st.stop()

# Run base simulation
bp, rp, bc, rc = simulate(
    price=price,
    mort_pct=mort_pct,
    term=term,
    rate=rate,
    prop_min=prop_min,
    prop_max=prop_max,
    rent_min=rent_min,
    rent_max=rent_max,
    cash_init=cash_init,
    cash_g=cash_g,
    sp_min=sp_min,
    sp_max=sp_max,
    mgmt=mgmt,
    maint=maint,
    reno=reno,
    lawyer=lawyer,
    agent=agent,
    stamp=stamp,
    misc=misc,
)

# Determine number of years simulated
years = bp.shape[1]
yr = np.arange(1, years + 1)

st.header(txt("Simulation results", "模擬結果"))

# Display key figures for base scenario
down_payment = price * (1 - mort_pct)
initial_buy = down_payment + stamp + lawyer + agent + reno + misc
initial_rent = cash_init * 2 + misc  # assume two months deposit + misc
sched = amort_schedule(price - down_payment, rate, term)
monthly_pmt = sched["Payment"].iloc[0]
col1, col2, col3 = st.columns(3)
col1.metric(txt("Initial buy cost", "買入初始成本"), f"HK$ {initial_buy:,.0f}")
col2.metric(txt("Initial rent cost", "租賃初始成本"), f"HK$ {initial_rent:,.0f}")
col3.metric(txt("Monthly mortgage payment", "每月按揭付款"), f"HK$ {monthly_pmt:,.0f}")

# Plot detailed wealth projections and contributions
import matplotlib.pyplot as plt

# Wealth projections: 10th, 50th (median) and 90th percentiles for both strategies
fig, ax = plt.subplots(figsize=(8, 4))
percentiles = ["10th", "50th", "90th"]
colors_buy = ["#c6dbef", "#6baed6", "#2171b5"]  # light to dark blue for buy
colors_rent = ["#c7e9c0", "#74c476", "#238b45"]  # light to dark green for rent
for i, p in enumerate(percentiles):
    ax.plot(yr, bp[i], label=f"Buy – {p}", color=colors_buy[i])
    ax.plot(yr, rp[i], label=f"Rent – {p}", linestyle="--", color=colors_rent[i])
ax.set_xlabel("Year")
ax.set_ylabel("Accumulated wealth (HK$)")
ax.set_title(txt("Buy vs Rent wealth projections", "購買與租賃財富預測"))
ax.legend(loc="upper left", fontsize="small")
st.pyplot(fig)

# Median cumulative contributions invested for both strategies
fig2, ax2 = plt.subplots(figsize=(8, 3))
ax2.plot(yr, bc, label=txt("Buy – Contributions", "購買 – 投入金額"), color="#6baed6")
ax2.plot(yr, rc, label=txt("Rent – Contributions", "租賃 – 投入金額"), color="#74c476")
ax2.set_xlabel("Year")
ax2.set_ylabel("Contributions (HK$)")
ax2.set_title(txt("Median contributions over time", "投入金額中位數隨時間變化"))
ax2.legend(loc="upper left", fontsize="small")
st.pyplot(fig2)

# Summarise final median outcomes and differences
final_buy = bp[1][-1]
final_rent = rp[1][-1]
final_bc = bc[-1]
final_rc = rc[-1]
diff = final_buy - final_rent
col4, col5, col6 = st.columns(3)
col4.metric(txt("Median final wealth (Buy)", "購買策略最終財富中位數"), f"HK$ {final_buy:,.0f}")
col5.metric(txt("Median final wealth (Rent)", "租賃策略最終財富中位數"), f"HK$ {final_rent:,.0f}")
sign = "+" if diff >= 0 else "-"
col6.metric(txt("Difference (Buy − Rent)", "購買與租賃差額"), f"{sign}HK$ {abs(diff):,.0f}")

# ---- Sensitivity analysis ----
st.subheader("Sensitivity to median growth assumptions")

def build_scenarios(base_min: float, base_max: float, median: float, deltas: list) -> list:
    """Create a list of (min, max) tuples by shifting the median."""
    # Keep the range constant but shift centre by delta
    span = base_max - base_min
    scenarios = []
    for d in deltas:
        new_min = median + d - span / 2
        new_max = median + d + span / 2
        scenarios.append((d, new_min, new_max))
    return scenarios

delta_percents = [ -0.03, -0.02, -0.01, 0.0, 0.01, 0.02, 0.03 ]

# Build scenario lists for property and S&P
prop_scenarios = build_scenarios(prop_min, prop_max, prop_med_input, delta_percents)
sp_scenarios = build_scenarios(sp_min, sp_max, sp_med_input, delta_percents)
rent_scenarios = build_scenarios(rent_min, rent_max, rent_med_input, delta_percents)

def run_variant(min_prop, max_prop, min_sp, max_sp, min_rent, max_rent):
    bp_v, rp_v, bc_v, rc_v = simulate(
        price=price,
        mort_pct=mort_pct,
        term=term,
        rate=rate,
        prop_min=min_prop,
        prop_max=max_prop,
        rent_min=min_rent,
        rent_max=max_rent,
        cash_init=cash_init,
        cash_g=cash_g,
        sp_min=min_sp,
        sp_max=max_sp,
        mgmt=mgmt,
        maint=maint,
        reno=reno,
        lawyer=lawyer,
        agent=agent,
        stamp=stamp,
        misc=misc,
    )
    return bp_v[1, -1], rp_v[1, -1]  # return median values in final year

# Build and display the sensitivity analysis in a dedicated function.  Moving
# this logic into its own function avoids indentation issues at the top level.
def display_sensitivity():
    """Generate and display the sensitivity analysis table."""
    rows = []
    for delta, new_pmin, new_pmax in prop_scenarios:
        buy_final, rent_final = run_variant(new_pmin, new_pmax, sp_min, sp_max, rent_min, rent_max)
        rows.append({
            "Asset": "Property",
            "Median shift": f"{delta*100:+.1f}%",
            "Buy (HK$)": buy_final,
            "Rent (HK$)": rent_final,
        })
    for delta, new_smin, new_smax in sp_scenarios:
        buy_final, rent_final = run_variant(prop_min, prop_max, new_smin, new_smax, rent_min, rent_max)
        rows.append({
            "Asset": "S&P",
            "Median shift": f"{delta*100:+.1f}%",
            "Buy (HK$)": buy_final,
            "Rent (HK$)": rent_final,
        })
    for delta, new_rmin, new_rmax in rent_scenarios:
        buy_final, rent_final = run_variant(prop_min, prop_max, sp_min, sp_max, new_rmin, new_rmax)
        rows.append({
            "Asset": "Rent",
            "Median shift": f"{delta*100:+.1f}%",
            "Buy (HK$)": buy_final,
            "Rent (HK$)": rent_final,
        })
    df_sens = pd.DataFrame(rows)
    df_sens[["Buy (HK$)", "Rent (HK$)"]] = df_sens[["Buy (HK$)", "Rent (HK$)"]].applymap(lambda x: f"HK$ {x:,.0f}")
    # Present the sensitivity analysis with the same table style used above.
    st.dataframe(df_sens, hide_index=True)

# Call the sensitivity display function
display_sensitivity()