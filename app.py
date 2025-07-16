# app.py

import streamlit as st
import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
import re
import math
import matplotlib.pyplot as plt

# ---- PAGE CONFIG & THEME ----
st.set_page_config(
    page_title="PropCalcHK",
    layout="wide",
    initial_sidebar_state="expanded"
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
    unsafe_allow_html=True
)

# ---- HELPERS ----
def txt(en, zh):
    return en if st.session_state.get("lang", "EN") == "EN" else zh

# Stamp duty calculator
def calculate_stamp_duty(price: float) -> float:
    tiers = [
        (4_000_000, lambda p: 100),
        (4_323_780, lambda p: 100 + 0.20*(p-4_000_000)),
        (4_500_000, lambda p: 0.015*p),
        (4_935_480, lambda p: 67_500 + 0.10*(p-4_500_000)),
        (6_000_000, lambda p: 0.0225*p),
        (6_642_860, lambda p: 135_000 + 0.10*(p-6_000_000)),
        (9_000_000, lambda p: 0.03*p),
        (10_080_000, lambda p: 270_000 + 0.10*(p-9_000_000)),
        (20_000_000, lambda p: 0.0375*p),
        (21_739_120, lambda p: 750_000 + 0.10*(p-20_000_000)),
        (math.inf,     lambda p: 0.0425*p)
    ]
    for cap, fn in tiers:
        if price <= cap:
            return math.ceil(fn(price))
    return 0

@st.cache_data
def get_listing_data(url: str) -> dict:
    if not url:
        return {}
    try:
        resp = requests.get(url, headers={'User-Agent':'Mozilla/5.0'}, timeout=5)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, 'html.parser')
        el = soup.select_one('.info-price .value')
        if el:
            return {'price': int(re.sub(r'[^\d]', '', el.text))}
    except:
        st.warning(txt("Auto-fetch failed; fill manually.", "自動獲取失敗，請手動輸入。"))
    return {}

# Amortization schedule
def amort_schedule(principal: float, rate: float, years: int) -> pd.DataFrame:
    m = rate/12
    n = years*12
    pmt = principal*(m*(1+m)**n)/((1+m)**n - 1)
    bal = principal
    recs = []
    for i in range(1, n+1):
        interest = bal*m
        princ = pmt - interest
        bal -= princ
        recs.append((i, pmt, interest, princ, max(bal,0)))
    df = pd.DataFrame(recs, columns=['Month','Payment','Interest','Principal','Balance'])
    df['Year'] = ((df['Month']-1)//12) + 1
    return df

# Monte Carlo simulation
@st.cache_data
def simulate(
    price, mort_pct, term, rate,
    prop_min, prop_max,
    avg_rent, rent_min, rent_max,
    cash_init, cash_g,
    sp_min, sp_max,
    mgmt, maint, reno, lawyer, agent, stamp, misc,
    years=30, runs=1000
):
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
        pv = price; invb = 0; cb = 0
        # Calculate initial buying cost (down payment + fees)
        down_payment = price * (1 - mort_pct)
        initial_invest = down_payment + stamp + lawyer + agent + reno + misc
        # Use initial buying cost as the initial deposit for renting
        invr = initial_invest; cr = invr
        cash = cash_init; rent = avg_rent
        for y in range(years):
            if y > 0:
                cash *= (1 + cash_g)
                rent *= (1 + rents[i, y])
            pv *= (1 + props[i, y])
            bal = sched.loc[sched['Year'] == y + 1, 'Balance'].iloc[0] if y < term else 0
            eq = pv - bal
            expense = sched['Payment'].iloc[0]*12 + mgmt + maint
            investable = max(cash*12 - expense, 0)
            cb += investable
            invb = invb*(1+sps[i, y]) + investable
            buy_vals[i, y] = eq + invb
            buy_con[i, y] = cb
            investable_r = max(cash*12 - rent*12, 0)
            cr += investable_r
            invr = invr*(1+sps[i, y]) + investable_r
            rent_vals[i, y] = invr
            rent_con[i, y] = cr
    bp = np.percentile(buy_vals, [10,50,90], axis=0)
    rp = np.percentile(rent_vals, [10,50,90], axis=0)
    bc = np.percentile(buy_con, 50, axis=0)
    rc = np.percentile(rent_con, 50, axis=0)
    return bp, rp, bc, rc

# ---- SIDEBAR INPUTS ----
st.sidebar.header(txt("Buying Inputs","買入參數"))
price = st.sidebar.number_input(txt("Price (HK$)","價格（HK$）"), 0, step=100000, value=10000000, key='price')
mort_pct = st.sidebar.slider(txt("Mortgage %","按揭%"), 0,100,70, key='mort_pct')/100
term = st.sidebar.number_input(txt("Term (yrs)","年限"), 1, step=1, value=20, key='term')
rate = st.sidebar.number_input(txt("Interest % (ann)","利率%（年）"), 0.0, step=0.1, value=2.5, key='rate')/100
prop_min = st.sidebar.number_input(txt("Prop growth min %","物業增長下限%"), -100.0, step=0.1, value=2.0, key='prop_min')/100
prop_max = st.sidebar.number_input(txt("Prop growth max %","物業增長上限%"), -100.0, step=0.1, value=5.0, key='prop_max')/100
mgmt = st.sidebar.number_input(txt("Mgmt fee/yr","管理費/年"), 0, step=1000, value=10000, key='mgmt')
maint = st.sidebar.number_input(txt("Maint cost/yr","維修費/年"), 0, step=1000, value=10000, key='maint')
reno = st.sidebar.number_input(txt("Reno cost","裝修費"), 0, step=10000, value=200000, key='reno')
cash_init = st.sidebar.number_input(txt("Cash flow/mo","每月現金流"), 0, step=1000, value=20000, key='cash_init')
cash_g = st.sidebar.number_input(txt("Cash inc %","現金增長%"), -100.0, step=0.1, value=2.0, key='cash_g')/100

st.sidebar.header(txt("Renting Inputs","租賃參數"))
listing_url = st.sidebar.text_input(txt("Listing URL (28Hse)","物業連結(28Hse)"), key='listing_url')
listing_data = get_listing_data(listing_url)
avg_rent = st.sidebar.number_input(txt("Avg monthly rent (HK$)","平均月租（HK$）"), 0.0, step=500.0, value=float(listing_data.get('price',30000))*0.003, key='avg_rent')
rent_min = st.sidebar.number_input(txt("Rent growth min %","租金增長下限%"), -100.0, step=0.1, value=1.0, key='rent_min')/100
rent_max = st.sidebar.number_input(txt("Rent growth max %","租金增長上限%"), -100.0, step=0.1, value=3.0, key='rent_max')/100

st.sidebar.header(txt("Investment & Fees","投資及費用"))
sp_min = st.sidebar.number_input(txt("S&P min %","標普下限%"), -100.0, step=0.1, value=5.0, key='sp_min')/100
sp_max = st.sidebar.number_input(txt("S&P max %","標普上限%"), -100.0, step=0.1, value=9.0, key='sp_max')/100
lawyer = st.sidebar.number_input(txt("Lawyer fees (HK$)","律師費（HK$）"), 0, step=500, value=15000, key='lawyer')
agent = st.sidebar.number_input(txt("Agent fees (HK$)","中介費（HK$）"), 0, step=1000, value=int(price*0.01), key='agent')
stamp = st.sidebar.number_input(txt("Stamp duty (HK$)","印花稅（HK$）"), 0, step=100, value=calculate_stamp_duty(price), key='stamp')
tax = st.sidebar.number_input(txt("Other taxes (HK$)","其他稅費（HK$）"), 0, step=100, value=int(price*0.002), key='tax')
misc = st.sidebar.number_input(txt("Misc fees (HK$)","其他費用（HK$）"), 0, step=100, value=5000, key='misc')

# ---- SIDEBAR STICKY CONTROLS ----
st.sidebar.markdown("""
<div style='position: sticky; bottom: 0; background: #e6f7ff; padding: 10px;'>
""", unsafe_allow_html=True)
# Language toggle
lang = st.sidebar.radio("", ["EN", "中文"], index=0, format_func=lambda x: "English" if x=="EN" else "中文", key='lang')
# Run Simulation
if st.sidebar.button(txt("Run Simulation","運行模擬"), key='run'):
    buy_pct, rent_pct, buy_con, rent_con = simulate(
        price, mort_pct, term, rate,
        prop_min, prop_max,
        avg_rent, rent_min, rent_max,
        cash_init, cash_g,
        sp_min, sp_max,
        mgmt, maint, reno, lawyer, agent, stamp, misc
    )
else:
    st.info(txt("Click 'Run Simulation' to see projections.","點擊「運行模擬」以查看預測。"))
    st.stop()
st.sidebar.markdown("""
</div>
""", unsafe_allow_html=True)

# ---- DISPLAY ----
# Define years and x-axis
years = buy_pct.shape[1]
yr = np.arange(1, years + 1)

# Key Figures
# Key Figures
st.header(txt("Key Figures","關鍵數據"))
col1, col2, col3 = st.columns(3)
# initial_buy and initial_rent calculations
down_payment = price * (1 - mort_pct)
loan_amount = price - down_payment
initial_buy = down_payment + stamp + lawyer + agent + reno + misc
initial_rent = avg_rent * 2 + misc
sched = amort_schedule(loan_amount, rate, term)
sched['Year'] = ((sched['Month'] - 1) // 12) + 1
monthly_pmt = sched['Payment'].iloc[0]
col1.metric(txt("Initial Buy Cost","買入初始成本"), f"HK$ {initial_buy:,.0f}")
col2.metric(txt("Initial Rent Cost","租賃初始成本"), f"HK$ {initial_rent:,.0f}")
col3.metric(txt("Monthly Mortgage Pmt","每月按揭付款"), f"HK$ {monthly_pmt:,.0f}")

st.subheader(txt("Initial Cost Breakdown","初始成本明細"))
# markdown both EN/ZH
st.markdown(
    txt(
        f"""
- Down payment: HK$ {down_payment:,.0f}
- Stamp duty: HK$ {stamp:,.0f}
- Lawyer fees: HK$ {lawyer:,.0f}
- Agent fees: HK$ {agent:,.0f}
- Renovation cost: HK$ {reno:,.0f}
- Misc fees: HK$ {misc:,.0f}
**Total Buy Cost**: HK$ {initial_buy:,.0f}
""",
        f"""
- 首期: HK$ {down_payment:,.0f}
- 印花稅: HK$ {stamp:,.0f}
- 律師費: HK$ {lawyer:,.0f}
- 中介費: HK$ {agent:,.0f}
- 裝修費: HK$ {reno:,.0f}
- 其他費用: HK$ {misc:,.0f}
**總買入成本**: HK$ {initial_buy:,.0f}
"""
    ),
    unsafe_allow_html=True
)

st.subheader(txt("Initial Rent Cost Breakdown","租賃初始成本明細"))
# markdown both EN/ZH
st.markdown(
    txt(
        f"""
- Deposit (2 months): HK$ {initial_rent - misc:,.0f}
- Misc fees: HK$ {misc:,.0f}
**Total Rent Cost**: HK$ {initial_rent:,.0f}
""",
        f"""
- 押金(2個月): HK$ {initial_rent - misc:,.0f}
- 其他費用: HK$ {misc:,.0f}
**總租賃成本**: HK$ {initial_rent:,.0f}
"""
    ),
    unsafe_allow_html=True
)

# Buying Portfolio Breakdown
st.subheader(txt("Buying Portfolio Breakdown","買入組合明細"))
mean_pg = np.mean([prop_min, prop_max])
buy_equity = price * (1 + mean_pg) ** yr - [sched.loc[sched['Year'] == y, 'Balance'].iloc[0] if y <= term else 0 for y in yr]
# Stock value should start at 0 and grow with contributions
buy_stock = buy_con
buy_total = buy_equity + buy_stock
buy_df = pd.DataFrame({
    txt("Year","年"): yr,
    txt("Property Price","物業價格"): price * (1 + mean_pg) ** yr,
    txt("Remaining Loan","未償還貸款"): [sched.loc[sched['Year'] == y, 'Balance'].iloc[0] if y <= term else 0 for y in yr],
    txt("Equity Owned","持有權益"): buy_equity,
    txt("Stock Value","股票價值"): buy_stock,
    txt("Total Portfolio Value","總組合價值"): buy_total
})
st.dataframe(buy_df.style.format({col: "{:,.0f}" for col in buy_df.columns}), use_container_width=True)

# Rent + Invest Breakdown
st.subheader(txt("Rent + Invest Breakdown","租賃加投資明細"))
mean_rg = np.mean([rent_min, rent_max])
rent_df = pd.DataFrame({
    txt("Year","年"): yr,
    txt("Rent Price","租金"): [avg_rent * (1 + mean_rg) ** y for y in yr],
    txt("Cumulative Contribution","累計投資"): rent_con,
    txt("Total Portfolio Value","總組合價值"): rent_pct[1]
})
st.dataframe(rent_df.style.format({col: "{:,.0f}" for col in rent_df.columns}), use_container_width=True)

# Monte Carlo Explanation
st.subheader(txt("Monte Carlo Simulation Explanation","蒙地卡羅模擬說明"))
st.markdown(txt(
    "Monitors random annual growth to produce percentile bands.",
    "通過隨機年度增長生成百分位區間。"
), unsafe_allow_html=True)

# Projected Portfolio Over Time with deterministic bounds
st.subheader(txt("Projected Portfolio Over Time","預測投資組合價值"))
fig, ax = plt.subplots()
ax.fill_between(yr, buy_pct[0], buy_pct[2], alpha=0.2, label=txt("MC Buy 10–90%","蒙買10–90%"))
ax.plot(yr, buy_pct[1], color='blue', label=txt("MC Buy Median","蒙買中位"))
ax.fill_between(yr, rent_pct[0], rent_pct[2], alpha=0.2, label=txt("MC Rent 10–90%","蒙租10–90%"))
ax.plot(yr, rent_pct[1], color='green', label=txt("MC Rent Median","蒙租中位"))
# deterministic
pv_min=price; pv_max=price; ib_min=0; ib_max=0; ir_min=initial_buy; ir_max=initial_buy; cs=cash_init; rt=avg_rent
det_bmin=[]; det_bmax=[]; det_rmin=[]; det_rmax=[]
for y in range(years):
    pv_min*=(1+prop_min)
    ib_min=ib_min*(1+sp_min)+max(cs*12 - (monthly_pmt*12+mgmt+maint),0)
    det_bmin.append(pv_min - (sched.loc[sched['Year']==y+1,'Balance'].iloc[0] if y+1<=term else 0) + ib_min)
    pv_max*=(1+prop_max)
    ib_max=ib_max*(1+sp_max)+max(cs*12 - (monthly_pmt*12+mgmt+maint),0)
    det_bmax.append(pv_max - (sched.loc[sched['Year']==y+1,'Balance'].iloc[0] if y+1<=term else 0) + ib_max)
    det_rmin.append(ir_min*(1+sp_min) + max(cs*12 - rt*12,0)); det_rmax.append(ir_max*(1+sp_max) + max(cs*12 - rt*12,0))
    cs*=(1+cash_g); rt*=(1+rent_min)
ax.plot(yr, det_bmin, linestyle='--', color='navy', label=txt("Det Buy Min","確定買入下限"))
ax.plot(yr, det_bmax, linestyle=':', color='navy', label=txt("Det Buy Max","確定買入上限"))
ax.plot(yr, det_rmin, linestyle='--', color='darkgreen', label=txt("Det Rent Min","確定租用下限"))
ax.plot(yr, det_rmax, linestyle=':', color='darkgreen', label=txt("Det Rent Max","確定租用上限"))
ax.set_xlabel(txt("Year","年")); ax.set_ylabel(txt("Portfolio Value (HK$)","投資組合價值（HK$）")); ax.legend(); st.pyplot(fig)

# Yearly Comparison Table
st.subheader(txt("Yearly Comparison Table","年度比較表"))
# Compute monthly contribution available each year
monthly_buy = np.empty(years)
monthly_rent = np.empty(years)
monthly_buy[0] = buy_con[0] / 12
monthly_rent[0] = rent_con[0] / 12
monthly_buy[1:] = np.diff(buy_con) / 12
monthly_rent[1:] = np.diff(rent_con) / 12
# Build comparison table with MC medians and monthly contributions
comp = pd.DataFrame({
    txt("Year","年"): yr,
    txt("Buy Median","買入中位"): buy_pct[1],
    txt("Rent Median","租用中位"): rent_pct[1],
    txt("Buy Monthly Contribution","買入每月貢獻"): monthly_buy,
    txt("Rent Monthly Contribution","租賃每月貢獻"): monthly_rent,
    txt("Δ","差額"): buy_pct[1] - rent_pct[1]
})
st.dataframe(comp.style.format({col: "{:,.0f}" for col in comp.columns}), use_container_width=True)

# Run instructions
st.sidebar.markdown(
    """
    ```bash
    python -m streamlit run app.py
    ```
    """,
    unsafe_allow_html=True
)

# Sticky sidebar controls
st.markdown(
    """
    <style>
    /* Make sidebar content layout vertical */
    .sidebar .sidebar-content {
        display: flex;
        flex-direction: column;
    }
    /* Push last two elements (language toggle & run button) to bottom */
    .sidebar .sidebar-content > div:last-child,
    .sidebar .sidebar-content > div:nth-last-child(2) {
        margin-top: auto;
    }
    </style>
    """,
    unsafe_allow_html=True
)
# ---- STICKY SIDEBAR BUTTONS ----
lang_secondary = st.sidebar.radio(
    "",
    ["EN", "中文"],
    index=0,
    format_func=lambda x: "English" if x == "EN" else "中文",
    key="lang_secondary"
)
if st.sidebar.button(txt("Run Simulation", "運行模擬"), key='run_secondary'):
    buy_pct, rent_pct, buy_con, rent_con = simulate(
        price, mort_pct, term, rate,
        prop_min, prop_max,
        avg_rent, rent_min, rent_max,
        cash_init, cash_g,
        sp_min, sp_max,
        mgmt, maint, reno, lawyer, agent, stamp, misc
    )
else:
    st.info(txt("Click 'Run Simulation' to see projections.", "點擊「運行模擬」以查看預測。"))
    st.stop()
