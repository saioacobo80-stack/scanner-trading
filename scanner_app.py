import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go


# =====================================
# CONFIG
# =====================================

st.set_page_config(
    page_title="Scanner Trading",
    layout="wide"
)

st.title("Scanner Graphseo + Weinstein")


# =====================================
# GROSSE LISTE D'ACTIONS
# =====================================

@st.cache_data
def load_big_ticker_list():

    return sorted(list(set([

        # =====================================
        # BIG TECH / AI
        # =====================================

        "AAPL","MSFT","NVDA","AMD","AVGO","MU","QCOM","ARM",
        "GOOGL","META","AMZN","NFLX","TSLA","PLTR",
        "SMCI","AI","SOUN","BBAI","APLD",

        # =====================================
        # QUANTUM
        # =====================================

        "IONQ","RGTI","QBTS","QUBT",

        # =====================================
        # BITCOIN / MINERS
        # =====================================

        "WULF","MARA","RIOT","IREN","CIFR",
        "CLSK","HUT","BTDR","HIVE","BITF",
        "CORZ","CAN","ARBK",

        # =====================================
        # GOLD MINERS
        # =====================================

        "NEM","GOLD","AEM","AU","KGC",
        "AGI","BTG","EGO","IAG","OR",
        "WPM","FNV","RGLD",

        # =====================================
        # SILVER MINERS
        # =====================================

        "HL","CDE","PAAS","AG","EXK",
        "SVM","MAG","SILV",

        # =====================================
        # URANIUM
        # =====================================

        "CCJ","UEC","UUUU","DNN","NXE",
        "LEU","URNJ","URA",

        # =====================================
        # OIL / PETROLE
        # =====================================

        "XOM","CVX","COP","OXY","FANG",
        "EOG","SLB","HAL","BKR","DVN",
        "MRO","APA","SM","CRK","CTRA",
        "VLO","MPC",

        # =====================================
        # NATURAL GAS / LNG
        # =====================================

        "LNG","TELL","NEXT","CHK",

        # =====================================
        # COPPER / METALS
        # =====================================

        "FCX","SCCO","TECK","RIO","BHP",
        "VALE","AA","CENX",

        # =====================================
        # LITHIUM
        # =====================================

        "ALB","SQM","LAC","PLL","SGML",

        # =====================================
        # STEEL
        # =====================================

        "CLF","X","NUE","STLD","MT",

        # =====================================
        # ENERGY ETF
        # =====================================

        "XLE","OIH","URA","SIL","GDX","GDXJ",

        # =====================================
        # GOLD / SILVER ETF
        # =====================================

        "GLD","SLV","IAU","SIVR",

        # =====================================
        # CRYPTO ETF
        # =====================================

        "IBIT","FBTC","BITB","ARKB",

        # =====================================
        # FINTECH
        # =====================================

        "SOFI","HOOD","COIN","UPST","AFRM",

        # =====================================
        # CYBERSECURITY
        # =====================================

        "CRWD","PANW","ZS","NET","FTNT",

        # =====================================
        # CLOUD
        # =====================================

        "SNOW","DDOG","MDB","ESTC","DOCN",

        # =====================================
        # BIOTECH
        # =====================================

        "MRNA","BNTX","VRTX","REGN","GILD",

        # =====================================
        # SPACE
        # =====================================

        "RKLB","ASTS","LUNR","RDW","SPCE",

        # =====================================
        # MEME / HIGH BETA
        # =====================================

        "GME","AMC","DJT",

        # =====================================
        # ETF
        # =====================================

        "SPY","QQQ","IWM","SMH","ARKK"

    ])))


# =====================================
# CHOIX ACTION
# =====================================

actions = load_big_ticker_list()

mode = st.radio(
    "Mode de sélection",
    ["Liste", "Ticker manuel"],
    horizontal=True
)

if mode == "Liste":

    ticker = st.selectbox(
        "Choisir une action",
        actions,
        index=actions.index("WULF")
    )

else:

    ticker = st.text_input(
        "Entrer un ticker",
        "WULF"
    ).upper()


# =====================================
# PERIODE
# =====================================

period = st.selectbox(
    "Période",
    ["6mo", "1y", "2y", "3y", "5y"],
    index=2
)


# =====================================
# INDICATEURS
# =====================================

def add_indicators(df):

    df["EMA20"] = df["Close"].ewm(span=20).mean()

    df["EMA50"] = df["Close"].ewm(span=50).mean()

    df["EMA200"] = df["Close"].ewm(span=200).mean()

    df["Volume_MA20"] = df["Volume"].rolling(20).mean()

    df["High_20"] = df["High"].rolling(20).max()

    # RSI

    delta = df["Close"].diff()

    gain = delta.clip(lower=0)

    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(14).mean()

    avg_loss = loss.rolling(14).mean()

    rs = avg_gain / avg_loss

    df["RSI"] = 100 - (100 / (1 + rs))

    # Weinstein

    df["MM30_Weinstein"] = df["Close"].rolling(150).mean()

    df["MM30_Slope"] = df["MM30_Weinstein"].diff(5)

    return df


# =====================================
# SIGNAUX
# =====================================

def detect_signals(df):

    df["Score"] = 0

    # Breakout

    df.loc[
        df["Close"] > df["High_20"].shift(1),
        "Score"
    ] += 25

    # Volume

    df.loc[
        df["Volume"] > df["Volume_MA20"] * 1.8,
        "Score"
    ] += 25

    # Tendance

    df.loc[
        df["EMA20"] > df["EMA50"],
        "Score"
    ] += 20

    df.loc[
        df["EMA50"] > df["EMA200"],
        "Score"
    ] += 20

    # RSI

    df.loc[
        df["RSI"] > 60,
        "Score"
    ] += 10

    # Weinstein

    df["Weinstein_Stage2"] = (
        (df["Close"] > df["MM30_Weinstein"]) &
        (df["MM30_Slope"] > 0)
    )

    df.loc[
        df["Weinstein_Stage2"],
        "Score"
    ] += 20

    # Achat

    df["Buy_Signal"] = df["Score"] >= 80

    # Vente

    df["Sell_Signal"] = (
        (df["Close"] < df["EMA20"]) &
        (df["RSI"] < 45)
    )

    return df


# =====================================
# DOWNLOAD
# =====================================

df = yf.download(
    ticker,
    period=period,
    interval="1d",
    auto_adjust=True,
    progress=False,
    threads=False
)

if isinstance(df.columns, pd.MultiIndex):

    df.columns = df.columns.get_level_values(0)

if df.empty:

    st.error("Aucune donnée trouvée")

    st.stop()


# =====================================
# CALCULS
# =====================================

df = add_indicators(df)

df = detect_signals(df)

latest = df.iloc[-1]


# =====================================
# METRICS
# =====================================

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Prix actuel",
    f"{latest['Close']:.2f}"
)

col2.metric(
    "RSI",
    f"{latest['RSI']:.2f}"
)

col3.metric(
    "Score",
    f"{latest['Score']}/120"
)

col4.metric(
    "Weinstein Stage 2",
    str(latest["Weinstein_Stage2"])
)


# =====================================
# MESSAGE
# =====================================

if latest["Buy_Signal"]:

    st.success("Signal ACHAT fort")

elif latest["Sell_Signal"]:

    st.warning("Signal VENTE / prudence")

else:

    st.info("Pas de signal fort")


# =====================================
# GRAPHIQUE PRIX
# =====================================

fig = go.Figure()

fig.add_trace(
    go.Scatter(
        x=df.index,
        y=df["Close"],
        name="Prix"
    )
)

fig.add_trace(
    go.Scatter(
        x=df.index,
        y=df["EMA20"],
        name="EMA20"
    )
)

fig.add_trace(
    go.Scatter(
        x=df.index,
        y=df["EMA50"],
        name="EMA50"
    )
)

fig.add_trace(
    go.Scatter(
        x=df.index,
        y=df["EMA200"],
        name="EMA200"
    )
)

fig.add_trace(
    go.Scatter(
        x=df.index,
        y=df["MM30_Weinstein"],
        name="MM30 Weinstein"
    )
)

# Signaux Achat / Vente

buy_points = df[df["Buy_Signal"]]

sell_points = df[df["Sell_Signal"]]

fig.add_trace(
    go.Scatter(
        x=buy_points.index,
        y=buy_points["Close"],
        mode="markers",
        name="Achat",
        marker=dict(
            symbol="triangle-up",
            size=12,
            color="green"
        )
    )
)

fig.add_trace(
    go.Scatter(
        x=sell_points.index,
        y=sell_points["Close"],
        mode="markers",
        name="Vente",
        marker=dict(
            symbol="triangle-down",
            size=10,
            color="red"
        )
    )
)

fig.update_layout(
    title=f"Graphique {ticker}",
    height=650,
    xaxis_title="Date",
    yaxis_title="Prix"
)

st.plotly_chart(
    fig,
    use_container_width=True
)


# =====================================
# VOLUMES
# =====================================

st.subheader("Volumes")

volume_colors = np.where(
    df["Close"] >= df["Open"],
    "green",
    "red"
)

vol_fig = go.Figure()

vol_fig.add_trace(
    go.Bar(
        x=df.index,
        y=df["Volume"],
        name="Volume",
        marker_color=volume_colors
    )
)

vol_fig.add_trace(
    go.Scatter(
        x=df.index,
        y=df["Volume_MA20"],
        name="Volume MA20"
    )
)

vol_fig.update_layout(
    height=300
)

st.plotly_chart(
    vol_fig,
    use_container_width=True
)


# =====================================
# TABLEAU
# =====================================

st.subheader("Dernières données")

st.dataframe(
    df.tail(20),
    use_container_width=True
)