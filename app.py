import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------
st.set_page_config(page_title="BITCOIN PREDICTION", layout="wide")

# --------------------------------------------------
# CUSTOM UI STYLE
# --------------------------------------------------
st.markdown("""
<style>
.main-title {
    text-align:center;
    font-size:48px;
    font-weight:700;
    color:#00b4d8;
    margin-bottom:5px;
}
.sub-title {
    text-align:center;
    font-size:18px;
    color:gray;
    margin-bottom:30px;
}
.block-container {
    padding-top: 2rem;
}
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# TITLE
# --------------------------------------------------
st.markdown("<div class='main-title'>BITCOIN PREDICTION</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-title'>Machine Learning Project using Historical Bitcoin Data</div>", unsafe_allow_html=True)

# --------------------------------------------------
# DATA DOWNLOAD
# --------------------------------------------------
@st.cache_data
def load_data():
    try:
        df = yf.download("BTC-USD", period="5y", interval="1d", progress=False)
        return df
    except:
        return pd.DataFrame()

st.header("1️⃣ Data Collection")

data = load_data()

if data.empty:
    st.error("Data download failed")
    st.stop()

col1, col2 = st.columns(2)
col1.metric("Total Rows", len(data))
col2.metric("Columns", len(data.columns))

with st.expander("Show Data"):
    st.dataframe(data.tail())

# --------------------------------------------------
# PREPROCESSING
# --------------------------------------------------
st.header("2️⃣ Data Preprocessing")

data["MA7"] = data["Close"].rolling(7).mean()
data["MA30"] = data["Close"].rolling(30).mean()
data["Return"] = data["Close"].pct_change()

delta = data["Close"].diff()
gain = delta.where(delta > 0, 0).rolling(14).mean()
loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
rs = gain / loss
data["RSI"] = 100 - (100 / (1 + rs))

data = data.dropna().copy()

st.metric("Rows After Cleaning", len(data))

# --------------------------------------------------
# FEATURES
# --------------------------------------------------
features = ["Open","High","Low","Volume","MA7","MA30","Return","RSI"]

X = data[features]
y = data["Close"]

split_index = int(len(data)*0.8)

X_train = X[:split_index]
X_test = X[split_index:]
y_train = y[:split_index]
y_test = y[split_index:]

# --------------------------------------------------
# MODEL
# --------------------------------------------------
st.header("3️⃣ Model Selection")

model_choice = st.selectbox(
    "Choose Model",
    ["Linear Regression","Random Forest","SVR"]
)

if model_choice == "Linear Regression":
    model = LinearRegression()
elif model_choice == "Random Forest":
    model = RandomForestRegressor(n_estimators=200, random_state=42)
else:
    model = SVR(kernel="rbf")

model.fit(X_train, y_train)
predictions = model.predict(X_test)

# --------------------------------------------------
# EVALUATION
# --------------------------------------------------
st.header("4️⃣ Model Evaluation")

mae = mean_absolute_error(y_test, predictions)
rmse = np.sqrt(mean_squared_error(y_test, predictions))

c1, c2 = st.columns(2)
c1.metric("MAE", f"{mae:.2f}")
c2.metric("RMSE", f"{rmse:.2f}")

# --------------------------------------------------
# MODEL COMPARISON
# --------------------------------------------------
st.header("5️⃣ Model Comparison")

models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
    "SVR": SVR(kernel="rbf")
}

results = []

for name, m in models.items():
    m.fit(X_train, y_train)
    p = m.predict(X_test)
    results.append({
        "Model": name,
        "MAE": mean_absolute_error(y_test, p),
        "RMSE": np.sqrt(mean_squared_error(y_test, p))
    })

st.dataframe(pd.DataFrame(results))

# --------------------------------------------------
# GRAPH
# --------------------------------------------------
st.header("6️⃣ Actual vs Predicted Prices")

fig, ax = plt.subplots(figsize=(12,6))
ax.plot(y_test.index, y_test, label="Actual Price")
ax.plot(y_test.index, predictions, label="Predicted Price")
ax.set_title("Bitcoin Price Prediction")
ax.legend()

st.pyplot(fig)

# --------------------------------------------------
# NEXT DAY PREDICTION
# --------------------------------------------------
st.header("7️⃣ Next Day Prediction")

latest_data = X.tail(1)
next_price = float(model.predict(latest_data)[0])

st.metric("Predicted Next Closing Price ($)", f"{next_price:.2f}")

st.markdown("---")
st.write("Built using Python • Streamlit • Scikit-learn • Yahoo Finance")