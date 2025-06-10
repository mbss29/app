# --- Imports ---
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error

st.set_page_config(page_title="Car Purchase Predictor", layout="centered")
st.title("🚗 Car Purchase Amount Predictor with Recommendations")

# --- Load datasets ---
@st.cache_data
def load_data():
    try:
        df_main = pd.read_csv("car_purchasing.csv", encoding='ISO-8859-1')
        df_cars = pd.read_csv("Sport car price.csv")
        return df_main, df_cars
    except Exception as e:
        st.error(f"Data loading error: {e}")
        return None, None

df, cars_df = load_data()
if df is None or cars_df is None:
    st.stop()

# --- Prepare model dataset ---
df_model = df.drop(columns=["customer name", "customer e-mail", "country"])
X = df_model.drop(columns=["car purchase amount"])
y = df_model["car purchase amount"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model = RandomForestRegressor(random_state=42)
model.fit(X_scaled, y)

y_pred = model.predict(X_scaled)
r2 = r2_score(y, y_pred)
mse = mean_squared_error(y, y_pred)

# --- Prepare visualizations ---
fig1, ax1 = plt.subplots()
ax1.scatter(y, y_pred, edgecolors="black")
ax1.plot([y.min(), y.max()], [y.min(), y.max()], "r--", lw=2)
ax1.set_xlabel("Actual Purchase Amount")
ax1.set_ylabel("Predicted Purchase Amount")
ax1.set_title("Actual vs Predicted")

# Feature Importance
importances = model.feature_importances_
features = X.columns
sorted_idx = np.argsort(importances)

fig2, ax2 = plt.subplots()
ax2.barh(range(len(importances)), importances[sorted_idx], align="center")
ax2.set_yticks(range(len(importances)))
ax2.set_yticklabels(features[sorted_idx])
ax2.set_title("Feature Importance")

# Prediction Distribution
fig3, ax3 = plt.subplots()
ax3.hist(y, bins=30, alpha=0.5, label="Actual", color='skyblue')
ax3.hist(y_pred, bins=30, alpha=0.5, label="Predicted", color='orange')
ax3.legend()
ax3.set_title("Prediction Distribution")

# --- Prepare car listing dataset ---
cars_df["Price"] = cars_df["Price (in USD)"].str.replace(",", "", regex=False).astype(float)
cars_df["Car Name"] = cars_df["Car Make"] + " " + cars_df["Car Model"]
cars_df["Brand"] = cars_df["Car Make"]
cars_df["Fuel Type"] = "N/A"

# --- Display Metrics and Visuals ---
st.subheader("📊 Model Evaluation")
st.write(f"**R² Score:** {r2:.4f}")
st.write(f"**Mean Squared Error:** {mse:,.2f}")

with st.expander("📉 See Model Visualizations"):
    st.pyplot(fig1)
    st.pyplot(fig2)
    st.pyplot(fig3)

# --- User input ---
st.subheader("🧾 Estimate Your Car Purchase Amount")

gender_input = st.selectbox("Gender", ["Male", "Female"])
gender = 0 if gender_input == "Male" else 1

age = st.number_input("Age", min_value=18, max_value=100, value=30)
annual_salary = st.number_input("Annual Salary ($)", min_value=10000, value=50000)
credit_card_debt = st.number_input("Credit Card Debt ($)", min_value=0, value=5000)
net_worth = st.number_input("Net Worth ($)", min_value=0, value=100000)

if st.button("Predict and Show Car Recommendations"):
    input_data = np.array([[gender, age, annual_salary, credit_card_debt, net_worth]])
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)[0]

    st.success(f"✅ Estimated Car Purchase Amount: **${prediction:,.2f}**")

    # Filter recommended cars
    lb, ub = prediction * 0.9, prediction * 1.1
    recommended_cars = cars_df[(cars_df["Price"] >= lb) & (cars_df["Price"] <= ub)]

    st.subheader("🚘 Recommended Cars Within Your Budget")

    if recommended_cars.empty:
        st.warning("No cars found in your estimated price range.")
    else:
        # --- Filters ---
        st.markdown("#### 🔎 Filter Options")
        selected_brands = st.multiselect("Filter by Brand", recommended_cars["Brand"].unique().tolist())
        min_p = int(recommended_cars["Price"].min())
        max_p = int(recommended_cars["Price"].max())
        price_range = st.slider("Price Range ($)", min_p, max_p, (min_p, max_p))

        filtered = recommended_cars[
            (recommended_cars["Price"] >= price_range[0]) &
            (recommended_cars["Price"] <= price_range[1])
        ]

        if selected_brands:
            filtered = filtered[filtered["Brand"].isin(selected_brands)]

        # Display filtered results
        if filtered.empty:
            st.warning("No cars match your filter criteria.")
        else:
            for _, row in filtered.iterrows():
                st.markdown(
                    f"**{row['Car Name']}** (${row['Price']:,.0f})  \n"
                    f"**Brand**: {row['Brand']}  \n"
                    f"**Fuel Type**: {row['Fuel Type']}"
                )
                st.markdown("---")
