# 🚗 EV Charging Station Optimizer

**Stochastic Loss System • Automatic Optimal Configuration + Sensitivity Analysis**

### 🌐 **Live Demo** 


→ **[Run EV Charging Optimizer](https://vghcx9fmr5x6wftvkf6hfv.streamlit.app/)**

. **Κατέβασε το dataset** (120 MB):
   - Go to: [Synthetic EV Data - Kaggle](https://www.kaggle.com/datasets/ahmedess/synthetic-ev-data)
   - Download `SYNTHETIC_EV_DATA.csv` and upload it on the app

# ⚡ EV Charging Station Optimization using Stochastic Simulation

## 📌 Overview
This project develops a **data-driven decision framework** for optimizing the number of chargers in an EV charging station.  
Using real-world charging data, stochastic modeling, and economic analysis, the goal is to **maximize Net Annual Profit** while balancing customer service quality and infrastructure costs.

---

## 🎯 Key Result
✅ Optimal number of chargers: **34**  
💰 Net Annual Profit: **€52,334**  
⚡ Payback Period: **5.2 years**

The model identifies the **economic sweet spot** where additional revenue from serving customers equals the marginal cost of adding new chargers.

---

## 🧠 Methodology

The project combines techniques from **data science, stochastic modeling, and optimization**:

- 📊 **Data Analysis**
  - Real EV charging session data
  - Feature extraction (arrival times, durations, energy delivered)

- 🔍 **Clustering (GMM)**
  - Identification of **Low-demand** and **High-demand** days

- ⏱ **Arrival Modeling**
  - Non-Homogeneous Poisson Process (NHPP)
  - Time-dependent arrival rates (peak at 14:00–17:00)

- 🔄 **Simulation (Digital Twin)**
  - Discrete Event Simulation (Monte Carlo)
  - System modeled as **M(t)/G/c/c loss system**

- 💰 **Economic Model**
  - Revenue, electricity cost, maintenance
  - Lost profit from blocked customers
  - Capital cost amortization

- 📈 **Optimization**
  - Exhaustive search over possible charger counts
  - Objective: maximize Net Annual Profit

- 🔎 **Sensitivity Analysis**
  - ±20% variation in key parameters
  - Identification of most critical business drivers

---

## 📊 Key Insights

- 📉 Profit follows a **concave curve** with a clear global maximum  
- ⚠️ Too few chargers → high customer blocking (lost revenue)  
- 💸 Too many chargers → underutilized capital and high costs  
- 🔥 **Revenue per kWh** is the most critical profitability driver  
- 🛡 The optimal solution is **robust** across different scenarios  

---

## ⚠️ Limitations

- The dataset includes only **served customers** (censored demand)
- Actual demand may be higher during peak hours
- Results should be interpreted as **scenario-based insights**

👉 To address this, an interactive tool allows testing:
- “What if demand is +10% or +20%?”

---

## 🖥 Interactive Dashboard

An interactive **Streamlit app** is included, allowing users to:

- Adjust economic parameters
- Simulate different demand scenarios
- Instantly see:
  - Optimal number of chargers
  - Expected profit
  - System performance metrics

---

![optimal34](https://github.com/user-attachments/assets/939c5cdf-be65-4693-b741-7d1d7ef895f5)
![tornado](https://github.com/user-attachments/assets/a2f798e7-3431-46d4-b4c1-b86104c2a5b7)

