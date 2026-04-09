import streamlit as st                                      # Imports the Streamlit library for building the web app
import numpy as np                                          # Imports NumPy for numerical operations and random number generation
import pandas as pd                                         # Imports pandas for data manipulation and analysis
import matplotlib.pyplot as plt                             # Imports matplotlib for creating plots and charts
from heapq import heappush, heappop                         # Imports heap functions for priority queue (event scheduling)
import time                                                 # Imports time module to measure execution time

st.set_page_config(page_title="EV Charging Optimizer", layout="wide")   # Configures the Streamlit page title and sets wide layout
st.title("🚗 EV Charging Station Optimizer")                          # Displays the main title at the top of the app
st.markdown("**Stochastic Loss System • Automatic Optimal Configuration + Sensitivity Analysis**")  # Displays a subtitle with markdown formatting

# ====================== DATA LOADING ======================
st.sidebar.header("📁 Data Input")                                    # Creates a header in the sidebar for the data input section

@st.cache_data(show_spinner=False)                                    # Decorator that caches the function output to avoid reloading data every time
def load_ev_data(uploaded_file=None):                                 # Defines a function to load EV data, with optional uploaded file
    if uploaded_file is not None:                                     # Checks if user has uploaded a file
        df = pd.read_csv(uploaded_file)                               # Reads the uploaded CSV file into a pandas DataFrame
        st.sidebar.success("✅ Uploaded file loaded and cached")      # Shows success message in sidebar
    else:
        try:
            df = pd.read_csv("SYNTHETIC_EV_DATA.csv")                 # Attempts to load the default local CSV file
            st.sidebar.success("✅ Local dataset loaded and cached")   # Shows success message
        except FileNotFoundError:
            st.error("❌ File 'SYNTHETIC_EV_DATA.csv' not found.\n"
                     "Please place it in the same folder as the app or upload it.")  # Shows error message
            st.stop()                                                 # Stops the Streamlit app execution

    needed_cols = ['dayIndicator', 'chargingDuration', 'connectionTime_decimal', 'kWhDelivered']  # Defines the list of columns needed
    df = df[needed_cols].copy()                                       # Keeps only the required columns and makes a copy
    return df                                                         # Returns the filtered DataFrame


uploaded_file = st.sidebar.file_uploader(                             # Creates a file uploader widget in the sidebar
    "Upload SYNTHETIC_EV_DATA.csv (120MB)",
    type=["csv"],
    help="For production, just place the file in the app folder"
)

df = load_ev_data(uploaded_file)                                      # Calls the function to load the data

if 'hour' not in df.columns:                                          # Checks if the 'hour' column is missing
    df['hour'] = np.floor(df['connectionTime_decimal']).astype(int)   # Creates an integer hour column from decimal time

# ============================ GMM CLUSTERING ============================
# Section Summary: This section loads daily customer counts, performs Gaussian Mixture clustering to separate low and high demand days,
# and prepares separate datasets and duration distributions for each type of day.

daily_customers = df.groupby('dayIndicator').size().reset_index(name='num_customers')  # Counts total customers per day
X = daily_customers[['num_customers']].values                         # Converts customer counts to a numpy array for clustering

from sklearn.mixture import GaussianMixture                           # Imports GaussianMixture model from scikit-learn

gmm = GaussianMixture(n_components=2, random_state=42)                # Creates a GMM model with 2 clusters and fixed random seed
daily_customers['cluster'] = gmm.fit_predict(X)                       # Fits the model and assigns cluster labels to each day

means = daily_customers.groupby('cluster')['num_customers'].mean()    # Calculates the mean number of customers per cluster
low_label = means.idxmin()                                            # Identifies which cluster has fewer customers (low demand)
high_label = 1 - low_label                                            # Assigns the other cluster as high demand

daily_customers['type'] = daily_customers['cluster'].map({low_label: 'Low-demand', high_label: 'High-demand'})  # Maps cluster numbers to readable labels

P_LOW = (daily_customers['type'] == 'Low-demand').mean()              # Calculates the proportion of low-demand days
P_HIGH = 1 - P_LOW                                                    # Calculates the proportion of high-demand days

low_days = daily_customers[daily_customers['type'] == 'Low-demand']['dayIndicator'].values   # Extracts list of low-demand day IDs
high_days = daily_customers[daily_customers['type'] == 'High-demand']['dayIndicator'].values  # Extracts list of high-demand day IDs

df_low = df[df['dayIndicator'].isin(low_days)].copy()                 # Filters the main dataframe for low-demand days only
df_high = df[df['dayIndicator'].isin(high_days)].copy()               # Filters the main dataframe for high-demand days only

low_durations = df_low['chargingDuration'].dropna().values            # Extracts charging duration values for low-demand days
high_durations = df_high['chargingDuration'].dropna().values          # Extracts charging duration values for high-demand days

low_durations = low_durations[(low_durations > 0) & (low_durations < 48)]   # Removes unrealistic durations (negative or >48 hours)
high_durations = high_durations[(high_durations > 0) & (high_durations < 48)] # Same cleaning for high-demand durations

kwh_values = df['kWhDelivered'].dropna().values                       # Extracts all real kWh delivered values for later random sampling

st.sidebar.success(f"✅ Using real kWhDelivered data ({len(kwh_values):,} samples)")  # Displays success message with sample count
st.sidebar.success(f"GMM Clustering: Low = {P_LOW:.1%} | High = {P_HIGH:.1%}")      # Displays the percentage of low/high demand days

# ====================== SIDEBAR PARAMETERS ======================
# Section Summary: This section creates all the user-adjustable economic and simulation parameters in the sidebar.

st.sidebar.header("Economic Parameters")                              # Creates a header in the sidebar for parameters

inst_cost = st.sidebar.number_input("Installation Cost per Charger (€)", value=8000, step=500)   # Input box for installation cost
lifetime = st.sidebar.slider("Charger Lifetime (years)", 5, 15, 10)   # Slider for charger lifetime in years
revenue_kwh = st.sidebar.number_input("Revenue per kWh (€)", value=0.55, step=0.01)            # Input for revenue per kWh
elec_cost = st.sidebar.number_input("Electricity Cost per kWh (€)", value=0.35, step=0.01)     # Input for electricity cost per kWh
maint_cost = st.sidebar.number_input("Annual Maintenance per Charger (€)", value=800, step=50)  # Input for annual maintenance cost
lost_per_block = st.sidebar.number_input("Lost Profit per Blocked Customer (€)", value=8.0, step=1.0)  # Input for lost profit per blocked customer
num_sim_days = st.sidebar.slider("Simulation Days (per configuration)", 300, 2000, 600, step=100)  # Slider for number of simulation days

# ====================== ARRIVAL RATES ======================
# Section Summary: Defines the hourly arrival rates (lambda) for low-demand and high-demand days.

lambda_low = np.array([2.227, 2.464, 1.991, 1.514, 1.342, 1.231, 1.118, 1.045,
                       1.006, 1.03, 1.062, 1.155, 1.308, 1.616, 2.923, 6.408,
                       7.548, 4.152, 2.217, 2.273, 2.208, 2.154, 2.152, 1.642])  # Array of hourly arrival rates for low-demand days

lambda_high = np.array([4.094, 4.724, 3.433, 2.255, 1.768, 1.49, 1.253, 1.113,
                        1.058, 1.053, 1.122, 1.33, 1.733, 2.487, 5.657, 12.895,
                        14.815, 7.842, 3.822, 4.148, 4.011, 3.782, 3.834, 2.539])  # Array of hourly arrival rates for high-demand days


# ====================== SIMULATE DAY ======================
# Section Summary: This function simulates one full day of operation using discrete event simulation with a priority queue.

def simulate_day(c, is_high_day):                                     # Defines function to simulate one day with c chargers and day type
    lambda_day = lambda_high if is_high_day else lambda_low           # Selects the appropriate arrival rate array
    durations = high_durations if is_high_day else low_durations      # Selects the appropriate charging duration distribution
    max_lambda = np.max(lambda_day) + 0.01                            # Adds small buffer to max lambda for thinning method

    event_queue = []                                                  # Initializes empty priority queue for arrival events
    stats = {'arrivals': 0, 'served': 0, 'blocked': 0, 'total_kwh': 0.0, 'busy_time': 0.0}  # Creates dictionary to store daily statistics
    server_finish = [0.0] * c                                         # Initializes finish times for each charger (all free at t=0)
    t = 0.0                                                           # Initializes simulation clock at time 0

    while t < 24:                                                     # Generates arrivals until end of day (24 hours)
        t += np.random.exponential(1.0 / max_lambda)                  # Advances time using exponential inter-arrival
        if t >= 24: break                                             # Stops if time exceeds 24 hours
        hour = int(t) % 24                                            # Gets current hour of the day
        if np.random.rand() < lambda_day[hour] / max_lambda:          # Thinning acceptance check
            heappush(event_queue, (t, stats['arrivals']))             # Adds accepted arrival to priority queue
            stats['arrivals'] += 1                                    # Increments total arrivals counter

    while event_queue:                                                # Processes all events in the queue
        t, _ = heappop(event_queue)                                   # Gets next arrival event
        if t >= 24: break                                             # Ignores events after 24 hours

        if min(server_finish) <= t:                                   # Checks if at least one charger is free
            idx = np.argmin(server_finish)                            # Finds the first available charger
            service_time = np.clip(np.random.choice(durations), 0.5, 48)  # Samples and clips service time
            kwh = np.random.choice(kwh_values)                        # Samples real kWh value

            server_finish[idx] = t + service_time                     # Updates charger's finish time
            stats['served'] += 1                                      # Increments served customers
            stats['total_kwh'] += kwh                                 # Adds kWh delivered
            stats['busy_time'] += service_time                        # Adds busy time
        else:
            stats['blocked'] += 1                                     # Increments blocked customers if all chargers busy

    return stats                                                      # Returns statistics for the simulated day


# ====================== HELPER OPTIMIZATION (Fast Version) ======================
# Section Summary: Fast helper function used in sensitivity analysis to quickly find best c and profit for different parameters.

def run_optimization_fast(num_days, p_high, rev_kwh, e_cost, i_cost, l_block):
    results = []                                                      # List to store results for different charger counts
    for c in range(20, 71, 2):                                        # Loops over charger counts from 20 to 70 with step 2
        total_kwh = total_blocked = total_busy = total_arrivals = 0.0 # Resets cumulative counters
        for _ in range(num_days):                                     # Runs simulation for specified number of days
            is_high = np.random.rand() < p_high                       # Randomly decides if the day is high-demand
            stats = simulate_day(c, is_high)                          # Simulates one day
            total_kwh += stats['total_kwh']                           # Accumulates kWh
            total_blocked += stats['blocked']                         # Accumulates blocked customers
            total_busy += stats['busy_time']                          # Accumulates busy time
            total_arrivals += stats['arrivals']                       # Accumulates arrivals

        scale = 365.0 / num_days                                      # Scaling factor to annualize results
        annual_kwh = total_kwh * scale                                # Annual kWh delivered
        annual_blocked = total_blocked * scale                        # Annual blocked customers

        net_profit = (annual_kwh * rev_kwh - annual_kwh * e_cost -    # Calculates net annual profit
                      c * maint_cost - annual_blocked * l_block -
                      (c * i_cost) / lifetime)

        results.append({'c': c, 'net_profit': net_profit})            # Stores result for this c

    opt_df = pd.DataFrame(results)                                    # Converts results to DataFrame
    best = opt_df.loc[opt_df['net_profit'].idxmax()]                  # Finds the configuration with maximum profit
    return int(best['c']), round(best['net_profit'], 0)               # Returns best number of chargers and rounded profit


# ====================== MAIN OPTIMIZATION ======================
# Section Summary: Main optimization block that runs when user clicks the button. It tests many configurations, shows results, chart, and metrics.

if st.sidebar.button("🚀 Run Optimization", type="primary"):          # Runs this block when "Run Optimization" button is clicked
    progress_bar = st.progress(0)                                     # Creates progress bar
    status_text = st.empty()                                          # Placeholder for status text
    results = []                                                      # List to store all simulation results
    start_time = time.time()                                          # Records start time for performance measurement

    for i, c in enumerate(range(20, 71)):                             # Loops through charger counts 20 to 70
        total_kwh = total_blocked = total_busy = total_arrivals = 0.0 # Resets accumulators
        for _ in range(num_sim_days):                                 # Runs full simulation for selected number of days
            is_high = np.random.rand() < P_HIGH                       # Decides day type
            stats = simulate_day(c, is_high)                          # Simulates the day
            total_kwh += stats['total_kwh']
            total_blocked += stats['blocked']
            total_busy += stats['busy_time']
            total_arrivals += stats['arrivals']

        scale = 365.0 / num_sim_days                                  # Annual scaling factor
        annual_kwh = total_kwh * scale
        annual_blocked = total_blocked * scale

        net_profit = (annual_kwh * revenue_kwh - annual_kwh * elec_cost -
                      c * maint_cost - annual_blocked * lost_per_block -
                      (c * inst_cost) / lifetime)                     # Calculates net profit

        results.append({                                              # Stores detailed results for this configuration
            'c': c,
            'net_profit': net_profit,
            'blocking_prob': total_blocked / total_arrivals if total_arrivals > 0 else 0,
            'utilization': total_busy / (c * 24 * num_sim_days),
            'annual_kwh': annual_kwh
        })

        progress_bar.progress((i + 1) / 51)                           # Updates progress bar
        status_text.text(f"Testing {c} chargers... ({i + 1}/51)")     # Updates status message

    progress_bar.empty()                                              # Removes progress bar
    status_text.empty()                                               # Removes status text

    opt_df = pd.DataFrame(results)                                    # Creates DataFrame from all results
    best = opt_df.loc[opt_df['net_profit'].idxmax()]                  # Finds the best configuration

    st.success(f"🎯 **Optimal Number of Chargers: {int(best['c'])}**") # Displays optimal result

    col1, col2, col3, col4 = st.columns(4)                            # Creates 4 columns for metrics
    col1.metric("Blocking Probability", f"{best['blocking_prob']:.2%}")
    col2.metric("Utilization Rate", f"{best['utilization']:.2%}")
    col3.metric("Net Annual Profit", f"€{best['net_profit']:,.0f}")

    payback = (best['c'] * inst_cost) / best['net_profit'] if best['net_profit'] > 0 else float('inf')  # Calculates payback period
    col4.metric("Payback Period", f"{payback:.1f} years" if payback != float('inf') else "Not profitable")

    fig, ax = plt.subplots(figsize=(10, 5))                           # Creates profit vs chargers plot
    ax.plot(opt_df['c'], opt_df['net_profit'], marker='o', linewidth=2.5)
    ax.axvline(x=best['c'], color='green', linestyle='--', linewidth=2, label=f'Optimal = {int(best["c"])}')
    ax.set_xlabel("Number of Chargers")
    ax.set_ylabel("Net Annual Profit (€)")
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)                                                    # Displays the plot

    with st.expander("📊 Full Results Table"):                        # Creates expandable section with full table
        st.dataframe(opt_df.style.format({
            'net_profit': '€{:,.0f}',
            'blocking_prob': '{:.3%}',
            'utilization': '{:.2%}'
        }), use_container_width=True)

    st.caption(f"✅ Done in {time.time() - start_time:.1f}s")         # Shows total execution time


# ====================== SENSITIVITY ANALYSIS  ======================
# Section Summary: Runs sensitivity analysis by varying key parameters ±20% and shows impact on profit and optimal charger count.

if st.sidebar.button("📊 Run Sensitivity Analysis", type="secondary"): # Runs when Sensitivity button is clicked
    with st.spinner("Running Sensitivity Analysis (fast version)..."): # Shows spinner during execution
        start_time = time.time()                                      # Records start time

        params = {                                                    # Dictionary of parameters to test
            "Revenue per kWh": revenue_kwh,
            "Electricity Cost": elec_cost,
            "Installation Cost": inst_cost,
            "Lost Profit per Block": lost_per_block
        }

        sensitivity_results = []                                      # List to store sensitivity results
        variations = [-0.20, 0.20]                                    # ±20% variations

        for param_name, base_value in params.items():                 # Loops over each parameter
            profits = []
            opt_cs = []

            for v in variations:                                      # Tests both -20% and +20%
                new_value = base_value * (1 + v)                      # Calculates new parameter value

                best_c, best_profit = run_optimization_fast(          # Runs fast optimization with modified parameter
                    num_days=int(num_sim_days * 0.3),                 # Uses fewer days for speed
                    p_high=P_HIGH,
                    rev_kwh=new_value if param_name == "Revenue per kWh" else revenue_kwh,
                    e_cost=new_value if param_name == "Electricity Cost" else elec_cost,
                    i_cost=new_value if param_name == "Installation Cost" else inst_cost,
                    l_block=new_value if param_name == "Lost Profit per Block" else lost_per_block
                )

                profits.append(best_profit)
                opt_cs.append(best_c)

            sensitivity_results.append({                              # Stores results for this parameter
                "Parameter": param_name,
                "Base Value": round(base_value, 3),
                "-20% Profit": round(profits[0], 0),
                "+20% Profit": round(profits[1], 0),
                "Profit Swing (€)": round(max(profits) - min(profits), 0),
                "Optimal c Range": f"{min(opt_cs)}–{max(opt_cs)}"
            })

        sens_df = pd.DataFrame(sensitivity_results)                   # Converts to DataFrame

        st.subheader("📊 Sensitivity Analysis Results")               # Displays title
        st.dataframe(sens_df.style.format({                           # Displays formatted table
            "Base Value": "{:.3f}",
            "-20% Profit": "€{:,.0f}",
            "+20% Profit": "€{:,.0f}",
            "Profit Swing (€)": "€{:,.0f}"
        }), use_container_width=True)

        # Tornado Chart
        fig, ax = plt.subplots(figsize=(10, 6))                       # Creates tornado chart
        sorted_df = sens_df.sort_values("Profit Swing (€)", ascending=True)
        y_pos = np.arange(len(sorted_df))
        ax.barh(y_pos, sorted_df["Profit Swing (€)"], color='skyblue', edgecolor='black')
        ax.set_yticks(y_pos)
        ax.set_yticklabels(sorted_df["Parameter"])
        ax.set_xlabel("Profit Swing (€)")
        ax.set_title("Tornado Chart - Sensitivity Analysis (±20%)")
        ax.grid(True, axis='x')
        st.pyplot(fig)                                                # Displays tornado chart

        st.success(f"Sensitivity Analysis completed in {time.time() - start_time:.1f} seconds")  # Success message

st.caption("Fast Sensitivity Analysis • 30% simulation days • Ready for demonstration")  # Final caption at the bottomσ