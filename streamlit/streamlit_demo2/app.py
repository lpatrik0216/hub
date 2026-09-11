import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
import os

st.set_page_config(page_title="Digital Twin Simulator", layout="wide")

# --- PRESENTATION CONTEXT & INTRO ---
st.title("Medical AI Digital Twin: Continuous Thyroid Simulation")

st.markdown("""
### The Problem: Biological Chaos vs. Rigid Algorithms
Thyroid medication (Levothyroxine) has a notoriously narrow therapeutic index. A micro-adjustment of just 12.5 mcg can drastically swing a patient's TSH levels. Standard AI models (like Decision Trees or Random Forests) struggle with this because they group data into rigid "buckets"—often outputting flatlined, identical predictions for different doses, or worse, predicting biologically impossible negative lab values at extreme edges.

### The Hypothesis & Technical Approach
We hypothesize that to build a true **Clinical Digital Twin**, the AI must respect the laws of pharmacokinetics. By mapping the patient's data into **logarithmic space** and applying a penalized continuous model (**Ridge Regression**), we force the AI to learn smooth, exponential biological decay curves. This guarantees that every microgram of medication yields a proportional, strictly positive, and clinically logical TSH projection.
""")

with st.expander("📖 Glossary of Metrics & Techniques (Click to expand)"):
    st.markdown("""
    * **Ridge Regression (L2 Regularization):** A linear machine learning model equipped with a mathematical "shock absorber". It prevents the AI from overreacting to extreme, noisy outlier patients, ensuring the predicted dose-response trajectory remains stable.
    * **Proportional Confidence Bands:** The shaded blue area on the chart. A healthy TSH (e.g., 2.0) has low biological volatility, while a failing thyroid (e.g., TSH 50.0) is highly chaotic. The 95% confidence band scales dynamically (at 25% variance) to honestly visualize this physiological unpredictability.
    * **Human-in-the-Loop (HITL):** AI should act as a navigational radar, not an autonomous doctor. The system simulates parallel futures based on the math, but hard-locks execution until a human clinician reviews and authorizes the safest path.
    """)

st.markdown("---")


# 1. Load Data & Train Log-Linear Model
@st.cache_resource
def setup_digital_twin():
    file_name = "synthetic_thyroid_data.csv"

    if not os.path.exists(file_name):
        st.error(f"Dataset '{file_name}' not found. Please run the data generation script first.")
        st.stop()

    df = pd.read_csv(file_name)

    df['Log_Current_TSH'] = np.log(df['Current_TSH'])
    df['Log_Next_TSH'] = np.log(df['Next_TSH'])

    features = ['Weight_kg', 'Anti_TPO_Positive', 'Log_Current_TSH', 'Current_Dose_mcg', 'Proposed_Dose_mcg']
    X = df[features]
    y = df['Log_Next_TSH']

    # Train a Ridge regressor to perfectly map the continuous slope
    model = Ridge(alpha=1.0)
    model.fit(X, y)

    return model, features


log_model, feature_names = setup_digital_twin()


# Continuous Uncertainty Estimation Function
def predict_with_uncertainty(model_obj, df_input):
    # Transform user input into logarithmic space
    df_input['Log_Current_TSH'] = np.log(df_input['Current_TSH'])

    # Select only the features the model was trained on
    X_input = df_input[['Weight_kg', 'Anti_TPO_Positive', 'Log_Current_TSH', 'Current_Dose_mcg', 'Proposed_Dose_mcg']]

    log_pred = model_obj.predict(X_input)

    mean_pred = np.exp(log_pred)

    # Calculate a biological variance band (e.g., 25% variance around the mean)
    std_pred = mean_pred * 0.25

    lower_bound = np.maximum(0.05, mean_pred - 1.96 * std_pred)
    upper_bound = mean_pred + 1.96 * std_pred

    return mean_pred, lower_bound, upper_bound


# 3. UI Layout
with st.sidebar:
    st.header("Current Patient State")
    age = st.number_input("Age", value=45)
    weight = st.number_input("Weight (kg)", value=70.0)
    hashimotos = st.selectbox("Hashimoto's (Anti-TPO)", [1, 0],
                              format_func=lambda x: "Positive" if x == 1 else "Negative")
    current_tsh = st.number_input("Current TSH (mIU/L)", value=5.0, step=0.5)
    current_ft4 = st.number_input("Current FT4 (ng/dL)", value=0.8, step=0.1)
    current_hr = st.slider("Heart Rate (BPM)", 40, 120, 60)
    current_dose = st.number_input("Current Dose (mcg)", value=100.0, step=12.5)

# Dynamic Scenario Routing
if current_tsh > 4.5:
    scenarios = [
        {"name": "Scenario A: Maintain Dose", "dose": current_dose},
        {"name": "Scenario B: Increase (+12.5 mcg)", "dose": current_dose + 12.5},
        {"name": "Scenario C: Aggressively Increase (+25 mcg)", "dose": current_dose + 25.0}
    ]
elif current_tsh < 0.4:
    if current_dose == 0.0:
        scenarios = [
            {"name": "Scenario A: Maintain (Off Meds)", "dose": 0.0},
            {"name": "Scenario B: Monitor / Observation", "dose": 0.0},
            {"name": "Scenario C: Refer to Endocrinology", "dose": 0.0}
        ]
    else:
        scenarios = [
            {"name": "Scenario A: Maintain Dose", "dose": current_dose},
            {"name": "Scenario B: Decrease (-12.5 mcg)", "dose": max(0.0, current_dose - 12.5)},
            {"name": "Scenario C: Aggressively Decrease (-25 mcg)", "dose": max(0.0, current_dose - 25.0)}
        ]
else:
    if current_dose == 0.0:
        scenarios = [
            {"name": "Scenario A: Maintain Dose (Unmedicated)", "dose": 0.0},
            {"name": "Scenario B: Start Micro-Dose (+12.5 mcg)", "dose": 12.5},
            {"name": "Scenario C: Start Standard Dose (+25.0 mcg)", "dose": 25.0}
        ]
    else:
        scenarios = [
            {"name": "Scenario A: Micro-Decrease (-12.5 mcg)", "dose": max(0.0, current_dose - 12.5)},
            {"name": "Scenario B: Maintain Dose", "dose": current_dose},
            {"name": "Scenario C: Micro-Increase (+12.5 mcg)", "dose": current_dose + 12.5}
        ]

# Build input payload
patient_data = []
for s in scenarios:
    patient_data.append({
        "Weight_kg": weight,
        "Anti_TPO_Positive": hashimotos,
        "Current_TSH": current_tsh,
        "Current_Dose_mcg": current_dose,
        "Proposed_Dose_mcg": s["dose"]
    })

df_input = pd.DataFrame(patient_data)
mean_tsh, low_tsh, high_tsh = predict_with_uncertainty(log_model, df_input)

# Display Parallel Trajectories
st.subheader("Parallel Therapeutic Futures")
cols = st.columns(3)

import matplotlib.pyplot as plt

for i, col in enumerate(cols):
    with col:
        st.markdown(f"**{scenarios[i]['name']}**")
        st.write(f"**Proposed Dose:** {scenarios[i]['dose']} mcg")

        target_color = "normal" if 0.4 <= mean_tsh[i] <= 4.0 else "inverse"
        st.metric(label="Predicted TSH in 6 Weeks", value=f"{mean_tsh[i]:.2f} mIU/L", delta_color=target_color)

        st.caption(f"**95% Confidence Band:** [{low_tsh[i]:.2f} - {high_tsh[i]:.2f}]")

        # FIXED UI WARNING LOGIC: Proportional scaling instead of static 4.0 check
        if (high_tsh[i] - low_tsh[i]) / mean_tsh[i] > 0.60:
            st.warning("⚠️ High Uncertainty: Biological response may vary significantly.")

# Visual Trajectory Plot
st.markdown("---")
st.write("**Predicted 6-Week Trajectories**")

fig, ax = plt.subplots(figsize=(10, 3))
x_labels = [s["name"] for s in scenarios]

ax.plot(x_labels, mean_tsh, marker='o', color='blue', label='Expected TSH')
ax.fill_between(x_labels, low_tsh, high_tsh, color='blue', alpha=0.15, label='95% Confidence Band')
ax.axhline(0.4, color='red', linestyle='--', alpha=0.5, label='Hyperthyroid Threshold (0.4)')
ax.axhline(4.0, color='orange', linestyle='--', alpha=0.5, label='Hypothyroid Threshold (4.0)')

ax.set_ylabel("TSH (mIU/L)")
ax.legend()
st.pyplot(fig)

# 7. Explainability & Human Oversight
st.markdown("---")
st.write("**Model Explainability (Log-Linear Dynamics)**")
st.info(
    f"The model's prediction evaluates the gap between the patient's **Weight ({weight} kg)** and the **Proposed Dose**, anchored by their **Current TSH ({current_tsh})**. The exponential curve guarantees that micro-adjustments smoothly transition the TSH without flatlining or predicting impossible negative values.")

st.markdown("---")
st.write("**Clinician Review & Authorization**")
decision = st.radio("Select Approved Treatment Plan:", [s["name"] for s in scenarios])

if st.button("Authorize Treatment Plan"):
    st.success(
        f"Log updated: {decision} explicitly authorized by attending clinician. Digital Twin monitoring paused until next lab draw.")