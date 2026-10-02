import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from models import train_predict_zscore, train_predict_isolation_forest, train_predict_pca

# ==========================================
# 1. PAGE CONFIGURATION & STYLING
# ==========================================
st.set_page_config(
    page_title="ProductionPulse | Industrial Anomaly Intelligence",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    [data-testid="stHeader"] { visibility: hidden !important; height: 0px !important; }
    footer { visibility: hidden !important; }
    .block-container { padding-top: 2rem !important; padding-bottom: 2rem !important; }
    </style>
""", unsafe_allow_html=True)


# ==========================================
# 2. DATA INGESTION
# ==========================================
@st.cache_data
def load_skab_data(filename: str) -> pd.DataFrame:
    df = pd.read_csv(f"practical_data/{filename}", sep=";")
    df['datetime'] = pd.to_datetime(df['datetime'])

    # SAFETY CATCH: The 'anomaly-free.csv' lacks ground-truth columns.
    # If missing, populate it with 0s (Normal) so Tab 2 math doesn't crash.
    if 'anomaly' not in df.columns:
        df['anomaly'] = 0

    return df


# ==========================================
# 3. SIDEBAR CONTROLS
# ==========================================
with st.sidebar:
    st.header("Equipment Control Room")
    st.caption("SKAB Testbed Monitoring")

    scenario_map = {
        "1. Normal Operation (Baseline Test)": "anomaly-free.csv",
        "2. Sudden Valve Blockage (Obvious Fault)": "valve.csv",
        "3. Complex Motor Drift (Multivariate Fault)": "other.csv"
    }

    scenario_selection = st.selectbox("Equipment Failure Scenario", list(scenario_map.keys()), index=1)

    # Load data first to set dynamic slider bounds
    selected_file = scenario_map[scenario_selection]
    raw_data = load_skab_data(selected_file)
    max_data_length = len(raw_data)

    # Calculate half of the data length, rounded to the nearest multiple of 25
    max_slider_val = int(round((max_data_length / 2) / 25) * 25)

    st.divider()
    st.subheader("Model Configuration")

    active_model = st.selectbox(
        "Detection Algorithm",
        ["Isolation Forest", "PCA Reconstruction Error", "Rolling Z-Score"]
    )

    sensitivity = st.slider("Detection Sensitivity (Contamination)", 0.01, 0.15, 0.01, 0.01)

    training_cutoff = st.slider(
        "Normalcy Training Window (Rows)",
        min_value=100,
        max_value=max_slider_val,
        value=max_slider_val,
        step=25
    )

# ==========================================
# 4. MODEL EXECUTION PIPELINE
# ==========================================
with st.spinner(f"Running {active_model} on live telemetry..."):
    if active_model == "Rolling Z-Score":
        processed_data = train_predict_zscore(raw_data, training_cutoff=training_cutoff, z_threshold=4.0)
    elif active_model == "Isolation Forest":
        processed_data = train_predict_isolation_forest(raw_data, training_cutoff=training_cutoff,
                                                        contamination=sensitivity)
    elif active_model == "PCA Reconstruction Error":
        processed_data = train_predict_pca(raw_data, training_cutoff=training_cutoff, n_components=2,
                                           contamination=sensitivity)

# ==========================================
# 5. MAIN DASHBOARD HEADER
# ==========================================
st.title("ProductionPulse")
st.markdown("### Multivariate Time-Series Anomaly Intelligence")

kpi1, kpi2, kpi3 = st.columns(3)
kpi1.metric("Monitored Sensors", f"{len(raw_data.columns) - 3} Signals")
kpi2.metric("Telemetry Status", "ONLINE")
kpi3.metric("Active Model", active_model)

st.divider()

# ==========================================
# 6. TABBED WORKFLOW
# ==========================================
tab_control, tab_diag, tab_cfo = st.tabs([
    "Control Room (Live Monitoring)",
    "Diagnostic Lab (Model Explainability)",
    "CFO's Desk (ROI Analysis)"
])

# ------------------------------------------
# TAB 1: CONTROL ROOM (VISUALIZATION)
# ------------------------------------------
with tab_control:
    st.subheader("Real-Time Telemetry Feed")

    c1, c2 = st.columns([3, 1])

    with c1:
        fig = go.Figure()

        if 'Pressure' in processed_data.columns:
            fig.add_trace(go.Scatter(x=processed_data['datetime'], y=processed_data['Pressure'],
                                     name='Pressure', line=dict(color='#2563EB', width=2)))

        if 'Temperature' in processed_data.columns:
            fig.add_trace(go.Scatter(x=processed_data['datetime'], y=processed_data['Temperature'],
                                     name='Temperature', line=dict(color='#F59E0B', width=2)))

        # Handle safe cutoff and training window shading
        safe_cutoff = min(training_cutoff, len(processed_data) - 1)
        train_end_time = processed_data['datetime'].iloc[safe_cutoff]

        fig.add_vrect(
            x0=processed_data['datetime'].iloc[0],
            x1=train_end_time,
            fillcolor="#E2E8F0",
            opacity=0.5,
            layer="below",
            line_width=0,
            annotation_text="Phase 1: Model Baseline Training",
            annotation_position="top left"
        )

        # MASK ANOMALIES: Only show alerts that happen AFTER the training phase
        testing_data = processed_data[processed_data['datetime'] > train_end_time]
        anomalies = testing_data[testing_data['predicted_anomaly'] == 1]

        if not anomalies.empty:
            fig.add_trace(go.Scatter(
                x=anomalies['datetime'],
                y=anomalies['Pressure'] if 'Pressure' in processed_data.columns else anomalies.iloc[:, 1],
                mode='markers',
                name='AI Anomaly Alert',
                marker=dict(color='red', size=10, symbol='x', line=dict(width=2, color='darkred'))
            ))

        fig.update_layout(
            height=500,
            margin=dict(l=0, r=0, t=30, b=0),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            plot_bgcolor="white",
            hovermode="x unified"
        )
        fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='LightGray')
        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='LightGray')

        st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.subheader("Asset Status")

        # DENSITY DEBOUNCER: Trigger if 5 anomalies occur within any 30-row (30-second) window
        if len(testing_data) > 0:
            # min_periods=1 ensures it doesn't return NaN for the first few rows
            max_density = testing_data['predicted_anomaly'].rolling(window=30, min_periods=1).sum().max()
        else:
            max_density = 0

        if pd.isna(max_density):
            max_density = 0

        if max_density >= 5:
            st.error("SUSTAINED SYSTEM ANOMALY DETECTED")
            health_score = 0.42
        else:
            st.success("SYSTEM NORMAL (Minor noise filtered)")
            health_score = 0.98

        st.markdown("**Live Health Score:**")
        st.progress(health_score)

        st.markdown("**Data Summary:**")
        st.write(f"- Baseline Samples: `{safe_cutoff}`")
        st.write(f"- Testing Samples: `{len(processed_data) - safe_cutoff}`")
        st.write(f"- Alerts Triggered: `{len(anomalies)}`")


# ------------------------------------------
# TAB 2: DIAGNOSTIC LAB (EXPLAINABILITY)
# ------------------------------------------
with tab_diag:
    st.subheader("Model Diagnostic & Evaluation")

    # 1. Isolate the testing data to evaluate performance fairly (ignoring training data)
    eval_data = processed_data[processed_data['datetime'] > train_end_time]

    true_anomalies = eval_data['anomaly']
    pred_anomalies = eval_data['predicted_anomaly']

    # Calculate Core Metrics
    tp = ((true_anomalies == 1) & (pred_anomalies == 1)).sum()
    fp = ((true_anomalies == 0) & (pred_anomalies == 1)).sum()
    fn = ((true_anomalies == 1) & (pred_anomalies == 0)).sum()

    precision = tp / (tp + fp) if (tp + fp) > 0 else (1.0 if fp == 0 else 0.0)
    recall = tp / (tp + fn) if (tp + fn) > 0 else 1.0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    # Clean display handling for perfectly healthy datasets
    if (tp + fn) == 0:
        prec_disp = "N/A (Healthy)"
        rec_disp = "N/A (Healthy)"
        f1_disp = "N/A (Healthy)"
    else:
        prec_disp = f"{precision:.2f}"
        rec_disp = f"{recall:.2f}"
        f1_disp = f"{f1_score:.2f}"

    st.markdown("#### Testing Phase Performance Metrics")
    m1, m2, m3 = st.columns(3)
    m1.metric("Precision (Accuracy of Alarms)", prec_disp)
    m2.metric("Recall (Faults Caught)", rec_disp)
    m3.metric("F1-Score", f1_disp)

    st.divider()

    # 2. Timeline Comparison Chart
    st.markdown("#### Whole-Timeline: AI Prediction vs. Ground Truth")
    st.caption("Visualizing the model's binary outputs against the actual SKAB failure labels.")

    fig_diag = go.Figure()

    # Ground Truth Background (Shaded Red)
    fig_diag.add_trace(go.Scatter(
        x=processed_data['datetime'],
        y=processed_data['anomaly'],
        name='Actual Machine Fault',
        fill='tozeroy',
        fillcolor='rgba(239, 68, 68, 0.2)',
        line=dict(color='rgba(239, 68, 68, 0.8)', width=1, shape='hv')
    ))

    # Predicted Anomalies (Step Line)
    fig_diag.add_trace(go.Scatter(
        x=processed_data['datetime'],
        y=processed_data['predicted_anomaly'],
        name='AI Detection',
        mode='lines',
        line=dict(color='#2563EB', width=2, shape='hv')
    ))

    # Highlight the Training Window
    fig_diag.add_vrect(
        x0=processed_data['datetime'].iloc[0],
        x1=train_end_time,
        fillcolor="#E2E8F0",
        opacity=0.5,
        layer="below",
        line_width=0,
        annotation_text="Baseline Training Window",
        annotation_position="top left"
    )

    fig_diag.update_layout(
        height=350,
        margin=dict(l=0, r=0, t=30, b=0),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        plot_bgcolor="white",
        yaxis=dict(tickvals=[0, 1], ticktext=['Normal (0)', 'Anomaly (1)'])
    )
    fig_diag.update_xaxes(showgrid=True, gridwidth=1, gridcolor='LightGray')
    fig_diag.update_yaxes(showgrid=True, gridwidth=1, gridcolor='LightGray')

    st.plotly_chart(fig_diag, use_container_width=True)

    # 3. Explainability: Raw Anomaly Score
    st.markdown("#### Model Explainability: Internal Anomaly Score")
    st.caption(
        "This shows the raw mathematical deviation. When this line crosses the internal threshold, an alert is generated.")

    fig_score = go.Figure()
    fig_score.add_trace(go.Scatter(
        x=processed_data['datetime'],
        y=processed_data['anomaly_score'],
        name='Raw Anomaly Score',
        line=dict(color='#F59E0B', width=1.5)
    ))

    fig_score.add_vrect(
        x0=processed_data['datetime'].iloc[0],
        x1=train_end_time,
        fillcolor="#E2E8F0", opacity=0.5, layer="below", line_width=0
    )

    fig_score.update_layout(
        height=250,
        margin=dict(l=0, r=0, t=10, b=0),
        plot_bgcolor="white",
        showlegend=False
    )
    fig_score.update_xaxes(showgrid=True, gridwidth=1, gridcolor='LightGray')
    fig_score.update_yaxes(showgrid=True, gridwidth=1, gridcolor='LightGray')

    st.plotly_chart(fig_score, use_container_width=True)

with tab_cfo:
    st.subheader("Predictive Maintenance ROI Simulator")
    st.caption("Translates the AI's precision and recall into hypothetical financial impact.")

    # 1. Configurable Financial Parameters
    st.markdown("#### 1. Configure Asset Financials")
    c_param1, c_param2 = st.columns(2)
    cost_missed_fault = c_param1.number_input(
        "Cost of Undetected Fault ($ per second of damage)",
        value=15.00, step=1.00,
        help="How much financial damage occurs for every second the machine runs in a broken state?"
    )
    cost_false_alarm = c_param2.number_input(
        "Cost of False Alarm ($ per unnecessary inspection)",
        value=2.50, step=0.50,
        help="The wasted labor cost when the AI triggers an alert but the machine is healthy."
    )

    # 2. ROI Math
    # Baseline Cost = If we had no AI, we would miss every fault (Run-to-Failure)
    baseline_cost = (tp + fn) * cost_missed_fault

    # AI Cost = The damage done by faults we STILL missed (fn) + the wasted labor of false alarms (fp)
    ai_cost = (fn * cost_missed_fault) + (fp * cost_false_alarm)

    net_savings = baseline_cost - ai_cost

    st.divider()

    # 3. Financial Scorecard
    st.markdown("#### 2. Scenario Financial Impact")
    c1, c2, c3 = st.columns(3)

    c1.metric("Cost without AI (Run-to-Failure)", f"${baseline_cost:,.2f}")
    c2.metric("Cost with AI (Predictive)", f"${ai_cost:,.2f}")

    if net_savings > 0:
        c3.metric("Net Financial Savings", f"${net_savings:,.2f}", delta="Profitable Model")
    elif net_savings < 0:
        c3.metric("Net Financial Savings", f"${net_savings:,.2f}", delta="Money Lost to False Alarms",
                  delta_color="inverse")
    else:
        c3.metric("Net Financial Savings", f"${net_savings:,.2f}", delta="Break Even")

    st.divider()

    # 4. Visual Comparison
    st.markdown("#### 3. Operational Cost Comparison")
    fig_roi = go.Figure()

    fig_roi.add_trace(go.Bar(
        x=['Without AI', 'With AI'],
        y=[baseline_cost, ai_cost],
        marker_color=['#EF4444', '#10B981'],
        text=[f"${baseline_cost:,.0f}", f"${ai_cost:,.0f}"],
        textposition='auto',
        width=0.4
    ))

    fig_roi.update_layout(
        height=400,
        margin=dict(l=0, r=0, t=30, b=0),
        plot_bgcolor="white",
        yaxis_title="Estimated Incident Cost ($)",
        showlegend=False
    )
    fig_roi.update_yaxes(showgrid=True, gridwidth=1, gridcolor='LightGray')

    st.plotly_chart(fig_roi, use_container_width=True)