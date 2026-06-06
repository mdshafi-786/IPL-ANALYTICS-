import os
import streamlit as st
import joblib
import numpy as np
import plotly.graph_objects as go

_DIR = os.path.dirname(os.path.abspath(__file__))

# ── Page Config ──
st.set_page_config(
    page_title="IPL Chase Win Probability",
    page_icon="🏏",
    layout="wide"
)

# ── Custom Styling ──
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;800&display=swap');
    
    /* Global */
    .stApp { font-family: 'Inter', sans-serif; }
    
    /* Header */
    .main-header {
        background: linear-gradient(135deg, #0C0C1D 0%, #1A1A3E 40%, #0F3460 100%);
        padding: 2.5rem 2rem;
        border-radius: 16px;
        margin-bottom: 2rem;
        text-align: center;
        border: 1px solid rgba(233, 69, 96, 0.2);
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);

    }
    .main-header h1 {
        color: #E94560;
        font-size: 2.4rem;
        font-weight: 800;
        margin-bottom: 0.3rem;
        letter-spacing: -0.5px;
    }
    .main-header .subtitle {
        color: #8892B0;
        font-size: 1rem;
        font-weight: 300;
    }
    .main-header .badge {
        display: inline-block;
        background: rgba(233, 69, 96, 0.15);
        color: #E94560;
        padding: 4px 14px;
        border-radius: 20px;
        font-size: 0.8rem;
        margin-top: 0.8rem;
        border: 1px solid rgba(233, 69, 96, 0.3);
    }
    
    /* Gauge */
    .gauge-container {
        text-align: center;
        padding: 2rem 1rem;
    }
    .gauge-circle {
        width: 220px;
        height: 220px;
        border-radius: 50%;
        margin: 0 auto;
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        position: relative;
        box-shadow: 0 0 40px rgba(0,0,0,0.3), inset 0 0 30px rgba(0,0,0,0.2);
        animation: pulse 2s ease-in-out infinite;
    }
    @keyframes pulse {
        0%, 100% { box-shadow: 0 0 40px rgba(0,0,0,0.3), inset 0 0 30px rgba(0,0,0,0.2); }
        50% { box-shadow: 0 0 60px rgba(0,0,0,0.4), inset 0 0 30px rgba(0,0,0,0.2), 0 0 20px var(--glow-color, rgba(233,69,96,0.3)); }
    }
    
    /* What-if cards */
    .whatif-row {
        display: flex;
        gap: 1rem;
        margin: 1rem 0;
    }
    .whatif-card {
        flex: 1;
        background: linear-gradient(135deg, #1A1A2E, #16213E);
        padding: 1.2rem;
        border-radius: 12px;
        text-align: center;
        border: 1px solid rgba(255,255,255,0.05);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .whatif-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 25px rgba(0,0,0,0.3);
    }
    .whatif-card .scenario-label {
        font-size: 0.8rem;
        color: #687494;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .whatif-card .scenario-prob {
        font-size: 2rem;
        font-weight: 700;
        margin: 0.5rem 0;
    }
    .whatif-card .scenario-delta {
        font-size: 0.85rem;
        font-weight: 600;
    }
    .gauge-circle .pct {
        font-size: 3.5rem;
        font-weight: 800;
        line-height: 1;
    }
    .gauge-circle .label {
        font-size: 0.85rem;
        color: #8892B0;
        margin-top: 4px;
        text-transform: uppercase;
        letter-spacing: 2px;
    }
    
    /* Strategy badge */
    .strategy-badge {
        text-align: center;
        margin: 1.5rem 0;
        padding: 1rem 1.5rem;
        border-radius: 12px;
        font-size: 1.15rem;
        font-weight: 600;
    }
    
    /* Stat cards */
    .stat-row {
        display: flex;
        gap: 1rem;
        margin: 1rem 0;
        flex-wrap: wrap;
    }
    .stat-card {
        flex: 1;
        min-width: 120px;
        background: linear-gradient(135deg, #1A1A2E, #16213E);
        padding: 1.2rem;
        border-radius: 12px;
        text-align: center;
        border: 1px solid rgba(255,255,255,0.05);
    }
    .stat-card .stat-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #CCD6F6;
    }
    .stat-card .stat-label {
        font-size: 0.75rem;
        color: #687494;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-top: 4px;
    }
    
    /* Insight box */
    .insight-box {
        background: linear-gradient(135deg, #1A1A2E, #0F3460);
        border-left: 4px solid #E94560;
        padding: 1.2rem 1.5rem;
        border-radius: 0 12px 12px 0;
        margin: 1rem 0;
        border: 1px solid rgba(233, 69, 96, 0.15);
    }
    .insight-box p {
        color: #A8B2D1;
        margin: 0;
        line-height: 1.6;
    }
    
    /* Input section */
    .input-header {
        background: linear-gradient(135deg, #1A1A2E, #16213E);
        padding: 1rem 1.5rem;
        border-radius: 12px;
        margin-bottom: 1rem;
        border: 1px solid rgba(255,255,255,0.05);
    }
    .input-header h3 {
        color: #CCD6F6;
        margin: 0;
        font-size: 1.1rem;
    }
    .input-header p {
        color: #687494;
        margin: 0.3rem 0 0 0;
        font-size: 0.85rem;
    }
    
    /* Sidebar */
    .sidebar-scenario {
        background: linear-gradient(135deg, #1A1A2E, #16213E);
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 0.8rem;
        border: 1px solid rgba(255,255,255,0.05);
        cursor: pointer;
    }
    .sidebar-scenario h4 {
        color: #CCD6F6;
        margin: 0 0 0.3rem 0;
        font-size: 0.95rem;
    }
    .sidebar-scenario p {
        color: #687494;
        margin: 0;
        font-size: 0.8rem;
    }
    
    /* Footer */
    .footer {
        text-align: center;
        color: #4A5568;
        font-size: 0.8rem;
        padding: 2rem 0 1rem 0;
        border-top: 1px solid rgba(255,255,255,0.05);
        margin-top: 2rem;
    }
    .footer a { color: #E94560; text-decoration: none; }
</style>
""", unsafe_allow_html=True)

# ── Load Model ──
@st.cache_resource
def load_model():
    return joblib.load(os.path.join(_DIR, 'ipl_chase_model.pkl'))

model = load_model()

MAX_BALLS = 120
RR_GAP_CLIP = 30
FEATURE_VECTOR_ORDER = (
    "cum_runs",
    "runs_needed",
    "current_rr",
    "wickets_in_hand",
    "balls_remaining",
    "rr_gap",
    "is_boundary",
    "is_dot",
    "phase",
    "last_6_runs",
)


def get_phase(over_num):
    if over_num <= 6:
        return 1, "Powerplay", "🛡️"
    if over_num <= 15:
        return 2, "Middle Overs", "⚖️"
    return 3, "Death Overs", "⚡"


def calculate_current_rr(cum_runs, balls_bowled):
    return cum_runs / (balls_bowled / 6) if balls_bowled > 0 else 0.0


def calculate_required_rr(runs_needed, balls_remaining):
    if runs_needed <= 0:
        return 0.0
    if balls_remaining <= 0:
        return np.inf
    return runs_needed / (balls_remaining / 6)


def calculate_rr_gap(runs_needed, balls_remaining, current_rr):
    required_rr = calculate_required_rr(runs_needed, balls_remaining)
    if np.isinf(required_rr):
        return float(RR_GAP_CLIP)
    return float(np.clip(required_rr - current_rr, -RR_GAP_CLIP, RR_GAP_CLIP))


def make_feature_vector(**feature_values):
    import pandas as pd
    ordered_values = [feature_values[name] for name in FEATURE_VECTOR_ORDER]
    feature_names = [
        "cum_runs", "runs_needed", "current_rr", "wickets_in_hand", 
        "balls_remaining", "rr_gap", "is_boundary", "is_dot", 
        "phase_num", "momentum"
    ]
    return pd.DataFrame([ordered_values], columns=feature_names)


def predict_probability(model_obj, **feature_values):
    runs_needed = feature_values["runs_needed"]
    balls_remaining = feature_values["balls_remaining"]
    wickets_in_hand = feature_values["wickets_in_hand"]

    if runs_needed <= 0:
        return 100.0
    if balls_remaining <= 0 or wickets_in_hand <= 0:
        return 0.0

    return model_obj.predict_proba(make_feature_vector(**feature_values))[0][1] * 100

# ── Sidebar: Preset Scenarios ──
with st.sidebar:
    st.markdown("## ☑ Test Scenarios")
    st.markdown("Click to load a famous IPL-style match situation:")
    
    scenario = st.radio(
        "Select a scenario",
        ["Custom Input", "🟢 Comfortable Chase", "🔴 Hopeless Collapse", 
         "🟡 Nail-biter", "🟠 Death Overs Thriller", "🔴 Powerplay Disaster"],
        label_visibility="collapsed"
    )
    
    # Preset values
    scenarios = {
        "Custom Input": {"target": 180, "runs": 85, "over": 10, "wkts": 3, "momentum": 8, "ball": 1},
        "🟢 Comfortable Chase": {"target": 150, "runs": 120, "over": 16, "wkts": 2, "momentum": 10, "ball": 1},
        "🔴 Hopeless Collapse": {"target": 220, "runs": 80, "over": 14, "wkts": 7, "momentum": 3, "ball": 0},
        "🟡 Nail-biter": {"target": 180, "runs": 90, "over": 12, "wkts": 3, "momentum": 7, "ball": 1},
        "🟠 Death Overs Thriller": {"target": 200, "runs": 155, "over": 18, "wkts": 5, "momentum": 14, "ball": 2},
        "🔴 Powerplay Disaster": {"target": 175, "runs": 15, "over": 4, "wkts": 4, "momentum": 2, "ball": 0},
    }
    
    preset = scenarios[scenario]
    
    st.markdown("---")
    st.markdown("### 📊 About")
    st.markdown("""
    **Model:** Tuned Random Forest  
    **Training:** 106K deliveries (2007–2022)  
    **Testing:** 24K deliveries (2023–2025)  
    **Accuracy:** 78.5%  
    **Precision:** 81.1%  
     
    📄 Phase 1 — Exploratory Analysis  
    📂 Phase 2 — Win Predictor App
    """)

# ── Header ──
st.markdown("""
<div class="main-header">
    <h1>🏏 IPL Chase Win Probability</h1>
    <p class="subtitle">Predict the chasing team's win probability from any match state</p>
    <span class="badge">Phase 2 —  IPL Analytics</span>
</div>
""", unsafe_allow_html=True)

# ── Inputs ──
st.markdown("""
<div class="input-header">
    <h3>⚙️ Match State Input</h3>
    <p>Set the current match situation to generate a prediction</p>
</div>
""", unsafe_allow_html=True)

ball_options = ["Dot Ball (0 runs)", "Single/Double", "Boundary (4 or 6)"]

col1, col2, col3 = st.columns(3)

with col1:
    target = st.number_input("🎯 Target Score", min_value=80, max_value=300, value=preset["target"],
                              help="Total set by batting-first team")
    runs_scored = st.number_input("🏃 Runs Scored", min_value=0, max_value=299, value=preset["runs"],
                                   help="Chasing team's current score")

with col2:
    current_over = st.slider("⚾ Current Over", min_value=1, max_value=20, value=preset["over"])
    wickets_lost = st.slider("☝🏻 Wickets Lost", min_value=0, max_value=9, value=preset["wkts"])

with col3:
    last_6_runs = st.slider("🔥 Last 6 Balls (Momentum)", min_value=0, max_value=36, value=preset["momentum"])
    delivery_type = st.selectbox("🏏 Current Delivery", ball_options, index=preset["ball"])

# ── Calculate Features ──
balls_bowled = current_over * 6
balls_remaining = MAX_BALLS - balls_bowled
runs_needed = target - runs_scored
current_rr = calculate_current_rr(runs_scored, balls_bowled)
required_rr = calculate_required_rr(runs_needed, balls_remaining)
rr_gap = calculate_rr_gap(runs_needed, balls_remaining, current_rr)
wickets_in_hand = 10 - wickets_lost

phase_num, phase_name, phase_emoji = get_phase(current_over)

delivery_is_boundary = 1 if delivery_type == "Boundary (4 or 6)" else 0
delivery_is_dot = 1 if delivery_type == "Dot Ball (0 runs)" else 0

features = make_feature_vector(
    cum_runs=runs_scored,
    runs_needed=runs_needed,
    current_rr=current_rr,
    wickets_in_hand=wickets_in_hand,
    balls_remaining=balls_remaining,
    rr_gap=rr_gap,
    is_boundary=delivery_is_boundary,
    is_dot=delivery_is_dot,
    phase=phase_num,
    last_6_runs=last_6_runs,
)

# ── Predict ──
st.markdown("---")

if balls_remaining <= 0:
    st.error("⚠️ The innings is over! No balls remaining.")
elif runs_needed <= 0:
    st.success("🎉 The chasing team has already won!")
else:
    win_prob = model.predict_proba(features)[0][1]
    win_pct = win_prob * 100

    # Determine color and strategy
    if win_pct >= 70:
        gauge_bg = "linear-gradient(135deg, #0a3d0a, #1a6b1a)"
        gauge_border = "#2ECC71"
        pct_color = "#2ECC71"
        strategy_text = "🟢 ATTACK — Strong position. Maintain aggression."
        strategy_bg = "rgba(46, 204, 113, 0.1)"
        strategy_border = "1px solid rgba(46, 204, 113, 0.3)"
    elif win_pct >= 55:
        gauge_bg = "linear-gradient(135deg, #3d3a0a, #6b5f1a)"
        gauge_border = "#F39C12"
        pct_color = "#F39C12"
        strategy_text = "🟡 BUILD — Steady accumulation. Save aggression for death overs."
        strategy_bg = "rgba(243, 156, 18, 0.1)"
        strategy_border = "1px solid rgba(243, 156, 18, 0.3)"
    elif win_pct >= 40:
        gauge_bg = "linear-gradient(135deg, #3d250a, #6b3f1a)"
        gauge_border = "#E67E22"
        pct_color = "#E67E22"
        strategy_text = "🟠 CAUTIOUS — Under pressure. Preserve wickets above all."
        strategy_bg = "rgba(230, 126, 34, 0.1)"
        strategy_border = "1px solid rgba(230, 126, 34, 0.3)"
    else:
        gauge_bg = "linear-gradient(135deg, #3d0a0a, #6b1a1a)"
        gauge_border = "#E74C3C"
        pct_color = "#E74C3C"
        if phase_num == 3:
            strategy_text = "🔴 ALL-OUT ATTACK — Nothing to lose. Swing for boundaries."
        else:
            strategy_text = "🔴 SURVIVE — Tough position. Keep wickets, hope for momentum shift."
        strategy_bg = "rgba(231, 76, 60, 0.1)"
        strategy_border = "1px solid rgba(231, 76, 60, 0.3)"

    # ── Results Layout ──
    res_col1, res_col2 = st.columns([1, 1.5])
    
    with res_col1:
        # Circular gauge
        st.markdown(f"""
        <div class="gauge-container">
            <div class="gauge-circle" style="background: {gauge_bg}; border: 3px solid {gauge_border};">
                <span class="pct" style="color: {pct_color};">{win_pct:.1f}%</span>
                <span class="label">Win Probability</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with res_col2:
        # Strategy recommendation
        st.markdown(f"""
        <div class="strategy-badge" style="background: {strategy_bg}; border: {strategy_border};">
            {strategy_text}
        </div>
        """, unsafe_allow_html=True)
        
        # Stat cards
        st.markdown(f"""
        <div class="stat-row">
            <div class="stat-card">
                <div class="stat-value">{runs_needed}</div>
                <div class="stat-label">Runs Needed</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{balls_remaining}</div>
                <div class="stat-label">Balls Left</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{wickets_in_hand}</div>
                <div class="stat-label">Wickets Left</div>
            </div>
        </div>
        <div class="stat-row">
            <div class="stat-card">
                <div class="stat-value">{current_rr:.1f}</div>
                <div class="stat-label">Current RR</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{required_rr:.1f}</div>
                <div class="stat-label">Required RR</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{phase_emoji} {phase_name}</div>
                <div class="stat-label">Phase</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # ── Phase Insight ──
    st.markdown("---")
    st.markdown("#### 🧠 Phase 1 → Phase 2 Insight")
    
    if phase_num == 1:
        insight = ("📋 <strong>Powerplay finding:</strong> Our model shows boundaries REDUCE win probability "
                   "by up to 10.9% in the powerplay. Focus on preserving wickets — "
                   "the aggression revolution works best in the death overs, not here.")
    elif phase_num == 2:
        insight = ("📋 <strong>Middle overs finding:</strong> Aggression is mostly neutral in this phase. "
                   "Build steadily and rotate strike. Save the big shots for overs 16-20.")
    else:
        if rr_gap > 6:
            insight = ("📋 <strong>Death overs finding:</strong> When far behind in the death overs, "
                       "boundaries boost win probability by +3.5%. This is the one phase "
                       "where aggression genuinely helps. Swing hard.")
        else:
            insight = ("📋 <strong>Death overs finding:</strong> In a controlled position during death overs, "
                       "aggression is neutral. No need to take unnecessary risks — "
                       "steady accumulation will close out the chase.")

    st.markdown(f"""
    <div class="insight-box">
        <p>{insight}</p>
    </div>
    """, unsafe_allow_html=True)

    # ── Pressure Meter ──
    st.markdown("---")
    st.markdown("#### 🌡️ Pressure Meter")
    
    pressure_pct = min(max((rr_gap + 5) / 20 * 100, 0), 100)
    if rr_gap < 0:
        pressure_label = "😎 Cruising — ahead of required rate"
        pressure_color = "#2ECC71"
    elif rr_gap < 3:
        pressure_label = "😊 Comfortable — on track"
        pressure_color = "#F39C12"
    elif rr_gap < 6:
        pressure_label = "😰 Building — falling behind"
        pressure_color = "#E67E22"
    else:
        pressure_label = "🔥 Critical — need quick runs NOW"
        pressure_color = "#E74C3C"
    
    st.markdown(f"""
    <div style="background: #1A1A2E; padding: 1.2rem; border-radius: 12px; border: 1px solid rgba(255,255,255,0.05);">
        <div style="display: flex; justify-content: space-between; margin-bottom: 0.5rem;">
            <span style="color: #8892B0; font-size: 0.9rem;">RR Gap: {rr_gap:+.1f}</span>
            <span style="color: {pressure_color}; font-size: 0.9rem; font-weight: 600;">{pressure_label}</span>
        </div>
        <div style="background: #0C0C1D; border-radius: 8px; height: 12px; overflow: hidden;">
            <div style="width: {pressure_pct}%; height: 100%; background: linear-gradient(90deg, #2ECC71, #F39C12, #E74C3C); border-radius: 8px; transition: width 0.5s ease;"></div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── What If Analysis ──
    st.markdown("---")
    st.markdown("#### 🔮 What If — Next Ball Impact")
    st.caption("How would the win probability change based on what happens next?")
    
    def predict_scenario(cum_runs, runs_needed_state, balls_bowled_state, wickets_in_hand_state,
                         balls_remaining_state, is_boundary_state, is_dot_state, phase_state, momentum_state):
        current_rr_state = calculate_current_rr(cum_runs, balls_bowled_state)
        rr_gap_state = calculate_rr_gap(runs_needed_state, balls_remaining_state, current_rr_state)
        return predict_probability(
            model,
            cum_runs=cum_runs,
            runs_needed=runs_needed_state,
            current_rr=current_rr_state,
            wickets_in_hand=wickets_in_hand_state,
            balls_remaining=balls_remaining_state,
            rr_gap=rr_gap_state,
            is_boundary=is_boundary_state,
            is_dot=is_dot_state,
            phase=phase_state,
            last_6_runs=momentum_state,
        )

    scenario_balls_bowled = min(balls_bowled + 1, MAX_BALLS)
    scenario_balls_remaining = max(balls_remaining - 1, 0)
    
    # Scenario: Dot ball
    dot_prob = predict_scenario(
        runs_scored, runs_needed, scenario_balls_bowled, wickets_in_hand,
        scenario_balls_remaining, 0, 1, phase_num, max(last_6_runs - 2, 0)
    )
    
    # Scenario: Boundary (4 runs)
    new_runs_b = runs_scored + 4
    new_need_b = runs_needed - 4
    boundary_prob = predict_scenario(
        new_runs_b, new_need_b, scenario_balls_bowled, wickets_in_hand,
        scenario_balls_remaining, 1, 0, phase_num, min(last_6_runs + 4, 36)
    )
    
    # Scenario: Wicket falls
    wicket_prob = predict_scenario(
        runs_scored, runs_needed, scenario_balls_bowled, max(wickets_in_hand - 1, 0),
        scenario_balls_remaining, 0, 1, phase_num, max(last_6_runs - 2, 0)
    )
    
    dot_delta = dot_prob - win_pct
    boundary_delta = boundary_prob - win_pct
    wicket_delta = wicket_prob - win_pct
    
    def delta_color(d):
        return "#2ECC71" if d >= 0 else "#E74C3C"
    
    st.markdown(f"""
    <div class="whatif-row">
        <div class="whatif-card">
            <div class="scenario-label">⚫ Dot Ball</div>
            <div class="scenario-prob" style="color: {delta_color(dot_delta)};">{dot_prob:.1f}%</div>
            <div class="scenario-delta" style="color: {delta_color(dot_delta)};">{dot_delta:+.1f}%</div>
        </div>
        <div class="whatif-card" style="border-color: rgba(46, 204, 113, 0.3);">
            <div class="scenario-label">💥 Boundary (4)</div>
            <div class="scenario-prob" style="color: {delta_color(boundary_delta)};">{boundary_prob:.1f}%</div>
            <div class="scenario-delta" style="color: {delta_color(boundary_delta)};">{boundary_delta:+.1f}%</div>
        </div>
        <div class="whatif-card" style="border-color: rgba(231, 76, 60, 0.3);">
            <div class="scenario-label">💀 Wicket Falls</div>
            <div class="scenario-prob" style="color: {delta_color(wicket_delta)};">{wicket_prob:.1f}%</div>
            <div class="scenario-delta" style="color: {delta_color(wicket_delta)};">{wicket_delta:+.1f}%</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Win Probability Projection Chart ──
    st.markdown("---")
    st.markdown("#### 📈 Win Probability Projection")
    st.caption("Projected win probability if the team continues at the current run rate")
    
    proj_overs = [current_over]
    proj_probs = [win_pct]
    
    for future_over in range(current_over + 1, 21):
        future_balls_bowled = future_over * 6
        future_balls_remaining = max(MAX_BALLS - future_balls_bowled, 0)
        if future_balls_remaining <= 0 and runs_needed > 0:
            break
        
        # Project runs scored at current rate
        extra_balls = max(future_balls_bowled - balls_bowled, 0)
        projected_runs = runs_scored + (current_rr * (extra_balls / 6))
        projected_runs_needed = target - projected_runs
        
        if projected_runs_needed <= 0:
            proj_overs.append(future_over)
            proj_probs.append(100.0)
            break
        
        proj_phase, _, _ = get_phase(future_over)
        prob = predict_scenario(
            projected_runs, projected_runs_needed, future_balls_bowled, wickets_in_hand,
            future_balls_remaining, 0, 0, proj_phase, last_6_runs
        )
        
        proj_overs.append(future_over)
        proj_probs.append(prob)
    
    if len(proj_overs) > 1:
        fig = go.Figure()
        
        # Main projection line
        fig.add_trace(go.Scatter(
            x=proj_overs, y=proj_probs,
            mode='lines+markers',
            name='Projected Win %',
            line=dict(color='#E94560', width=3, shape='spline'),
            marker=dict(size=10, color='#E94560', line=dict(color='white', width=2)),
            fill='tozeroy',
            fillcolor='rgba(233, 69, 96, 0.1)',
            hovertemplate='Over %{x}<br>Win Probability: <b>%{y:.1f}%</b><extra></extra>'
        ))
        
        # Current position marker
        fig.add_trace(go.Scatter(
            x=[current_over], y=[win_pct],
            mode='markers+text',
            name='Current',
            marker=dict(size=16, color='#F39C12', symbol='diamond',
                       line=dict(color='white', width=2)),
            text=['NOW'],
            textposition='top center',
            textfont=dict(color='#F39C12', size=12, family='Inter'),
            hovertemplate='<b>Current State</b><br>Over %{x}<br>Win: <b>%{y:.1f}%</b><extra></extra>'
        ))
        
        # 50% reference line
        fig.add_hline(y=50, line_dash='dot', line_color='#687494', opacity=0.3,
                      annotation_text='50%', annotation_position='right',
                      annotation_font_color='#687494')
        
        # Phase backgrounds
        fig.add_vrect(x0=0.5, x1=6.5, fillcolor='#2ECC71', opacity=0.03,
                      annotation_text='Powerplay', annotation_position='top left',
                      annotation_font_color='#687494', annotation_font_size=9)
        fig.add_vrect(x0=6.5, x1=15.5, fillcolor='#F39C12', opacity=0.03,
                      annotation_text='Middle', annotation_position='top left',
                      annotation_font_color='#687494', annotation_font_size=9)
        fig.add_vrect(x0=15.5, x1=20.5, fillcolor='#E74C3C', opacity=0.03,
                      annotation_text='Death', annotation_position='top left',
                      annotation_font_color='#687494', annotation_font_size=9)
        
        fig.update_layout(
            plot_bgcolor='#0E1117',
            paper_bgcolor='#0E1117',
            font=dict(color='#8892B0', family='Inter'),
            xaxis=dict(
                title='Over', color='#8892B0',
                gridcolor='rgba(104, 116, 148, 0.1)',
                range=[max(proj_overs[0] - 0.5, 0.5), 20.5],
                dtick=2
            ),
            yaxis=dict(
                title='Win Probability %', color='#8892B0',
                gridcolor='rgba(104, 116, 148, 0.1)',
                range=[0, 100]
            ),
            showlegend=False,
            height=350,
            margin=dict(l=50, r=20, t=20, b=50),
            hovermode='x unified'
        )
        
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
    
    # ── Plotly Gauge ──
    st.markdown("---")
    st.markdown("#### 🎯 Win Probability Gauge")
    
    gauge_fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=win_pct,
        number={'suffix': '%', 'font': {'size': 40, 'color': pct_color}},
        delta={'reference': 50, 'increasing': {'color': '#2ECC71'}, 'decreasing': {'color': '#E74C3C'}},
        gauge={
            'axis': {'range': [0, 100], 'tickcolor': '#687494', 'tickwidth': 1},
            'bar': {'color': pct_color, 'thickness': 0.3},
            'bgcolor': '#1A1A2E',
            'borderwidth': 0,
            'steps': [
                {'range': [0, 40], 'color': 'rgba(231, 76, 60, 0.15)'},
                {'range': [40, 55], 'color': 'rgba(230, 126, 34, 0.15)'},
                {'range': [55, 70], 'color': 'rgba(243, 156, 18, 0.15)'},
                {'range': [70, 100], 'color': 'rgba(46, 204, 113, 0.15)'}
            ],
            'threshold': {
                'line': {'color': '#F39C12', 'width': 3},
                'thickness': 0.8,
                'value': 50
            }
        }
    ))
    
    gauge_fig.update_layout(
        paper_bgcolor='#0E1117',
        font=dict(color='#8892B0', family='Inter'),
        height=250,
        margin=dict(l=30, r=30, t=30, b=10)
    )
    
    st.plotly_chart(gauge_fig, use_container_width=True, config={'displayModeBar': False})

# ── Footer ──
st.markdown("""
<div class="footer">
    <p>Model: Tuned Random Forest · Accuracy: 78.5% · Precision: 81.1% · Trained on IPL 2007–2022<br>
    Built by <strong>Mohammed Shafiulla</strong> · IPL Analytics Project</p>
</div>
""", unsafe_allow_html=True)
