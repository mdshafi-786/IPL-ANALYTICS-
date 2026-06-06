# 🏏 IPL Chase Win Probability Predictor

> **Phase 2 of IPL Analytics** — Predicting the chasing team's win probability from any live match state using a tuned Random Forest model.

---

## 📌 Project Overview

This project answers a critical question in IPL cricket:

> *"At any moment in a chase, how likely is the batting team to win — and what should they do next?"*

Built in two phases:
- **Phase 1** — Exploratory analysis of batting aggression trends across 18 IPL seasons
- **Phase 2** *(this repo)* — A machine learning model that predicts win probability ball-by-ball and delivers real-time strategic recommendations

---

## 🚀 Live App Features

| Feature | Description |
|---|---|
| 🎯 **Win Probability Gauge** | Real-time circular & Plotly gauge with colour-coded confidence |
| 🧠 **Strategic Recommendation** | Attack / Build / Cautious / Survive — based on match state |
| 🔮 **What-If Analysis** | See how a dot ball, boundary, or wicket shifts the probability |
| 📈 **Win Probability Projection** | Forward-projected win % if the team continues at current RR |
| 🌡️ **Pressure Meter** | RR gap visualised as a live pressure bar |
| ☑ **Preset Scenarios** | Load famous IPL-style situations (nail-biter, powerplay disaster, etc.) |
| 💡 **Phase Insights** | Findings from Phase 1 EDA surfaced in context |

---

## 🧠 Model Details

| Property | Value |
|---|---|
| Algorithm | Random Forest Classifier (tuned) |
| Training Data | IPL 2007–2022 · ~106,000 deliveries |
| Test Data | IPL 2023–2025 · ~24,000 deliveries |
| Accuracy | **78.5%** |
| Precision | **81.1%** |
| Target Variable | Did the chasing team win? (binary) |

### Features Used

```
cum_runs         — Runs scored so far in the innings
runs_needed      — Runs still required to win
current_rr       — Current run rate
wickets_in_hand  — Wickets remaining (10 - wickets lost)
balls_remaining  — Balls left in the innings
rr_gap           — Required RR minus Current RR (pressure indicator)
is_boundary      — Was the last ball a boundary? (1/0)
is_dot           — Was the last ball a dot? (1/0)
phase            — 1 = Powerplay, 2 = Middle Overs, 3 = Death Overs
last_6_runs      — Momentum: runs scored in last 6 balls
```

---

## 📁 Project Structure

```
IPL-ANALYTICS/
│
├── phase 2/
│   ├── app.py                    # Streamlit web application
│   ├── ipl_chase_model.pkl       # Trained Random Forest model
│   ├── Phase 2.ipynb             # Full model building notebook (EDA → Training → Eval)
│   ├── assets/                   # Model evaluation & EDA visualizations
│   ├── requirements.txt          # Python dependencies
│   └── README.md
└── README.md
```

---

## ⚙️ Setup & Installation

### 1. Clone the Repository
```bash
git clone https://github.com/mdshafi-786/IPL-ANALYTICS.git
cd IPL-ANALYTICS/"phase 2"
```

### 2. Create a Virtual Environment
```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the App
```bash
streamlit run app.py
```

The app will open at `http://localhost:8501`

---

## 📦 Requirements

```
streamlit
joblib
numpy
plotly
scikit-learn
pandas
matplotlib
seaborn
```

---

## 📊 Data

The model was trained on IPL ball-by-ball data from 2007–2025.

- **Source:** [Kaggle — IPL Complete Dataset](https://www.kaggle.com/datasets/patrickb1912/ipl-complete-dataset-20082020) *(or equivalent)*
- The raw CSV files are **not included** in this repo due to file size. Place them in the `data/` directory before running the notebook.

---

## 🔬 Key Findings (from Phase 1 + Phase 2)

- **Boundaries in the Powerplay** reduce win probability by up to **10.9%** — preserving wickets matters more than early aggression
- **Middle overs aggression** is mostly neutral — rotate strike and build steadily
- **Death overs (overs 16–20)** are the only phase where boundaries consistently boost win probability (**+3.5%** when chasing far behind)
- The **RR gap** is the single strongest predictor of match outcome

---

## 🛠 How to Use the App

1. **Select a preset scenario** from the sidebar or enter a custom match state
2. Input: Target score, runs scored, current over, wickets lost, last 6 balls (momentum), and current delivery type
3. Hit predict — the app computes all features and calls the model in real time
4. Read the **strategy badge**, inspect the **pressure meter**, and explore the **What-If** cards

---

## 👤 Author

**Mohammed Shafiulla**  
  
[GitHub](https://github.com/mdshafi-786)

---

## 📄 License

This project is for educational and portfolio purposes. IPL data belongs to its respective rights holders.
