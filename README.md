# 🏏 IPL Analytics — From EDA to Win Prediction

> A two-phase data science project that explores 18 seasons of IPL batting evolution and builds a real-time chase win-probability predictor.

---

## 📂 Project Structure

```
IPL-Analytics/
│
├── phase 1/                          # Exploratory Data Analysis
│   ├── IPL_Analysis.ipynb            # 12-finding analysis notebook
│   ├── assets/                       # Output visualizations (12 charts)
│   └── README.md
│
├── phase 2/                          # Machine Learning + Streamlit App
│   ├── Phase 2.ipynb                 # Model building notebook
│   ├── app.py                        # Streamlit web application
│   ├── ipl_chase_model.pkl           # Trained Random Forest model
│   ├── assets/                       # Model evaluation charts (9 charts)
│   ├── requirements.txt              # Python dependencies
│   └── README.md
│
└── README.md                         # ← You are here
```

---

## 🔬 Phase 1 — Exploratory Data Analysis

**Title:** *From Dots to Damage — How IPL Batsmen Eliminated Dead Balls*

Across 12 findings, this phase traces how IPL batting transformed from a dot-ball-heavy game to an era of relentless boundary hitting:

- Sixes nearly **doubled** per match; fours stayed flat
- Dot ball % dropped significantly, especially in the Powerplay
- Aggression Index (Boundary% + Six% − Dot%) has **steadily climbed**
- Dismissals per match stayed flat — aggression came **without a cost**
- Bowler economy rates rose from **7.2 → 9.2** runs per over

📖 [Full Phase 1 README →](phase%201/README.md)

---

## 🧠 Phase 2 — Chase Win Probability Predictor

**Title:** *IPL Chase Win Probability Predictor*

A tuned **Random Forest Classifier** that predicts the chasing team's win probability from any live match state, deployed as an interactive Streamlit app.

| Property | Value |
|---|---|
| Algorithm | Random Forest (tuned) |
| Training Data | IPL 2007–2022 · ~106K deliveries |
| Test Data | IPL 2023–2025 · ~24K deliveries |
| Accuracy | **78.5%** |
| Precision | **81.1%** |

**App Features:** Win Probability Gauge · Strategic Recommendations · What-If Analysis · Pressure Meter · Preset Scenarios

📖 [Full Phase 2 README →](phase%202/README.md)

---

## 📊 Data

Both phases use IPL ball-by-ball data from **2007–2025**:

- `deliveries_updated_ipl_upto_2025.csv` — Every ball bowled across all seasons
- `matches_updated_ipl_upto_2025.csv` — Match-level metadata

> ⚠️ CSV files are **not included** in this repository due to file size. Source: [Kaggle IPL Dataset](https://www.kaggle.com/datasets/patrickb1912/ipl-complete-dataset-20082020) *(or equivalent)*

---

## ⚙️ Quick Start

```bash
# Clone the repo
git clone https://github.com/mdshafi-786/IPL-ANALYTICS.git
cd IPL-ANALYTICS

# Phase 1 — Open the notebook
jupyter notebook "phase 1/IPL_Analysis.ipynb"

# Phase 2 — Run the Streamlit app
cd "phase 2"
pip install -r requirements.txt
streamlit run app.py
```

---

## 👤 Author

**Mohammed Shafiulla**  


[![GitHub](https://img.shields.io/badge/GitHub-Follow-black?style=flat&logo=github)](https://github.com/mdshafi-786)

---

## 📄 License

This project is for educational and portfolio purposes. IPL data belongs to its respective rights holders.
