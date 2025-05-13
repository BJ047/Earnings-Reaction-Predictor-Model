# Earnings-Reaction-Predictor-Model

**What it does**  
- Fetches 15 years of AAPL earnings & price data from Yahoo Finance  
- Computes EPS estimate vs. actual (surprise %) and next‑day returns  
- Summarizes beat/miss rates and up/down rates  
- Trains a Random Forest (5‑fold CV) to predict next‑day movement  
- Forecasts the next (unreported) earnings outcome and movement

**Installation**  
```bash
git clone https://github.com/<you>/earnings-reaction-aapl.git
cd earnings-reaction-aapl
python3 -m venv venv
source venv/bin/activate  # or .\\venv\\Scripts\\activate on Windows
pip install -r requirements.txt
