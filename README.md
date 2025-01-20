# Market Crash Prediction System

A machine learning-based system for predicting market crashes using financial market data.

## Features

- Data preprocessing for Bloomberg format files
- Advanced feature engineering
- Multiple ML models (Decision Tree, Random Forest, XGBoost)
- Interactive visualizations
- Real-time market risk assessment

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/market-crash-prediction.git
cd market-crash-prediction
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Project Structure

```
market-crash-prediction/
│
├── app.py                 # Streamlit application
├── data.py               # Data processing and ML code
├── requirements.txt      # Project dependencies
├── README.md            # This file
│
├── models/              # Trained models
│   ├── decision_tree_model.pkl
│   ├── random_forest_model.pkl
│   ├── xgboost_model.pkl
│   ├── scaler.pkl
│   └── feature_names.pkl
│
└── data/               


2. Run the Streamlit application:
```bash
streamlit run app.py
```

3. Upload your market data CSV file in Bloomberg format:
   - Headers should be in row 6
   - Data should start from row 7
   - Include market indices for better predictions

## Data Format

The system expects Bloomberg-style CSV files with the following structure:
- Rows 1-5: Metadata
- Row 6: Column headers (Tickers)
- Row 7+: Actual data values

Example columns:
- Market indices (S&P, NASDAQ, etc.)
- Asset prices
- Economic indicators

## Model Training

The system uses three main models:
1. Decision Tree Classifier
2. Random Forest Classifier
3. XGBoost Classifier

Features include:
- Price returns
- Moving averages
- Volatility measures
- Technical indicators

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## Analysis Results

### Dataset Overview
- Size: 1,147 rows × 57 columns
- Time Period: 2000-01 to 2004-07
- Data Frequency: Daily

### Asset Analysis

#### Asset "284.250"
1. Price Movement:
- Strong upward trend from 500 (2000) to 2000 (2004)
- Notable acceleration during 2002-2003
- Consolidation periods in 2003-2004

2. Returns & Volatility:
- Returns typically between -10% and +10%
- Volatility clusters in:
  - Early 2000 (4.5% peak)
  - Mid-2001 to early 2002 (4% peak)
  - Early 2003 (3% level)
- Overall decreasing volatility trend
- ADF Test: Non-stationary (p-value: 0.8663)

#### Asset_4
1. Price Characteristics:
- Highly volatile oscillation between -1.0 and 1.0
- No clear long-term trend
- Sharp reversals and spikes

2. Returns & Volatility:
- Extreme volatility compared to Asset "284.250"
- Return spikes reaching 75%
- Major volatility event in 2003 (20x normal)
- Base volatility level: 1-2
- ADF Test: Strongly stationary (p-value: 1.048e-08)

### Trading Implications

1. Asset Characteristics:
- Asset "284.250": Suitable for trend-following
- Asset_4: Better for mean-reversion strategies
- Strong diversification potential due to different behaviors

2. Risk Management:
- Need for adaptive volatility models
- Evidence of fat-tailed distributions
- Volatility clustering suggests GARCH modeling
- Portfolio benefits from mixed characteristics

3. Market Context:
- Data reflects post-dot-com bubble period
- Shows both trending and mean-reverting assets
- Clear regime changes in volatility

### Model Performance Notes
- Successfully captures different asset behaviors
- Handles both trending and mean-reverting patterns
- Adapts to changing volatility regimes
- Provides early warning signals for extreme moves

