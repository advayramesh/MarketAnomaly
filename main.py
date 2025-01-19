import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report
import joblib
import plotly.express as px
import plotly.graph_objects as go

# Model Training Component
def train_anomaly_detection_model(data):
    """
    Train an Isolation Forest model for anomaly detection
    """
    # Prepare features (you'll need to adjust these based on your actual data columns)
    features = data.select_dtypes(include=[np.number]).columns
    X = data[features]
    
    # Initialize and train the model
    model = IsolationForest(contamination=0.1, random_state=42)
    model.fit(X)
    
    # Save the model
    joblib.dump(model, 'market_crash_model.pkl')
    
    return model

def create_investment_strategy(data, predictions):
    """
    Generate investment signals based on model predictions
    """
    strategy = pd.DataFrame()
    strategy['Date'] = data.index
    strategy['Signal'] = predictions
    strategy['Position'] = np.where(strategy['Signal'] == -1, 'Sell', 'Buy')
    
    # Calculate returns (assuming you have a 'Close' price column)
    strategy['Returns'] = data['Close'].pct_change()
    strategy['Strategy_Returns'] = strategy['Returns'] * np.where(strategy['Position'] == 'Buy', 1, -1)
    
    return strategy

class InvestmentBot:
    def explain_strategy(self, prediction, confidence):
        """
        Generate natural language explanation of the investment strategy
        """
        if prediction == -1:
            explanation = f"""
            🔴 MARKET CRASH ALERT:
            Our AI model has detected potential market instability.
            
            Recommended Actions:
            1. Consider reducing market exposure
            2. Review stop-loss positions
            3. Look for defensive assets
            
            Confidence Level: {confidence:.2f}%
            """
        else:
            explanation = f"""
            🟢 MARKET STABLE:
            Current market conditions appear normal.
            
            Recommended Actions:
            1. Maintain regular investment strategy
            2. Consider dollar-cost averaging
            3. Review portfolio diversification
            
            Confidence Level: {confidence:.2f}%
            """
        return explanation

# Streamlit Interface
def main():
    st.title("Market Crash Prediction System")
    
    # Sidebar
    st.sidebar.header("Controls")
    upload_file = st.sidebar.file_uploader("Upload Market Data", type=['csv'])
    
    if upload_file is not None:
        data = pd.read_csv(upload_file)
        
        # Data Preview
        st.subheader("Data Preview")
        st.write(data.head())
        
        # Model Training Section
        st.subheader("Model Training")
        if st.button("Train Model"):
            with st.spinner("Training model..."):
                model = train_anomaly_detection_model(data)
                st.success("Model trained successfully!")
        
        # Load existing model
        try:
            model = joblib.load('market_crash_model.pkl')
            
            # Make predictions
            features = data.select_dtypes(include=[np.number]).columns
            predictions = model.predict(data[features])
            scores = model.score_samples(data[features])
            
            # Convert scores to confidence percentages
            confidence = (scores - scores.min()) / (scores.max() - scores.min()) * 100
            
            # Create investment strategy
            strategy = create_investment_strategy(data, predictions)
            
            # Visualization
            st.subheader("Market Analysis")
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=data.index, y=data['Close'], name='Price'))
            fig.add_trace(go.Scatter(x=data.index[predictions == -1],
                                   y=data['Close'][predictions == -1],
                                   mode='markers',
                                   name='Crash Detection',
                                   marker=dict(color='red', size=10)))
            st.plotly_chart(fig)
            
            # Strategy Performance
            st.subheader("Strategy Performance")
            cumulative_returns = (1 + strategy['Strategy_Returns']).cumprod()
            st.line_chart(cumulative_returns)
            
            # AI Bot Explanation
            st.subheader("AI Investment Bot")
            bot = InvestmentBot()
            latest_prediction = predictions[-1]
            latest_confidence = confidence[-1]
            
            st.info(bot.explain_strategy(latest_prediction, latest_confidence))
            
        except FileNotFoundError:
            st.warning("Please train the model first!")

if __name__ == "__main__":
    main()