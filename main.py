import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
from datetime import datetime, timedelta

# Set page config
st.set_page_config(
    page_title="Market Crash Predictor",
    page_icon="📈",
    layout="wide"
)

# Load the models and scaler
@st.cache_resource
def load_models():
    try:
        rf_model = joblib.load('random_forest_model.pkl')
        xgb_model = joblib.load('xgboost_model.pkl')
        scaler = joblib.load('scaler.pkl')
        feature_names = joblib.load('feature_names.pkl')
        return rf_model, xgb_model, scaler, feature_names
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        return None, None, None, None

# Preprocess data
def preprocess_data(df, window_size=20):
    # Calculate returns
    returns = df.pct_change()
    
    # Create features DataFrame
    features = pd.DataFrame(index=df.index)
    
    for col in df.columns:
        # Lagged returns
        for i in range(1, 6):
            features[f'{col}_lag_{i}'] = returns[col].shift(i)
        
        # Rolling statistics with lag
        features[f'{col}_rolling_mean'] = returns[col].rolling(window=window_size).mean().shift(1)
        features[f'{col}_rolling_std'] = returns[col].rolling(window=window_size).std().shift(1)
        features[f'{col}_rolling_skew'] = returns[col].rolling(window=window_size).skew().shift(1)
        
        # Volatility features
        features[f'{col}_volatility'] = returns[col].rolling(window=window_size).std().shift(1)
    
    return features

# Make predictions
def predict_crashes(features, models, scaler):
    # Scale features
    features_scaled = scaler.transform(features)
    
    # Get predictions from both models
    rf_pred_proba = models[0].predict_proba(features_scaled)[:, 1]
    xgb_pred_proba = models[1].predict_proba(features_scaled)[:, 1]
    
    # Ensemble predictions (average)
    ensemble_proba = (rf_pred_proba + xgb_pred_proba) / 2
    
    return ensemble_proba

def main():
    st.title("📈 Market Crash Prediction System")
    
    # Load models
    rf_model, xgb_model, scaler, feature_names = load_models()
    
    if rf_model is None:
        st.error("Please ensure all model files are present in the directory.")
        return
    
    # File upload
    st.sidebar.header("Data Input")
    uploaded_file = st.sidebar.file_uploader("Upload market data (CSV)", type=['csv'])
    
    if uploaded_file is not None:
        try:
            # Load and preprocess data
            df = pd.read_csv(uploaded_file, header=1)
            st.success("Data loaded successfully!")
            
            # Show data preview
            st.subheader("Data Preview")
            st.dataframe(df.head())
            
            # Preprocess data
            features = preprocess_data(df)
            
            # Remove NaN values
            valid_idx = ~features.isna().any(axis=1)
            features = features[valid_idx]
            
            if len(features) > 0:
                # Make predictions
                crash_probabilities = predict_crashes(features, (rf_model, xgb_model), scaler)
                
                # Create results DataFrame
                results = pd.DataFrame({
                    'Date': features.index,
                    'Crash Probability': crash_probabilities
                })
                
                # Visualization
                st.subheader("Crash Probability Over Time")
                fig = go.Figure()
                
                # Add crash probability line
                fig.add_trace(go.Scatter(
                    x=results.index,
                    y=results['Crash Probability'],
                    name='Crash Probability',
                    line=dict(color='red', width=2)
                ))
                
                # Update layout
                fig.update_layout(
                    title='Market Crash Probability',
                    xaxis_title='Time',
                    yaxis_title='Probability',
                    hovermode='x unified'
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Risk levels
                st.subheader("Current Risk Assessment")
                latest_prob = crash_probabilities[-1]
                
                if latest_prob >= 0.7:
                    st.error(f"⚠️ High Risk (Probability: {latest_prob:.2%})")
                elif latest_prob >= 0.4:
                    st.warning(f"⚠️ Medium Risk (Probability: {latest_prob:.2%})")
                else:
                    st.success(f"✅ Low Risk (Probability: {latest_prob:.2%})")
                
                # Download predictions
                st.subheader("Download Predictions")
                csv = results.to_csv(index=False)
                st.download_button(
                    label="Download predictions as CSV",
                    data=csv,
                    file_name="crash_predictions.csv",
                    mime="text/csv"
                )
                
            else:
                st.error("Not enough data points after preprocessing.")
                
        except Exception as e:
            st.error(f"Error processing data: {str(e)}")
            
    else:
        st.info("Please upload a CSV file to begin analysis.")
        
    # Add information about the model
    st.sidebar.markdown("""
    ### About
    This model uses ensemble learning to predict market crashes based on:
    - Historical price data
    - Market volatility
    - Technical indicators
    
    The prediction combines results from:
    - Random Forest
    - XGBoost
    
    ### Instructions
    1. Upload your market data CSV
    2. Review the predictions
    3. Monitor risk levels
    """)

if __name__ == "__main__":
    main()