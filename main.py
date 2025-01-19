import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta

# Page config
st.set_page_config(
    page_title="Advanced Market Crash Predictor",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .stMetric .metric-label { font-size: 16px !important; }
    .stAlert .alert-text { font-size: 18px !important; }
    </style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_models():
    models = {
        'Random Forest': joblib.load('random_forest_model.pkl'),
        'XGBoost': joblib.load('xgboost_model.pkl')
    }
    scaler = joblib.load('scaler.pkl')
    feature_names = joblib.load('feature_names.pkl')
    return models, scaler, feature_names

def calculate_uncertainty(predictions, threshold=0.5):
    """Calculate model uncertainty metrics"""
    pred_std = np.std(predictions, axis=0)
    pred_mean = np.mean(predictions, axis=0)
    uncertainty_score = pred_std / (pred_mean + 1e-10)  # Avoid division by zero
    
    confidence_level = 1 - uncertainty_score
    return confidence_level, uncertainty_score

def create_prediction_intervals(predictions, confidence=0.95):
    """Create prediction intervals"""
    lower = np.percentile(predictions, (1 - confidence) * 100 / 2, axis=0)
    upper = np.percentile(predictions, (1 + confidence) * 100 / 2, axis=0)
    mean = np.mean(predictions, axis=0)
    return lower, mean, upper

def main():
    st.title("📊 Advanced Market Crash Prediction System")
    
    # Load models and data
    try:
        models, scaler, feature_names = load_models()
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        return

    # Sidebar
    st.sidebar.header("Configuration")
    prediction_threshold = st.sidebar.slider(
        "Crash Probability Threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.05
    )
    
    confidence_level = st.sidebar.slider(
        "Confidence Level",
        min_value=0.8,
        max_value=0.99,
        value=0.95,
        step=0.01
    )

    # File upload
    uploaded_file = st.sidebar.file_uploader("Upload Market Data", type=['csv'])
    
    if uploaded_file is not None:
        try:
            # Load and preprocess data
            df = pd.read_csv(uploaded_file)
            
            # Main content area
            tab1, tab2, tab3 = st.tabs(["Predictions", "Model Analysis", "Risk Assessment"])
            
            with tab1:
                st.subheader("Market Crash Predictions")
                col1, col2 = st.columns([2, 1])
                
                # Get predictions from both models
                predictions = []
                for name, model in models.items():
                    pred = model.predict_proba(scaler.transform(df))[:, 1]
                    predictions.append(pred)
                
                predictions = np.array(predictions)
                lower, mean_pred, upper = create_prediction_intervals(
                    predictions, confidence_level
                )
                
                # Prediction plot
                with col1:
                    fig = go.Figure()
                    
                    # Add prediction interval
                    fig.add_trace(go.Scatter(
                        name='Upper Bound',
                        y=upper,
                        mode='lines',
                        line=dict(width=0),
                        showlegend=False
                    ))
                    fig.add_trace(go.Scatter(
                        name='Lower Bound',
                        y=lower,
                        mode='lines',
                        line=dict(width=0),
                        fillcolor='rgba(68, 68, 68, 0.3)',
                        fill='tonexty',
                        showlegend=False
                    ))
                    
                    # Add mean prediction
                    fig.add_trace(go.Scatter(
                        y=mean_pred,
                        mode='lines',
                        line=dict(color='red'),
                        name='Mean Prediction'
                    ))
                    
                    fig.update_layout(
                        title='Crash Probability Over Time with Uncertainty',
                        yaxis_title='Crash Probability',
                        hovermode='x unified'
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                # Current risk metrics
                with col2:
                    latest_pred = mean_pred[-1]
                    conf_level, uncert = calculate_uncertainty(predictions)
                    
                    # Risk gauge
                    fig = go.Figure(go.Indicator(
                        mode="gauge+number+delta",
                        value=latest_pred * 100,
                        domain={'x': [0, 1], 'y': [0, 1]},
                        delta={'reference': 50},
                        gauge={
                            'axis': {'range': [0, 100]},
                            'bar': {'color': "darkred"},
                            'steps': [
                                {'range': [0, 30], 'color': "lightgreen"},
                                {'range': [30, 70], 'color': "yellow"},
                                {'range': [70, 100], 'color': "red"}
                            ],
                            'threshold': {
                                'line': {'color': "black", 'width': 4},
                                'thickness': 0.75,
                                'value': prediction_threshold * 100
                            }
                        }
                    ))
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Metrics
                    st.metric("Model Confidence", f"{conf_level[-1]:.2%}")
                    st.metric("Uncertainty", f"{uncert[-1]:.2%}")
            
            with tab2:
                st.subheader("Model Analysis")
                
                # Feature importance
                if hasattr(models['Random Forest'], 'feature_importances_'):
                    importances = models['Random Forest'].feature_importances_
                    feat_imp = pd.DataFrame({
                        'Feature': feature_names,
                        'Importance': importances
                    }).sort_values('Importance', ascending=False)
                    
                    fig = px.bar(
                        feat_imp.head(15),
                        x='Importance',
                        y='Feature',
                        orientation='h',
                        title='Top 15 Most Important Features'
                    )
                    st.plotly_chart(fig, use_container_width=True)
            
            with tab3:
                st.subheader("Risk Assessment")
                
                # Risk metrics over time
                risk_df = pd.DataFrame({
                    'Date': range(len(mean_pred)),
                    'Crash Probability': mean_pred,
                    'Uncertainty': uncert,
                    'Model Confidence': conf_level
                })
                
                # Risk metrics plot
                fig = px.line(
                    risk_df,
                    x='Date',
                    y=['Crash Probability', 'Uncertainty', 'Model Confidence'],
                    title='Risk Metrics Over Time'
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Download predictions
                st.download_button(
                    label="Download Risk Analysis",
                    data=risk_df.to_csv(index=False),
                    file_name="risk_analysis.csv",
                    mime="text/csv"
                )
                
        except Exception as e:
            st.error(f"Error processing data: {str(e)}")
            st.write("Please ensure your data format matches the training data.")
    
    else:
        st.info("Please upload a CSV file to begin analysis.")
        
        # Show sample format
        st.markdown("""
        ### Expected Data Format:
        Your CSV should contain the following types of features:
        - Market indices (S&P, Nasdaq, etc.)
        - Volatility indicators
        - Interest rates
        - Currency rates
        """)

if __name__ == "__main__":
    main()