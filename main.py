import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta

def clean_raw_data(df):
    """Clean the raw input data and prepare it for feature generation"""
    # Drop unnamed columns
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    
    # Remove metadata rows (if any)
    df = df[~df.iloc[:, 0].str.contains('Show/Hide|Name|Index Currency|Ticker', na=False)]
    
    # Convert all columns to numeric
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Forward fill missing values
    df = df.ffill()
    
    return df

def generate_features(df, window_size=20):
    """Generate features matching the training data format"""
    feature_df = pd.DataFrame()
    
    # Calculate returns
    returns = df.pct_change()
    
    for col in df.columns:
        # Original value
        feature_df[col] = df[col]
        
        # Returns
        feature_df[f'{col}_return'] = returns[col]
        
        # Rolling statistics
        feature_df[f'{col}_rolling_mean'] = returns[col].rolling(window=window_size).mean()
        feature_df[f'{col}_rolling_std'] = returns[col].rolling(window=window_size).std()
        feature_df[f'{col}_rolling_skew'] = returns[col].rolling(window=window_size).skew()
    
    return feature_df

@st.cache_resource
def load_models():
    try:
        models = {
            'Random Forest': joblib.load('random_forest_model.pkl'),
            'XGBoost': joblib.load('xgboost_model.pkl')
        }
        scaler = joblib.load('scaler.pkl')
        feature_names = joblib.load('feature_names.pkl')
        return models, scaler, feature_names
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        return None, None, None

def align_features(df, expected_features):
    """Ensure the features match the expected format"""
    # Create DataFrame with expected features
    aligned_df = pd.DataFrame(columns=expected_features)
    
    # Copy over matching features
    for col in expected_features:
        if col in df.columns:
            aligned_df[col] = df[col]
        else:
            st.warning(f"Missing feature: {col}. Filling with zeros.")
            aligned_df[col] = 0
    
    return aligned_df

def main():
    st.title("📊 Market Crash Prediction System")
    
    # Load models and expected features
    models, scaler, feature_names = load_models()
    
    if models is None:
        st.error("Failed to load models. Please check model files.")
        return
    
    # File upload
    uploaded_file = st.sidebar.file_uploader("Upload Market Data", type=['csv'])
    
    if uploaded_file is not None:
        try:
            # Load raw data
            df = pd.read_csv(uploaded_file)
            
            # Show raw data preview
            st.subheader("Raw Data Preview")
            st.dataframe(df.head())
            
            # Clean raw data
            cleaned_df = clean_raw_data(df)
            
            # Generate features
            features_df = generate_features(cleaned_df)
            
            # Align features with expected format
            aligned_features = align_features(features_df, feature_names)
            
            # Show feature alignment status
            st.subheader("Feature Processing")
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("Expected Features:", len(feature_names))
                st.write("Generated Features:", len(aligned_features.columns))
            
            with col2:
                missing_features = set(feature_names) - set(aligned_features.columns)
                if missing_features:
                    st.warning(f"Missing features filled with zeros: {len(missing_features)}")
            
            # Scale features
            scaled_features = scaler.transform(aligned_features)
            
            # Get predictions from both models
            predictions = []
            for name, model in models.items():
                pred = model.predict_proba(scaled_features)[:, 1]
                predictions.append(pred)
            
            predictions = np.array(predictions)
            mean_pred = np.mean(predictions, axis=0)
            
            # Visualization
            st.subheader("Crash Probability Over Time")
            fig = go.Figure()
            
            # Add mean prediction line
            fig.add_trace(go.Scatter(
                y=mean_pred,
                mode='lines',
                name='Crash Probability',
                line=dict(color='red', width=2)
            ))
            
            # Add prediction bands
            std_pred = np.std(predictions, axis=0)
            fig.add_trace(go.Scatter(
                y=mean_pred + 2*std_pred,
                mode='lines',
                name='Upper Bound',
                line=dict(color='gray', width=1, dash='dash')
            ))
            
            fig.add_trace(go.Scatter(
                y=mean_pred - 2*std_pred,
                mode='lines',
                name='Lower Bound',
                line=dict(color='gray', width=1, dash='dash'),
                fill='tonexty'
            ))
            
            fig.update_layout(
                title='Market Crash Probability with Uncertainty Bands',
                yaxis_title='Probability',
                hovermode='x unified'
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Risk Assessment
            latest_prob = mean_pred[-1]
            latest_std = std_pred[-1]
            
            st.subheader("Current Risk Assessment")
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric(
                    "Crash Probability",
                    f"{latest_prob:.1%}",
                    f"±{latest_std:.1%}"
                )
            
            with col2:
                if latest_prob > 0.7:
                    st.error("High Risk Alert! 🚨")
                elif latest_prob > 0.3:
                    st.warning("Medium Risk Warning ⚠️")
                else:
                    st.success("Low Risk Status ✅")
            
            # Download predictions
            predictions_df = pd.DataFrame({
                'Date': df.index,
                'Crash_Probability': mean_pred,
                'Uncertainty': std_pred
            })
            
            st.download_button(
                label="Download Predictions",
                data=predictions_df.to_csv(index=False),
                file_name="crash_predictions.csv",
                mime="text/csv"
            )
            
        except Exception as e:
            st.error(f"Error processing data: {str(e)}")
            st.write("Detailed error information for debugging:")
            st.write(e)
            
    else:
        st.info("Please upload a CSV file to begin analysis.")
        st.write("Expected features:", feature_names)

if __name__ == "__main__":
    main()