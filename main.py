import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta

def clean_raw_data(df):
    """Clean the raw input data with proper handling of the specific format"""
    try:
        # Extract actual column names from row 5 (Ticker row)
        ticker_row = df.iloc[5]
        actual_columns = []
        
        for i, val in enumerate(ticker_row):
            if pd.notna(val) and val != '':
                actual_columns.append(val)
            else:
                actual_columns.append(f'Column_{i}')
        
        # Get the data starting from row 7 (actual values start here)
        data_df = df.iloc[7:].copy()
        data_df.columns = actual_columns
        
        # Convert numeric columns
        numeric_df = pd.DataFrame()
        for col in data_df.columns:
            try:
                # Replace '#N/A N/A' with NaN
                series = data_df[col].replace('#N/A N/A', np.nan)
                numeric_df[col] = pd.to_numeric(series, errors='coerce')
            except:
                st.warning(f"Dropping non-numeric column: {col}")
                continue
        
        # Drop columns with too many NaN values
        thresh = len(numeric_df) * 0.5  # 50% threshold
        numeric_df = numeric_df.dropna(axis=1, thresh=thresh)
        
        # Forward fill missing values
        numeric_df = numeric_df.ffill()
        
        # Fill remaining NaN with 0
        numeric_df = numeric_df.fillna(0)
        
        # Show retained columns
        st.write("Retained columns:", numeric_df.columns.tolist())
        
        return numeric_df
        
    except Exception as e:
        st.error(f"Error in clean_raw_data: {str(e)}")
        raise e

def generate_features(df, window_size=20):
    """Generate features for the cleaned data"""
    try:
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
        
        # Forward fill NaN values
        feature_df = feature_df.ffill()
        
        # Fill remaining NaN with 0
        feature_df = feature_df.fillna(0)
        
        # Drop first few rows where rolling calculations create NaN
        feature_df = feature_df.iloc[window_size:]
        
        return feature_df
        
    except Exception as e:
        st.error(f"Error in generate_features: {str(e)}")
        raise e

def main():
    st.title("📊 Market Crash Prediction System")
    
    # Load models
    try:
        rf_model = joblib.load('random_forest_model.pkl')
        xgb_model = joblib.load('xgboost_model.pkl')
        scaler = joblib.load('scaler.pkl')
        feature_names = joblib.load('feature_names.pkl')
        
        st.write("Expected features:", feature_names)
        
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        return
    
    # File upload
    uploaded_file = st.sidebar.file_uploader("Upload Market Data", type=['csv'])
    
    if uploaded_file is not None:
        try:
            # Load data
            df = pd.read_csv(uploaded_file)
            
            # Show initial data info
            st.subheader("Initial Data Information")
            st.write("Raw data shape:", df.shape)
            
            # Clean data
            cleaned_df = clean_raw_data(df)
            st.write("Cleaned data shape:", cleaned_df.shape)
            
            # Generate features
            features_df = generate_features(cleaned_df)
            st.write("Features data shape:", features_df.shape)
            
            if len(features_df) == 0:
                st.error("No valid data after preprocessing. Please check your input file.")
                return
            
            # Ensure all required features exist
            missing_features = set(feature_names) - set(features_df.columns)
            if missing_features:
                st.warning(f"Missing features will be filled with zeros: {missing_features}")
                for feature in missing_features:
                    features_df[feature] = 0
            
            # Reorder columns to match training data
            features_df = features_df.reindex(columns=feature_names, fill_value=0)
            
            # Scale features
            scaled_features = scaler.transform(features_df)
            
            # Get predictions
            rf_pred = rf_model.predict_proba(scaled_features)[:, 1]
            xgb_pred = xgb_model.predict_proba(scaled_features)[:, 1]
            
            # Combine predictions
            mean_pred = (rf_pred + xgb_pred) / 2
            std_pred = np.std([rf_pred, xgb_pred], axis=0)
            
            # Visualization
            fig = go.Figure()
            
            # Add mean prediction line
            fig.add_trace(go.Scatter(
                y=mean_pred,
                mode='lines',
                name='Crash Probability',
                line=dict(color='red', width=2)
            ))
            
            # Add confidence bands
            fig.add_trace(go.Scatter(
                y=mean_pred + 2*std_pred,
                mode='lines',
                line=dict(width=0),
                showlegend=False
            ))
            
            fig.add_trace(go.Scatter(
                y=mean_pred - 2*std_pred,
                mode='lines',
                line=dict(width=0),
                fill='tonexty',
                name='95% Confidence'
            ))
            
            fig.update_layout(
                title='Market Crash Probability Over Time',
                yaxis_title='Probability',
                hovermode='x unified'
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Risk Assessment
            latest_prob = mean_pred[-1]
            latest_std = std_pred[-1]
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Crash Probability", f"{latest_prob:.1%}")
                st.metric("Uncertainty", f"±{latest_std:.1%}")
            
            with col2:
                if latest_prob > 0.7:
                    st.error("🚨 High Risk Alert!")
                elif latest_prob > 0.3:
                    st.warning("⚠️ Medium Risk Warning")
                else:
                    st.success("✅ Low Risk Status")
            
            # Download predictions
            predictions_df = pd.DataFrame({
                'Date': range(len(mean_pred)),
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
            st.error("Error processing data")
            st.error(f"Detailed error: {str(e)}")
            st.write("DataFrame info:")
            st.write(df.info())
            
    else:
        st.info("Please upload a CSV file to begin analysis.")

if __name__ == "__main__":
    main()