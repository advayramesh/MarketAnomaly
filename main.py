import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta

def clean_raw_data(df):
    """Clean the raw input data considering its specific structure"""
    try:
        # Get the actual column names from row 1 (0-based index)
        actual_columns = df.iloc[1].fillna('')
        
        # Create a mapping of unnamed columns to actual names
        col_mapping = {}
        for i, col in enumerate(df.columns):
            if col.startswith('Unnamed:'):
                if actual_columns[i] != '':
                    col_mapping[col] = actual_columns[i]
                else:
                    # Keep the last valid name for empty cells
                    last_valid = actual_columns[i-1] if i > 0 and actual_columns[i-1] != '' else f'Column_{i}'
                    col_mapping[col] = last_valid
            else:
                col_mapping[col] = col
        
        # Rename columns
        df = df.rename(columns=col_mapping)
        
        # Skip metadata rows (first 6 rows)
        df = df.iloc[6:].reset_index(drop=True)
        
        # Convert to numeric, handling errors
        numeric_df = pd.DataFrame()
        for col in df.columns:
            try:
                numeric_df[col] = pd.to_numeric(df[col], errors='coerce')
            except:
                st.warning(f"Dropping non-numeric column: {col}")
                continue
        
        # Forward fill missing values
        numeric_df = numeric_df.ffill()
        
        # Fill remaining NaN with 0
        numeric_df = numeric_df.fillna(0)
        
        return numeric_df
        
    except Exception as e:
        st.error(f"Error in clean_raw_data: {str(e)}")
        st.write("DataFrame head:", df.head())
        st.write("DataFrame info:", df.info())
        raise e

def generate_features(df, window_size=20):
    """Generate features for the cleaned data"""
    try:
        feature_df = pd.DataFrame()
        
        # Calculate returns and add features
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
        
        # Drop rows with any remaining NaN values
        feature_df = feature_df.dropna()
        
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
            st.write("First few rows:")
            st.dataframe(df.head())
            
            # Clean data
            cleaned_df = clean_raw_data(df)
            st.write("Cleaned data shape:", cleaned_df.shape)
            
            # Generate features
            features_df = generate_features(cleaned_df)
            st.write("Features data shape:", features_df.shape)
            
            if len(features_df) == 0:
                st.error("No valid data after preprocessing. Please check your input file.")
                return
            
            # Ensure we have all required features
            missing_features = set(feature_names) - set(features_df.columns)
            if missing_features:
                st.warning(f"Missing features: {missing_features}")
                # Add missing features with zeros
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
            
            # Create visualization
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
            
    else:
        st.info("Please upload a CSV file to begin analysis.")

if __name__ == "__main__":
    main()