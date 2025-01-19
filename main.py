import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta

def clean_raw_data(df):
    """Clean the raw input data and prepare it for feature generation"""
    try:
        # Print initial shape
        st.write("Initial data shape:", df.shape)
        st.write("Initial columns:", df.columns.tolist())
        
        # Drop unnamed columns
        df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
        st.write("Shape after dropping unnamed columns:", df.shape)
        
        # Convert all columns to numeric, dropping those that can't be converted
        numeric_df = pd.DataFrame()
        for col in df.columns:
            try:
                numeric_df[col] = pd.to_numeric(df[col], errors='coerce')
            except:
                st.warning(f"Dropping non-numeric column: {col}")
                continue
        
        # Drop columns with too many null values
        null_threshold = len(numeric_df) * 0.5  # 50% threshold
        numeric_df = numeric_df.dropna(axis=1, thresh=null_threshold)
        
        # Forward fill remaining NA values
        numeric_df = numeric_df.ffill()
        
        # Fill any remaining NaN values with 0
        numeric_df = numeric_df.fillna(0)
        
        st.write("Final shape after cleaning:", numeric_df.shape)
        return numeric_df
        
    except Exception as e:
        st.error(f"Error in clean_raw_data: {str(e)}")
        raise e

def generate_features(df, window_size=20):
    """Generate features matching the training data format"""
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
        
        # Forward fill any NaN values created by rolling calculations
        feature_df = feature_df.ffill().fillna(0)
        
        return feature_df
        
    except Exception as e:
        st.error(f"Error in generate_features: {str(e)}")
        raise e

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
    try:
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
        
    except Exception as e:
        st.error(f"Error in align_features: {str(e)}")
        raise e

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
            
            # Show raw data info
            st.subheader("Data Information")
            st.write("Raw data shape:", df.shape)
            st.write("Sample of raw data:")
            st.dataframe(df.head())
            
            # Clean raw data
            cleaned_df = clean_raw_data(df)
            
            # Show cleaned data info
            st.write("Cleaned data shape:", cleaned_df.shape)
            st.write("Sample of cleaned data:")
            st.dataframe(cleaned_df.head())
            
            # Generate features
            features_df = generate_features(cleaned_df)
            
            # Show generated features info
            st.write("Generated features shape:", features_df.shape)
            st.write("Sample of generated features:")
            st.dataframe(features_df.head())
            
            # Align features with expected format
            aligned_features = align_features(features_df, feature_names)
            
            # Show alignment info
            st.write("Aligned features shape:", aligned_features.shape)
            st.write("Sample of aligned features:")
            st.dataframe(aligned_features.head())
            
            # Scale features
            scaled_features = scaler.transform(aligned_features)
            
            # Get predictions
            predictions = []
            for name, model in models.items():
                pred = model.predict_proba(scaled_features)[:, 1]
                predictions.append(pred)
            
            predictions = np.array(predictions)
            mean_pred = np.mean(predictions, axis=0)
            std_pred = np.std(predictions, axis=0)
            
            # Visualization
            fig = go.Figure()
            
            # Add mean prediction
            fig.add_trace(go.Scatter(
                y=mean_pred,
                mode='lines',
                name='Crash Probability',
                line=dict(color='red', width=2)
            ))
            
            # Add confidence intervals
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
                'Timestamp': range(len(mean_pred)),
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
            st.write("Please check your data format and try again.")
            
    else:
        st.info("Please upload a CSV file to begin analysis.")
        st.write("The CSV should contain market data with numeric columns.")

if __name__ == "__main__":
    main()