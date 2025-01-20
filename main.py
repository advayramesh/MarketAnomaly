import streamlit as st
# Must be first Streamlit command
st.set_page_config(page_title="Market Crash Prediction", layout="wide")

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import joblib
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

def load_models():
    """Load trained models and artifacts"""
    try:
        models = {
            'Decision Tree': joblib.load('decision_tree_model.pkl'),
            'Random Forest': joblib.load('random_forest_model.pkl'),
            'XGBoost': joblib.load('xgboost_model.pkl')
        }
        scaler = joblib.load('scaler.pkl')
        feature_names = joblib.load('feature_names.pkl')
        return models, scaler, feature_names
    except Exception as e:
        st.warning(f"Some models could not be loaded. Running in analysis-only mode.")
        return None, None, None

def process_uploaded_data(uploaded_file):
    """Process uploaded Bloomberg format data"""
    try:
        df = pd.read_csv(uploaded_file)
        st.write("Data shape:", df.shape)
        
        # Extract ticker row (row 6)
        ticker_row = df.iloc[5]
        actual_columns = []
        
        for i, val in enumerate(ticker_row):
            if pd.notna(val) and str(val).strip() != '':
                actual_columns.append(str(val).strip())
            else:
                actual_columns.append(f'Asset_{i}')
        
        # Get data starting from row 7
        data_df = df.iloc[7:].copy()
        data_df.columns = actual_columns
        
        # Convert to numeric
        numeric_df = pd.DataFrame()
        for col in data_df.columns:
            try:
                series = data_df[col].replace(['#N/A N/A', 'N/A', ''], np.nan)
                numeric_series = pd.to_numeric(series, errors='coerce')
                if numeric_series.notna().sum() > len(numeric_series) * 0.5:  # Keep if >50% valid
                    numeric_df[col] = numeric_series
            except:
                continue
        
        # Handle dates
        try:
            dates = pd.date_range(start='2000-01-01', periods=len(numeric_df), freq='B')
            numeric_df.index = dates
        except Exception as e:
            st.error(f"Error setting dates: {str(e)}")
            numeric_df.index = pd.date_range(start='2000-01-01', periods=len(numeric_df), freq='B')
        
        numeric_df = numeric_df.fillna(method='ffill').fillna(method='bfill')
        
        st.write("Processed data shape:", numeric_df.shape)
        st.write("Available columns:", numeric_df.columns.tolist())
        
        return numeric_df
    
    except Exception as e:
        st.error(f"Error processing data: {str(e)}")
        return None

def get_asset_explanations(column_name):
    """Provide explanations for common financial assets and indicators"""
    explanations = {
        # Market Indices
        'S&P': 'S&P 500 - A stock market index tracking 500 large US companies',
        'SP': 'S&P 500 Index',
        'NASDAQ': 'NASDAQ Composite - Index of US technology stocks',
        'DAX': 'German Stock Index tracking 40 large German companies',
        'FTSE': 'UK Stock Index tracking 100 companies on London Stock Exchange',
        'EUROSTOXX': 'European Stock Index tracking 50 large Eurozone companies',
        
        # Fixed Income
        'BUND': 'German Government Bond',
        'TREASURY': 'US Treasury Bond',
        'UST': 'US Treasury Security',
        'JGB': 'Japanese Government Bond',
        
        # Commodities
        'GOLD': 'Gold Spot Price',
        'BRENT': 'Brent Crude Oil Price',
        'OIL': 'Crude Oil Price',
        'XAU': 'Gold Price in USD',
        
        # Currencies
        'EUR': 'Euro Currency',
        'USD': 'US Dollar',
        'GBP': 'British Pound',
        'JPY': 'Japanese Yen',
        
        # Volatility
        'VIX': 'CBOE Volatility Index - Measures market fear'
    }
    
    for key, explanation in explanations.items():
        if key.upper() in column_name.upper():
            return explanation
            
    return "Financial asset or indicator"

def create_market_visualization(df):
    """Create market overview visualization"""
    if df.empty:
        return None
    
    # Try to identify market indices or use first few columns
    market_cols = df.columns[:3]  # Use first 3 columns
    
    st.write("Plotting columns:", market_cols)
    
    fig = go.Figure()
    
    for col in market_cols:
        # Normalize the data
        series = df[col]
        normalized = (series - series.mean()) / series.std()
        
        fig.add_trace(go.Scatter(
            x=df.index,
            y=normalized,
            name=col,
            mode='lines'
        ))
    
    fig.update_layout(
        title='Market Performance (Normalized)',
        yaxis_title='Standard Deviations',
        template='plotly_white',
        height=600
    )
    
    return fig

def calculate_market_metrics(df):
    """Calculate key market metrics"""
    metrics = {}
    returns = df.pct_change()
    
    # Calculate metrics safely
    metrics['Daily Returns'] = returns.mean().mean() * 100
    metrics['Volatility'] = returns.std().mean() * 100
    
    # Calculate drawdown
    rolling_max = df.expanding().max()
    drawdown = (df - rolling_max) / rolling_max
    metrics['Max Drawdown'] = drawdown.min().min() * 100
    
    return metrics

def main():
    st.title("📈 Market Crash Prediction System")
    
    # Sidebar
    st.sidebar.header("Upload Data")
    st.sidebar.markdown("""
    ### Data Format Requirements
    - Bloomberg-style CSV
    - Headers in row 6
    - Data starts from row 7
    - Include market indices
    """)
    
    uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file is not None:
        models, scaler, feature_names = load_models()
        df = process_uploaded_data(uploaded_file)
        
        if df is not None and not df.empty:
            tab1, tab2, tab3 = st.tabs(["Market Overview", "Asset Analysis", "Risk Analysis"])
            
            with tab1:
                st.subheader("Market Performance")
                fig = create_market_visualization(df)
                if fig is not None:
                    st.plotly_chart(fig, use_container_width=True)
                
                st.subheader("Market Statistics")
                metrics = calculate_market_metrics(df)
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Average Daily Return", f"{metrics['Daily Returns']:.2f}%")
                with col2:
                    st.metric("Daily Volatility", f"{metrics['Volatility']:.2f}%")
                with col3:
                    st.metric("Maximum Drawdown", f"{metrics['Max Drawdown']:.2f}%")
            
            with tab2:
                st.subheader("Asset Information")
                for col in df.columns:
                    with st.expander(f"{col}"):
                        st.write(get_asset_explanations(col))
                        st.write("Recent value:", df[col].iloc[-1])
                        st.line_chart(df[col])
            
            with tab3:
                st.subheader("Risk Analysis")
                
                # Correlation heatmap
                corr = df.corr()
                fig = px.imshow(corr, 
                              title='Asset Correlation Matrix',
                              color_continuous_scale='RdBu')
                st.plotly_chart(fig, use_container_width=True)
                
                # Volatility analysis
                returns = df.pct_change()
                vol = returns.rolling(window=20).std() * np.sqrt(252)
                st.line_chart(vol)
                st.write("Higher values indicate greater market uncertainty")
        
        else:
            st.error("Please check your file format and try again.")
    
    else:
        st.info("Please upload a CSV file to begin analysis.")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")