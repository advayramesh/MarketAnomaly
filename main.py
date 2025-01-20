import streamlit as st
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
            'Decision Tree': joblib.load('models/decision_tree_model.pkl'),
            'Random Forest': joblib.load('models/random_forest_model.pkl'),
            'XGBoost': joblib.load('models/xgboost_model.pkl')
        }
        scaler = joblib.load('models/scaler.pkl')
        feature_names = joblib.load('models/feature_names.pkl')
        return models, scaler, feature_names
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        return None, None, None

def process_uploaded_data(uploaded_file):
    """Process uploaded Bloomberg format data"""
    try:
        df = pd.read_csv(uploaded_file)
        
        # Extract ticker row
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
                numeric_df[col] = pd.to_numeric(series, errors='coerce')
            except:
                continue
        
        # Handle dates
        try:
            dates = pd.to_datetime(data_df['Date'], format='%m/%d/%Y', errors='coerce')
            numeric_df.index = dates
        except:
            numeric_df.index = pd.date_range(start='2000-01-01', periods=len(numeric_df), freq='B')
        
        numeric_df = numeric_df.fillna(method='ffill').fillna(method='bfill')
        return numeric_df
    
    except Exception as e:
        st.error(f"Error processing data: {str(e)}")
        return None

def create_market_visualization(df):
    """Create market overview visualization"""
    fig = go.Figure()
    
    for col in df.columns[:3]:
        fig.add_trace(go.Scatter(
            x=df.index,
            y=df[col],
            name=col,
            mode='lines'
        ))
    
    fig.update_layout(
        title='Market Performance',
        yaxis_title='Value',
        template='plotly_white',
        height=600
    )
    
    return fig

def main():
    st.set_page_config(page_title="Market Crash Prediction", layout="wide")
    
    st.title("📈 Market Crash Prediction System")
    
    # Sidebar
    st.sidebar.header("Upload Data")
    st.sidebar.markdown("""
    ### Required Format
    - Bloomberg-style CSV
    - Headers in row 6
    - Data starts from row 7
    - Market indices included
    """)
    
    uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file is not None:
        # Load models
        models, scaler, feature_names = load_models()
        
        # Process data
        df = process_uploaded_data(uploaded_file)
        
        if df is not None and not df.empty:
            # Create tabs
            tab1, tab2, tab3 = st.tabs([
                "Market Overview", 
                "Risk Analysis", 
                "Predictions"
            ])
            
            with tab1:
                st.subheader("Market Performance")
                fig = create_market_visualization(df)
                st.plotly_chart(fig, use_container_width=True)
                
                st.subheader("Market Statistics")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Total Trading Days", len(df))
                with col2:
                    volatility = df.pct_change().std().mean()
                    st.metric("Average Volatility", f"{volatility:.2%}")
                with col3:
                    returns = df.iloc[-1] / df.iloc[0] - 1
                    st.metric("Total Return", f"{returns.mean():.2%}")
            
            with tab2:
                st.subheader("Risk Metrics")
                
                # Calculate risk metrics
                returns = df.pct_change()
                rolling_vol = returns.rolling(20).std() * np.sqrt(252)
                
                fig = px.line(rolling_vol, title='Annualized Volatility')
                st.plotly_chart(fig, use_container_width=True)
                
                # Correlation matrix
                corr = df.corr()
                fig = px.imshow(corr, 
                              title='Asset Correlation Matrix',
                              color_continuous_scale='RdBu')
                st.plotly_chart(fig, use_container_width=True)
            
            with tab3:
                if models is not None:
                    st.subheader("Crash Predictions")
                    # Add prediction logic here
                    st.info("Model predictions will be added in the next update")
                else:
                    st.warning("Models not loaded. Please check model files.")
        
        else:
            st.error("Error processing the uploaded file. Please check the format.")
    
    else:
        st.info("Please upload a CSV file to begin analysis.")
        
        # Example data format
        st.markdown("""
        ### Sample Data Format
        Your CSV should look like this:
        ```
        Row 1-5: Metadata
        Row 6: Column Headers (Tickers)
        Row 7+: Data
        ```
        """)

if __name__ == "__main__":
    main()