import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import seaborn as sns
import matplotlib.pyplot as plt

def clean_raw_data(df):
    """
    Clean the raw financial market data with proper handling of multiple headers
    """
    try:
        # Get the ticker row (row 6)
        ticker_row = df.iloc[5]
        actual_columns = []
        
        # Extract meaningful column names
        for i, val in enumerate(ticker_row):
            if pd.notna(val) and val != '':
                actual_columns.append(val)
            else:
                actual_columns.append(f'Column_{i}')
        
        # Get the actual data starting from row 7
        data_df = df.iloc[7:].copy()
        data_df.columns = actual_columns
        
        # Convert numeric columns
        numeric_df = pd.DataFrame()
        for col in data_df.columns:
            try:
                series = data_df[col].replace('#N/A N/A', np.nan)
                numeric_df[col] = pd.to_numeric(series, errors='coerce')
            except:
                continue
        
        # Drop columns with too many NaN values
        thresh = len(numeric_df) * 0.5  # 50% threshold
        numeric_df = numeric_df.dropna(axis=1, thresh=thresh)
        
        # Forward fill missing values
        numeric_df = numeric_df.ffill()
        
        # Add date index
        numeric_df.index = pd.to_datetime(data_df['Date'], errors='coerce')
        
        return numeric_df
        
    except Exception as e:
        st.error(f"Error in clean_raw_data: {str(e)}")
        raise e

def calculate_market_indicators(df):
    """Calculate technical indicators for market analysis"""
    indicators = pd.DataFrame(index=df.index)
    
    # Calculate indicators for main market indices
    market_cols = ['S&P', 'Eurostoxx', 'Nasdaq'] if all(col in df.columns for col in ['S&P', 'Eurostoxx', 'Nasdaq']) else df.columns[:3]
    
    for col in market_cols:
        # Moving averages
        indicators[f'{col}_MA20'] = df[col].rolling(window=20).mean()
        indicators[f'{col}_MA50'] = df[col].rolling(window=50).mean()
        
        # Volatility
        indicators[f'{col}_VOL20'] = df[col].rolling(window=20).std()
        
        # Momentum
        indicators[f'{col}_MOM'] = df[col].pct_change(periods=20)
        
        # Relative strength
        indicators[f'{col}_RS'] = df[col] / df[col].rolling(window=20).mean()
    
    return indicators

def plot_market_dashboard(df, indicators):
    """Create a comprehensive market dashboard"""
    fig = go.Figure()
    
    # Plot main indices
    for col in df.columns[:3]:
        fig.add_trace(go.Scatter(
            x=df.index,
            y=df[col],
            name=col,
            mode='lines'
        ))
        
        # Add moving averages
        fig.add_trace(go.Scatter(
            x=df.index,
            y=indicators[f'{col}_MA20'],
            name=f'{col} MA20',
            line=dict(dash='dash'),
            visible='legendonly'
        ))
    
    fig.update_layout(
        title='Market Indices Performance',
        yaxis_title='Value',
        hovermode='x unified',
        height=600
    )
    
    return fig

def generate_risk_metrics(df, indicators):
    """Calculate risk metrics for the market"""
    recent_data = df.iloc[-20:]  # Last 20 days
    
    metrics = {
        'Volatility': recent_data.std().mean(),
        'Momentum': indicators.iloc[-1].filter(like='MOM').mean(),
        'Relative_Strength': indicators.iloc[-1].filter(like='RS').mean(),
    }
    
    return metrics

def get_market_sentiment(metrics):
    """Determine market sentiment based on metrics"""
    volatility_score = 1 if metrics['Volatility'] < 0.01 else (0 if metrics['Volatility'] > 0.03 else 0.5)
    momentum_score = 1 if metrics['Momentum'] > 0.02 else (0 if metrics['Momentum'] < -0.02 else 0.5)
    rs_score = 1 if metrics['Relative_Strength'] > 1.02 else (0 if metrics['Relative_Strength'] < 0.98 else 0.5)
    
    overall_score = (volatility_score + momentum_score + rs_score) / 3
    
    if overall_score > 0.7:
        return "🟢 Bullish", "green"
    elif overall_score < 0.3:
        return "🔴 Bearish", "red"
    else:
        return "🟡 Neutral", "yellow"

def main():
    st.set_page_config(layout="wide", page_title="Market Analysis Dashboard")
    
    st.title("📊 Financial Market Analysis Dashboard")
    
    st.sidebar.header("📝 Data Upload")
    st.sidebar.markdown("""
    ### Required Format:
    - CSV file with market data
    - Multiple header rows (standard Bloomberg format)
    - Key indicators:
        - Market Indices (S&P, Nasdaq, etc.)
        - Fixed Income
        - Commodities
        - Currencies
    """)
    
    uploaded_file = st.sidebar.file_uploader("Upload Market Data", type=['csv'])
    
    if uploaded_file is not None:
        try:
            # Load and process data
            raw_df = pd.read_csv(uploaded_file)
            df = clean_raw_data(raw_df)
            indicators = calculate_market_indicators(df)
            
            # Create tabs for different analyses
            tab1, tab2, tab3 = st.tabs([
                "📈 Market Overview",
                "📊 Technical Analysis",
                "🎯 Risk Assessment"
            ])
            
            with tab1:
                # Market dashboard
                st.plotly_chart(
                    plot_market_dashboard(df, indicators),
                    use_container_width=True
                )
                
                # Key metrics
                metrics = generate_risk_metrics(df, indicators)
                sentiment, color = get_market_sentiment(metrics)
                
                cols = st.columns(4)
                cols[0].metric("Market Sentiment", sentiment)
                cols[1].metric("Volatility", f"{metrics['Volatility']:.2%}")
                cols[2].metric("Momentum", f"{metrics['Momentum']:.2%}")
                cols[3].metric("Relative Strength", f"{metrics['Relative_Strength']:.2f}")
            
            with tab2:
                st.subheader("Technical Indicators")
                
                # Correlation heatmap
                st.write("#### Market Correlations")
                corr = df.corr()
                fig = px.imshow(
                    corr,
                    title="Asset Correlation Matrix",
                    color_continuous_scale="RdBu"
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Technical indicators plot
                st.write("#### Technical Indicators")
                tech_fig = go.Figure()
                
                for col in df.columns[:3]:
                    tech_fig.add_trace(go.Scatter(
                        x=df.index,
                        y=indicators[f'{col}_VOL20'],
                        name=f'{col} Volatility',
                        mode='lines'
                    ))
                
                tech_fig.update_layout(
                    title='20-Day Rolling Volatility',
                    yaxis_title='Volatility',
                    hovermode='x unified'
                )
                st.plotly_chart(tech_fig, use_container_width=True)
            
            with tab3:
                st.subheader("Risk Assessment")
                
                # Risk metrics
                risk_metrics = {
                    'Market Volatility': metrics['Volatility'],
                    'Trend Strength': abs(metrics['Momentum']),
                    'Market Divergence': abs(1 - metrics['Relative_Strength'])
                }
                
                # Create risk gauge
                risk_score = sum(risk_metrics.values()) / len(risk_metrics)
                
                fig = go.Figure(go.Indicator(
                    mode = "gauge+number",
                    value = risk_score * 100,
                    domain = {'x': [0, 1], 'y': [0, 1]},
                    title = {'text': "Market Risk Score"},
                    gauge = {
                        'axis': {'range': [0, 100]},
                        'bar': {'color': "darkblue"},
                        'steps': [
                            {'range': [0, 30], 'color': "lightgreen"},
                            {'range': [30, 70], 'color': "yellow"},
                            {'range': [70, 100], 'color': "red"}
                        ]
                    }
                ))
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Investment recommendations
                st.markdown("### Investment Recommendations")
                
                if risk_score < 0.3:
                    st.success("""
                    🟢 **Low Risk Environment**
                    - Consider increasing equity exposure
                    - Focus on growth sectors
                    - Reduce cash holdings
                    - Monitor for entry points in high-beta assets
                    """)
                elif risk_score < 0.7:
                    st.warning("""
                    🟡 **Moderate Risk Environment**
                    - Maintain balanced portfolio
                    - Keep strategic cash reserves
                    - Focus on quality stocks
                    - Consider protective options strategies
                    """)
                else:
                    st.error("""
                    🔴 **High Risk Environment**
                    - Reduce equity exposure
                    - Increase cash positions
                    - Focus on defensive sectors
                    - Consider hedging strategies
                    """)
            
            # Download processed data
            st.sidebar.download_button(
                label="Download Processed Data",
                data=df.to_csv(),
                file_name="processed_market_data.csv",
                mime="text/csv"
            )
            
        except Exception as e:
            st.error("Error processing data")
            st.error(f"Detailed error: {str(e)}")
            
    else:
        st.info("Please upload a CSV file to begin analysis.")
        st.markdown("""
        ### Sample Data Format:
        The dashboard expects a CSV file with:
        1. Multiple header rows (Bloomberg format)
        2. Market indices and indicators
        3. Daily price/value data
        
        Upload your data to see:
        - Market trends and correlations
        - Technical indicators
        - Risk assessment
        - Investment recommendations
        """)

if __name__ == "__main__":
    main()