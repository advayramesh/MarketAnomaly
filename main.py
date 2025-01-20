# Add this function to identify and explain assets
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
        
        # Fixed Income Terms
        '2Y': '2-Year Bond/Rate',
        '5Y': '5-Year Bond/Rate',
        '10Y': '10-Year Bond/Rate',
        '30Y': '30-Year Bond/Rate',
        
        # Volatility
        'VIX': 'CBOE Volatility Index - Measures market fear',
    }
    
    # Try to match the column name with explanations
    for key, explanation in explanations.items():
        if key.upper() in column_name.upper():
            return explanation
            
    return "No detailed explanation available for this asset"

def add_market_context(fig, df, selected_cols):
    """Add market context annotations to the plot"""
    # Find major market events
    returns = df[selected_cols].pct_change()
    major_moves = returns.abs() > 0.02  # 2% daily move threshold
    
    for col in selected_cols:
        major_dates = major_moves[major_moves[col]].index
        for date in major_dates:
            fig.add_annotation(
                x=date,
                y=df.loc[date, col],
                text="Major Move",
                showarrow=True,
                arrowhead=1
            )
    
    return fig

def create_market_insights(df):
    """Generate market insights from the data"""
    insights = []
    
    # Calculate metrics
    returns = df.pct_change()
    volatility = returns.std() * np.sqrt(252)  # Annualized volatility
    correlation = df.corr()
    
    # Overall market trend
    recent_trend = df.iloc[-20:].mean() > df.iloc[-40:-20].mean()
    trend_text = "upward" if recent_trend.mean() > 0.5 else "downward"
    insights.append(f"🎯 Recent Market Trend: The market has been trending {trend_text} over the last 20 days.")
    
    # Volatility insight
    high_vol_assets = volatility[volatility > volatility.mean()].index.tolist()
    if high_vol_assets:
        insights.append(f"📊 High Volatility Assets: {', '.join(high_vol_assets)} show above-average volatility.")
    
    # Correlation insight
    high_corr_pairs = []
    for i in range(len(correlation.columns)):
        for j in range(i+1, len(correlation.columns)):
            if correlation.iloc[i,j] > 0.8:
                high_corr_pairs.append(f"{correlation.columns[i]} & {correlation.columns[j]}")
    
    if high_corr_pairs:
        insights.append(f"🔗 Highly Correlated Pairs: {', '.join(high_corr_pairs)}")
    
    return insights

def main():
    st.set_page_config(page_title="Market Crash Prediction", layout="wide")
    
    st.title("📈 Market Crash Prediction System")
    
    # Sidebar with explanations
    st.sidebar.header("Upload Data")
    st.sidebar.markdown("""
    ### About This System
    This application analyzes financial market data to:
    - 📊 Monitor market performance
    - ⚠️ Detect potential market risks
    - 📈 Track technical indicators
    - 🔄 Analyze market correlations
    
    ### Required Data Format
    - Bloomberg-style CSV format
    - Headers in row 6 (Ticker names)
    - Data starts from row 7
    - Should include market indices
    """)
    
    uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file is not None:
        # Process data
        df = process_uploaded_data(uploaded_file)
        
        if df is not None and not df.empty:
            # Create tabs
            tab1, tab2, tab3, tab4 = st.tabs([
                "Market Overview", 
                "Asset Analysis",
                "Risk Analysis", 
                "Technical Indicators"
            ])
            
            with tab1:
                st.subheader("Market Performance")
                fig = create_market_visualization(df)
                if fig is not None:
                    st.plotly_chart(fig, use_container_width=True)
                
                # Market insights
                st.subheader("Market Insights")
                insights = create_market_insights(df)
                for insight in insights:
                    st.markdown(insight)
                
                # Market statistics
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
                st.subheader("Asset Analysis")
                
                # Display asset explanations
                st.markdown("### Asset Descriptions")
                for col in df.columns:
                    with st.expander(f"📌 {col}"):
                        explanation = get_asset_explanations(col)
                        st.write(explanation)
                        
                        # Add basic statistics
                        stats = df[col].describe()
                        st.markdown("#### Key Statistics")
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Average", f"{stats['mean']:.2f}")
                        with col2:
                            st.metric("Std Dev", f"{stats['std']:.2f}")
                        with col3:
                            st.metric("Latest", f"{df[col].iloc[-1]:.2f}")
            
            with tab3:
                st.subheader("Risk Analysis")
                
                # Rolling volatility
                returns = df.pct_change()
                rolling_vol = returns.rolling(20).std() * np.sqrt(252)
                
                fig = px.line(rolling_vol, title='Annualized Rolling Volatility')
                st.plotly_chart(fig, use_container_width=True)
                
                # Risk explanation
                st.markdown("""
                ### Understanding Risk Metrics
                - **Volatility**: Measures price fluctuations. Higher values indicate more uncertainty.
                - **Correlation**: Shows how assets move together. High correlation might indicate systemic risk.
                - **Drawdown**: Maximum loss from peak to trough. Shows worst-case scenarios.
                """)
                
                # Correlation matrix
                st.subheader("Asset Correlations")
                corr = df.corr()
                fig = px.imshow(corr, 
                              title='Asset Correlation Matrix',
                              color_continuous_scale='RdBu')
                st.plotly_chart(fig, use_container_width=True)
            
            with tab4:
                st.subheader("Technical Indicators")
                
                # Asset selection with explanation
                st.markdown("""
                ### Technical Analysis
                Technical analysis uses price and volume data to identify trading opportunities.
                Key indicators include:
                - Moving Averages (MA): Show trend direction
                - Relative Strength Index (RSI): Indicates overbought/oversold conditions
                - Bollinger Bands: Show volatility and potential price extremes
                """)
                
                # Select asset for technical analysis
                selected_asset = st.selectbox("Select Asset for Analysis", df.columns.tolist())
                
                if selected_asset:
                    # Calculate indicators
                    ma_20 = df[selected_asset].rolling(window=20).mean()
                    ma_50 = df[selected_asset].rolling(window=50).mean()
                    std_20 = df[selected_asset].rolling(window=20).std()
                    
                    # Create technical analysis plot
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=df.index, y=df[selected_asset], 
                                           name='Price', mode='lines'))
                    fig.add_trace(go.Scatter(x=df.index, y=ma_20, 
                                           name='20-day MA', line=dict(dash='dash')))
                    fig.add_trace(go.Scatter(x=df.index, y=ma_50, 
                                           name='50-day MA', line=dict(dash='dot')))
                    
                    # Add Bollinger Bands
                    fig.add_trace(go.Scatter(x=df.index, y=ma_20 + (std_20 * 2),
                                           name='Upper BB', line=dict(dash='dash')))
                    fig.add_trace(go.Scatter(x=df.index, y=ma_20 - (std_20 * 2),
                                           name='Lower BB', line=dict(dash='dash')))
                    
                    fig.update_layout(title=f'{selected_asset} Technical Analysis',
                                    height=500)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Technical Analysis Interpretation
                    st.markdown("### Technical Analysis Interpretation")
                    
                    # Trend Analysis
                    current_price = df[selected_asset].iloc[-1]
                    ma20_last = ma_20.iloc[-1]
                    ma50_last = ma_50.iloc[-1]
                    
                    trend = "Bullish" if current_price > ma20_last > ma50_last else "Bearish" if current_price < ma20_last < ma50_last else "Mixed"
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Trend", trend)
                    with col2:
                        st.metric("Price vs 20-day MA", f"{((current_price/ma20_last)-1)*100:.2f}%")
        
        else:
            st.error("Error processing the uploaded file. Please check the format.")
    
    else:
        st.info("Please upload a CSV file to begin analysis.")
        
        # Example data format
        st.markdown("""
        ### Understanding the Analysis
        This system provides:
        1. **Market Overview**: Overall market performance and trends
        2. **Asset Analysis**: Detailed analysis of individual assets
        3. **Risk Analysis**: Assessment of market risks and correlations
        4. **Technical Indicators**: Advanced technical analysis tools
        
        ### Sample Data Format
        Your CSV should follow Bloomberg format:
        ```
        Row 1-5: Metadata
        Row 6: Column Headers (Tickers)
        Row 7+: Daily price/value data
        ```
        """)

if __name__ == "__main__":
    main()