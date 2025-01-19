import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go

# Load the trained model
@st.cache_resource
def load_model():
    return joblib.load('market_crash_model.pkl')

def main():
    st.title("Market Crash Prediction System")
    
    # Load model
    try:
        model = load_model()
        st.success("Model loaded successfully!")
    except:
        st.error("Please ensure market_crash_model.pkl is in the same directory")
        return
    
    # File upload
    uploaded_file = st.file_uploader("Upload market data", type=['csv'])
    
    if uploaded_file is not None:
        # Load and preprocess data
        data = pd.read_csv(uploaded_file)
        
        # Convert columns to numeric where possible
        numeric_cols = []
        for col in data.columns:
            try:
                data[col] = pd.to_numeric(data[col])
                numeric_cols.append(col)
            except:
                continue
        
        # Use only numeric columns
        data_numeric = data[numeric_cols]
        
        # Make predictions
        predictions = model.predict(data_numeric)
        scores = model.score_samples(data_numeric)
        
        # Calculate confidence
        confidence = (scores - scores.min()) / (scores.max() - scores.min()) * 100
        
        # Display results
        st.subheader("Prediction Results")
        
        # Create DataFrame with results
        results = pd.DataFrame({
            'Prediction': ['Anomaly' if p == -1 else 'Normal' for p in predictions],
            'Confidence': confidence
        })
        
        st.write(results)
        
        # Visualization
        fig = go.Figure()
        
        # Assuming first numeric column as main value
        fig.add_trace(go.Scatter(
            x=data.index,
            y=data_numeric.iloc[:, 0],
            name='Value',
            line=dict(color='blue')
        ))
        
        # Add anomaly points
        anomaly_indices = np.where(predictions == -1)[0]
        fig.add_trace(go.Scatter(
            x=anomaly_indices,
            y=data_numeric.iloc[anomaly_indices, 0],
            mode='markers',
            name='Anomaly',
            marker=dict(color='red', size=10)
        ))
        
        st.plotly_chart(fig)

if __name__ == "__main__":
    main()