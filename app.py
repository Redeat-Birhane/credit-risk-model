# app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go

# Page config
st.set_page_config(
    page_title="Fraud Detection Dashboard",
    page_icon="🔍",
    layout="wide"
)

# Load model
@st.cache_resource
def load_model():
    try:
        model = joblib.load('models/fraud_model.pkl')
        return model
    except:
        st.warning("⚠️ Model not found. Please run: python simple_train.py")
        return None

@st.cache_data
def load_data():
    try:
        df = pd.read_csv('data/processed/credit_data_processed.csv')
        return df
    except:
        st.warning("⚠️ Data not found. Please run data processing first.")
        return None

def main():
    st.title("🔍 Credit Risk & Fraud Detection Dashboard")
    st.markdown("---")
    
    # Load data and model
    df = load_data()
    model = load_model()
    
    if df is None:
        st.error("❌ No data loaded. Please run the data processing pipeline first.")
        st.code("python -m src.data_processing")
        return
    
    # Sidebar
    st.sidebar.header("📊 Dashboard Overview")
    
    # Data stats
    total_transactions = len(df)
    fraud_count = df['FraudResult'].sum()  # Changed from 'default' to 'FraudResult'
    fraud_rate = fraud_count / total_transactions
    
    st.sidebar.metric("Total Transactions", f"{total_transactions:,}")
    st.sidebar.metric("Fraud Cases", f"{fraud_count:,}")
    st.sidebar.metric("Fraud Rate", f"{fraud_rate:.2%}")
    
    # Main dashboard
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if model is not None:
            st.metric("Model Status", "✅ Loaded", delta="Ready")
        else:
            st.metric("Model Status", "❌ Not Loaded", delta="Run training")
    
    with col2:
        st.metric("Total Features", f"{len(df.columns) - 1}")  # Excluding target
    
    with col3:
        st.metric("Fraud Detection Rate", f"{fraud_rate:.2%}")
    
    st.markdown("---")
    
    # Two columns for charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("💰 Amount Distribution")
        fig, ax = plt.subplots(figsize=(8, 4))
        fraud = df[df['FraudResult'] == 1]['Amount']
        legit = df[df['FraudResult'] == 0]['Amount']
        
        ax.hist(legit, bins=50, alpha=0.5, label='Legit', color='green')
        ax.hist(fraud, bins=50, alpha=0.5, label='Fraud', color='red')
        ax.set_xlabel('Amount')
        ax.set_ylabel('Frequency')
        ax.legend()
        st.pyplot(fig)
    
    with col2:
        st.subheader("📊 Fraud vs Legit Transactions")
        fraud_counts = df['FraudResult'].value_counts()
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.pie(fraud_counts, labels=['Legit', 'Fraud'], autopct='%1.1f%%', 
               colors=['green', 'red'], startangle=90)
        ax.axis('equal')
        st.pyplot(fig)
    
    st.markdown("---")
    
    # Feature Importance (if model is loaded)
    if model is not None and hasattr(model, 'feature_importances_'):
        st.subheader("📈 Top Feature Importances")
        
        feature_cols = [col for col in df.columns if col != 'FraudResult']
        importance = model.feature_importances_
        
        # Create DataFrame
        importance_df = pd.DataFrame({
            'Feature': feature_cols,
            'Importance': importance
        }).sort_values('Importance', ascending=False)
        
        # Plot top 15
        fig, ax = plt.subplots(figsize=(10, 6))
        top_features = importance_df.head(15)
        ax.barh(top_features['Feature'], top_features['Importance'])
        ax.set_xlabel('Importance')
        ax.set_title('Top 15 Most Important Features')
        ax.invert_yaxis()
        st.pyplot(fig)
    
    # Prediction Section
    st.markdown("---")
    st.header("🎯 Test Fraud Prediction")
    
    st.write("Enter transaction details to get a fraud prediction:")
    
    col1, col2 = st.columns(2)
    
    with col1:
        amount = st.number_input("💰 Amount", min_value=0.0, value=100.0, step=10.0)
        value = st.number_input("💵 Value", min_value=0, value=50, step=10)
    
    with col2:
        # Provider selection
        provider_options = [2, 3, 4, 5, 6]
        provider = st.selectbox("🏢 Provider ID", options=provider_options)
        
        # Product category selection
        product_options = ['data_bundles', 'financial_services', 'movies', 'other', 
                          'ticket', 'transport', 'tv', 'utility_bill']
        product = st.selectbox("📦 Product Category", options=product_options)
    
    # Create input DataFrame
    if st.button("🔍 Predict Fraud", type="primary"):
        if model is None:
            st.error("❌ Model not loaded. Please train the model first.")
        else:
            # Create input data with all features
            input_data = {}
            
            # Numeric features
            input_data['Amount'] = amount
            input_data['Value'] = value
            
            # Provider one-hot encoding
            provider_cols = ['ProviderId_ProviderId_2', 'ProviderId_ProviderId_3', 
                           'ProviderId_ProviderId_4', 'ProviderId_ProviderId_5', 
                           'ProviderId_ProviderId_6']
            for i, col in enumerate(provider_cols):
                input_data[col] = 1 if provider == (i + 2) else 0
            
            # Product one-hot encoding (only set the selected one)
            product_cols = [col for col in df.columns if col.startswith('ProductId_')]
            for col in product_cols:
                input_data[col] = 0
            
            # Product category one-hot encoding
            product_cat_cols = [col for col in df.columns if col.startswith('ProductCategory_')]
            for col in product_cat_cols:
                category = col.replace('ProductCategory_', '')
                input_data[col] = 1 if category == product else 0
            
            # Channel (default to Channel 2)
            input_data['ChannelId_ChannelId_2'] = 1
            input_data['ChannelId_ChannelId_3'] = 0
            input_data['ChannelId_ChannelId_5'] = 0
            
            # Pricing strategy (default to 2)
            input_data['PricingStrategy_1'] = 0
            input_data['PricingStrategy_2'] = 1
            input_data['PricingStrategy_4'] = 0
            
            # Convert to DataFrame
            input_df = pd.DataFrame([input_data])
            
            # Add missing columns (set to 0)
            feature_cols = [col for col in df.columns if col != 'FraudResult']
            for col in feature_cols:
                if col not in input_df.columns:
                    input_df[col] = 0
            
            # Reorder columns to match training
            input_df = input_df[feature_cols]
            
            # Predict
            try:
                prediction = model.predict(input_df)[0]
                probability = model.predict_proba(input_df)[0, 1]
                
                st.subheader("📊 Prediction Result")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    if prediction == 1:
                        st.error(f"🚨 **FRAUD DETECTED!**")
                        st.metric("Fraud Probability", f"{probability:.2%}")
                    else:
                        st.success(f"✅ **Legitimate Transaction**")
                        st.metric("Fraud Probability", f"{probability:.2%}")
                
                with col2:
                    # Gauge chart for probability
                    fig = go.Figure(go.Indicator(
                        mode = "gauge+number+delta",
                        value = probability * 100,
                        title = {'text': "Fraud Risk Score"},
                        domain = {'x': [0, 1], 'y': [0, 1]},
                        gauge = {
                            'axis': {'range': [None, 100]},
                            'bar': {'color': "darkblue"},
                            'steps': [
                                {'range': [0, 30], 'color': "green"},
                                {'range': [30, 70], 'color': "yellow"},
                                {'range': [70, 100], 'color': "red"}
                            ],
                            'threshold': {
                                'line': {'color': "red", 'width': 4},
                                'thickness': 0.75,
                                'value': 50
                            }
                        }
                    ))
                    fig.update_layout(height=250)
                    st.plotly_chart(fig, use_container_width=True)
                
            except Exception as e:
                st.error(f"❌ Prediction error: {e}")
    
    st.markdown("---")
    st.caption("🔍 Fraud Detection Dashboard - Credit Risk Model")

if __name__ == "__main__":
    main()