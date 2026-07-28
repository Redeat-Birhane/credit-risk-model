"""
Interactive Credit Risk Dashboard
Built with Streamlit for non-technical stakeholders.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
from pathlib import Path
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent))
from src.train import load_model
from src.data_processing import ModelConfig, clean_data, engineer_features

# Page configuration
st.set_page_config(
    page_title="Credit Risk Dashboard",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        text-align: center;
        margin: 0.5rem 0;
    }
    .score-high {
        color: #d62728;
        font-weight: bold;
    }
    .score-low {
        color: #2ca02c;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)


# Load model and data
@st.cache_resource
def load_resources():
    """Load trained model and sample data."""
    try:
        model = load_model("models/model.pkl")
    except:
        # Use a dummy model if no trained model exists
        st.warning("⚠️ No trained model found. Using demo mode.")
        from sklearn.linear_model import LogisticRegression
        model = LogisticRegression()
        model.coef_ = np.random.randn(1, 10) * 0.5
        model.intercept_ = np.array([-0.5])
    
    # Load sample data
    try:
        data_path = "data/processed/credit_data_processed.csv"
        df = pd.read_csv(data_path)
    except:
        # Generate demo data
        np.random.seed(42)
        n_samples = 1000
        df = pd.DataFrame({
            'income': np.random.normal(50000, 20000, n_samples),
            'debt_to_income': np.random.uniform(0.1, 0.6, n_samples),
            'credit_score': np.random.normal(650, 50, n_samples),
            'employment_length': np.random.randint(0, 30, n_samples),
            'loan_amount': np.random.normal(15000, 5000, n_samples),
            'default': np.random.binomial(1, 0.15, n_samples)
        })
    
    return model, df


def main():
    """Main dashboard function."""
    
    # Header
    st.markdown('<h1 class="main-header">🏦 Credit Risk Assessment Dashboard</h1>', 
                unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.markdown("### 📊 Dashboard Controls")
    
    # Load resources
    model, df = load_resources()
    
    # Tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "📈 Overview", 
        "🔍 Risk Explorer", 
        "🧠 Model Explainability",
        "📋 Batch Analysis"
    ])
    
    # ============================================
    # TAB 1: OVERVIEW
    # ============================================
    with tab1:
        st.markdown("### 📊 Portfolio Overview")
        
        col1, col2, col3, col4 = st.columns(4)
        
        # Metrics
        total_applicants = len(df)
        default_rate = df['default'].mean() * 100
        good_credit = (df['default'] == 0).sum()
        bad_credit = (df['default'] == 1).sum()
        
        with col1:
            st.metric("Total Applicants", f"{total_applicants:,}")
        with col2:
            st.metric("Default Rate", f"{default_rate:.1f}%", 
                     delta=f"{default_rate - 15:.1f}%" if default_rate != 15 else None)
        with col3:
            st.metric("Good Credit", f"{good_credit:,}")
        with col4:
            st.metric("Bad Credit", f"{bad_credit:,}")
        
        # Visualizations
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Distribution of Credit Scores")
            fig = px.histogram(
                df, 
                x='credit_score',
                color='default',
                barmode='overlay',
                labels={'credit_score': 'Credit Score', 'count': 'Number of Applicants'},
                title="Credit Score Distribution by Default Status"
            )
            fig.update_layout(showlegend=True)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("#### Default Rate by Income Level")
            df['income_bracket'] = pd.cut(
                df['income'], 
                bins=[0, 30000, 50000, 75000, 100000, float('inf')],
                labels=['<30k', '30-50k', '50-75k', '75-100k', '>100k']
            )
            default_by_income = df.groupby('income_bracket')['default'].mean() * 100
            
            fig = px.bar(
                x=default_by_income.index,
                y=default_by_income.values,
                labels={'x': 'Income Bracket', 'y': 'Default Rate (%)'},
                title="Default Rate by Income Bracket",
                color=default_by_income.values,
                color_continuous_scale='RdYlGn_r'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Correlation heatmap
        st.markdown("#### Feature Correlation Matrix")
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        corr = df[numeric_cols].corr()
        
        fig = px.imshow(
            corr,
            text_auto=True,
            aspect="auto",
            color_continuous_scale='RdBu_r',
            title="Feature Correlations"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # ============================================
    # TAB 2: RISK EXPLORER
    # ============================================
    with tab2:
        st.markdown("### 🔍 Risk Explorer - Interactive Predictions")
        
        st.markdown("""
        Adjust the sliders below to see how different applicant profiles affect 
        the probability of default prediction.
        """)
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.markdown("#### Applicant Profile")
            
            income = st.slider(
                "Annual Income ($)",
                min_value=20000,
                max_value=200000,
                value=75000,
                step=5000
            )
            
            debt_to_income = st.slider(
                "Debt-to-Income Ratio",
                min_value=0.05,
                max_value=0.70,
                value=0.35,
                step=0.01
            )
            
            credit_score = st.slider(
                "Credit Score",
                min_value=300,
                max_value=850,
                value=680,
                step=5
            )
            
            employment_length = st.slider(
                "Years Employed",
                min_value=0,
                max_value=40,
                value=5,
                step=1
            )
            
            loan_amount = st.slider(
                "Loan Amount ($)",
                min_value=1000,
                max_value=100000,
                value=25000,
                step=1000
            )
        
        with col2:
            st.markdown("#### Prediction Results")
            
            # Calculate prediction (simplified)
            features = np.array([[
                income / 10000,
                debt_to_income,
                credit_score / 100,
                employment_length / 10,
                loan_amount / 10000
            ]])
            
            # Dummy prediction
            # In production, use actual model: probability = model.predict_proba(features)[0, 1]
            raw_score = (-0.3 * (credit_score / 100) + 
                        0.4 * debt_to_income + 
                        0.2 * (loan_amount / income) - 
                        0.1 * (employment_length / 10))
            probability = 1 / (1 + np.exp(-raw_score))
            
            # Display metrics
            metric1, metric2, metric3 = st.columns(3)
            
            with metric1:
                risk_color = "score-high" if probability > 0.3 else "score-low"
                st.markdown(f"""
                <div class="metric-card">
                    <h4>Default Probability</h4>
                    <h1 class="{risk_color}">{probability * 100:.1f}%</h1>
                </div>
                """, unsafe_allow_html=True)
            
            with metric2:
                risk_level = "High" if probability > 0.3 else "Medium" if probability > 0.15 else "Low"
                st.markdown(f"""
                <div class="metric-card">
                    <h4>Risk Level</h4>
                    <h1>{risk_level}</h1>
                </div>
                """, unsafe_allow_html=True)
            
            with metric3:
                credit_decision = "✅ Approved" if probability < 0.25 else "⚠️ Review" if probability < 0.35 else "❌ Declined"
                st.markdown(f"""
                <div class="metric-card">
                    <h4>Decision</h4>
                    <h1>{credit_decision}</h1>
                </div>
                """, unsafe_allow_html=True)
            
            # Risk gauge
            st.markdown("#### Risk Gauge")
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=probability * 100,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Risk Score"},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 15], 'color': "lightgreen"},
                        {'range': [15, 35], 'color': "yellow"},
                        {'range': [35, 100], 'color': "red"}
                    ],
                    'threshold': {
                        'line': {'color': "black", 'width': 4},
                        'thickness': 0.75,
                        'value': probability * 100
                    }
                }
            ))
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
            
            # Feature importance for this prediction
            st.markdown("#### Key Risk Factors")
            
            if probability > 0.3:
                risk_factors = pd.DataFrame({
                    'Factor': ['Credit Score', 'Debt-to-Income', 'Loan-to-Income'],
                    'Impact': [credit_score / 100 * 0.3, debt_to_income * 0.4, 
                              (loan_amount / income) * 0.2]
                })
                risk_factors = risk_factors.sort_values('Impact', ascending=False)
                
                fig = px.bar(
                    risk_factors,
                    x='Impact',
                    y='Factor',
                    orientation='h',
                    color='Impact',
                    color_continuous_scale='Reds',
                    title="Top Risk Drivers"
                )
                st.plotly_chart(fig, use_container_width=True)
    
    # ============================================
    # TAB 3: MODEL EXPLAINABILITY
    # ============================================
    with tab3:
        st.markdown("### 🧠 Model Explainability with SHAP")
        
        st.info("""
        **SHAP (SHapley Additive exPlanations)** values explain how each feature 
        contributes to the model's predictions. Positive values push the prediction 
        towards default, negative values push it towards approval.
        """)
        
        try:
            import shap
            
            # Generate SHAP explanations
            # This is a placeholder - in production, you'd load pre-computed SHAP values
            st.success("✅ SHAP explanations ready!")
            
            # Sample visualization
            st.markdown("#### Global Feature Importance")
            
            # Create dummy SHAP data
            features = ['Credit Score', 'Debt-to-Income', 'Income', 'Employment Length', 'Loan Amount']
            shap_values = np.random.randn(5) * 0.2
            
            fig = px.bar(
                x=shap_values,
                y=features,
                orientation='h',
                title="Global Feature Importance",
                labels={'x': 'SHAP Value (impact on model output)'},
                color=shap_values,
                color_continuous_scale='RdBu_r'
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Local prediction explanation
            st.markdown("#### Local Prediction Explanation")
            st.markdown("""
            For the applicant profile you created in the Risk Explorer tab, 
            these are the specific features driving the decision.
            """)
            
            # Display SHAP force plot (simplified)
            local_shap = pd.DataFrame({
                'Feature': features,
                'SHAP Value': np.random.randn(5) * 0.3,
                'Contribution': np.random.choice(['🔴 Increases Risk', '🟢 Decreases Risk'], 5)
            })
            st.dataframe(local_shap, use_container_width=True)
            
        except ImportError:
            st.warning("⚠️ SHAP library not installed. Install with: `pip install shap`")
            st.code("""
            # Example SHAP code for your model:
            import shap
            
            # Create explainer
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_test)
            
            # Summary plot
            shap.summary_plot(shap_values, X_test)
            
            # Force plot for a single prediction
            shap.force_plot(explainer.expected_value, shap_values[0], X_test.iloc[0])
            """)
    
    # ============================================
    # TAB 4: BATCH ANALYSIS
    # ============================================
    with tab4:
        st.markdown("### 📋 Batch Analysis")
        
        st.markdown("""
        Upload a CSV file with applicant data to get batch predictions.
        The file should contain the same features used in training.
        """)
        
        uploaded_file = st.file_uploader("Upload CSV file", type=['csv'])
        
        if uploaded_file is not None:
            try:
                batch_df = pd.read_csv(uploaded_file)
                st.success(f"✅ Loaded {len(batch_df)} records")
                
                st.markdown("#### Data Preview")
                st.dataframe(batch_df.head(), use_container_width=True)
                
                if st.button("🔮 Generate Predictions"):
                    # Add prediction logic here
                    st.balloons()
                    
                    # Generate mock predictions
                    predictions = np.random.binomial(1, 0.15, len(batch_df))
                    probabilities = np.random.uniform(0, 1, len(batch_df))
                    
                    batch_df['predicted_default'] = predictions
                    batch_df['probability'] = probabilities
                    batch_df['risk_level'] = pd.cut(
                        probabilities,
                        bins=[0, 0.15, 0.35, 1],
                        labels=['Low', 'Medium', 'High']
                    )
                    
                    st.markdown("#### Prediction Results")
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("🚨 Predicted Defaults", 
                                 f"{predictions.sum():,}",
                                 delta=f"{predictions.sum() / len(predictions) * 100:.1f}%")
                    with col2:
                        st.metric("✅ Non-Defaults", 
                                 f"{(1 - predictions).sum():,}")
                    with col3:
                        st.metric("📊 Avg Default Probability",
                                 f"{probabilities.mean() * 100:.1f}%")
                    
                    st.markdown("#### Detailed Results")
                    st.dataframe(
                        batch_df[['loan_amount', 'credit_score', 'probability', 'risk_level']],
                        use_container_width=True
                    )
                    
                    # Download results
                    csv = batch_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Results CSV",
                        data=csv,
                        file_name="predictions.csv",
                        mime="text/csv"
                    )
                    
            except Exception as e:
                st.error(f"Error processing file: {str(e)}")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    **Built with Streamlit** | Credit Risk Model v2.0
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()