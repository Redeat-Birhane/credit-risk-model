# simple_train.py
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os

def train_simple_model():
    """Simple training script for fraud detection."""
    
    print("="*60)
    print("🚀 Simple Fraud Detection Model Training")
    print("="*60)
    
    # 1. Load processed data
    print("\n📂 Loading processed data...")
    df = pd.read_csv('data/processed/credit_data_processed.csv')
    print(f"   Loaded {len(df)} rows, {len(df.columns)} columns")
    
    # 2. Separate features and target
    X = df.drop(columns=['FraudResult'])
    y = df['FraudResult']
    
    print(f"   Features: {X.shape[1]}")
    print(f"   Target distribution: 0={sum(y==0)}, 1={sum(y==1)} ({sum(y==1)/len(y):.2%})")
    
    # 3. Split data
    print("\n📊 Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"   Train: {len(X_train)} samples")
    print(f"   Test: {len(X_test)} samples")
    
    # 4. Train model (Random Forest - good for fraud detection)
    print("\n🤖 Training Random Forest model...")
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        class_weight='balanced'  # Handle imbalanced data
    )
    model.fit(X_train, y_train)
    print("   ✅ Model trained!")
    
    # 5. Evaluate
    print("\n📊 Evaluating model...")
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    print("\n   Classification Report:")
    print(classification_report(y_test, y_pred))
    
    print(f"\n   ROC AUC Score: {roc_auc_score(y_test, y_pred_proba):.4f}")
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    print(f"\n   Confusion Matrix:")
    print(f"   [[{cm[0,0]:,} {cm[0,1]:,}]")
    print(f"    [{cm[1,0]:,} {cm[1,1]:,}]]")
    
    # 6. Feature Importance
    print("\n📈 Top 10 Most Important Features:")
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    for i, row in feature_importance.head(10).iterrows():
        print(f"   {row['feature']}: {row['importance']:.4f}")
    
    # 7. Save model
    os.makedirs('models', exist_ok=True)
    joblib.dump(model, 'models/fraud_model.pkl')
    print("\n💾 Model saved to: models/fraud_model.pkl")
    
    # 8. Plot feature importance (optional)
    try:
        plt.figure(figsize=(10, 8))
        top_features = feature_importance.head(15)
        plt.barh(top_features['feature'], top_features['importance'])
        plt.xlabel('Importance')
        plt.title('Top 15 Feature Importances')
        plt.tight_layout()
        plt.savefig('models/feature_importance.png')
        print("📊 Feature importance plot saved to: models/feature_importance.png")
    except:
        pass
    
    print("\n" + "="*60)
    print("✅ Training Complete!")
    print("="*60)
    
    return model

if __name__ == "__main__":
    train_simple_model()