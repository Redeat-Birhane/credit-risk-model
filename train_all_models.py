# train_all_models.py
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, roc_auc_score
import os

def train_all_models():
    """Train and compare multiple models."""
    
    print("="*60)
    print("🚀 Training Multiple Models for Comparison")
    print("="*60)
    
    # Load data
    df = pd.read_csv('data/processed/credit_data_processed.csv')
    X = df.drop(columns=['FraudResult'])
    y = df['FraudResult']
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    models = {
        'Logistic Regression': LogisticRegression(
            max_iter=1000, 
            class_weight='balanced', 
            random_state=42
        ),
        'Random Forest': RandomForestClassifier(
            n_estimators=100, 
            max_depth=10, 
            random_state=42, 
            class_weight='balanced'
        ),
        'XGBoost': XGBClassifier(
            n_estimators=100, 
            max_depth=6, 
            learning_rate=0.1, 
            random_state=42,
            scale_pos_weight=sum(y==0)/sum(y==1)  # Handle imbalance
        )
    }
    
    results = {}
    
    for name, model in models.items():
        print(f"\n{'='*50}")
        print(f"📊 Training: {name}")
        print('='*50)
        
        # Train
        model.fit(X_train, y_train)
        
        # Predict
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Evaluate
        auc = roc_auc_score(y_test, y_pred_proba)
        results[name] = auc
        
        print(f"   ROC AUC: {auc:.4f}")
        print(f"   Classification Report:")
        print(classification_report(y_test, y_pred))
        
        # Save model
        os.makedirs('models', exist_ok=True)
        model_path = f"models/{name.replace(' ', '_').lower()}.pkl"
        joblib.dump(model, model_path)
        print(f"   💾 Model saved to: {model_path}")
    
    # Compare results
    print("\n" + "="*60)
    print("📊 Model Comparison (ROC AUC)")
    print("="*60)
    for name, auc in sorted(results.items(), key=lambda x: x[1], reverse=True):
        print(f"   {name}: {auc:.4f}")
    
    # Find best model
    best_model = max(results, key=results.get)
    print(f"\n🏆 Best Model: {best_model} with AUC = {results[best_model]:.4f}")
    
    return results

if __name__ == "__main__":
    train_all_models()