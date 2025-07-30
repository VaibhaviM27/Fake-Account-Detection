#!/usr/bin/env python3
"""
Credit Card Fraud Detection Script
==================================

This script implements a complete fraud detection pipeline using multiple
machine learning algorithms including Logistic Regression, Decision Trees,
Random Forest, and XGBoost.

Usage:
    python fraud_detection_script.py

Requirements:
    Install dependencies: pip install -r requirements.txt
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score, 
    roc_curve, precision_recall_curve, f1_score
)
from imblearn.over_sampling import SMOTE
import xgboost as xgb
import joblib
import os

# Suppress warnings
warnings.filterwarnings('ignore')

class FraudDetectionPipeline:
    """Complete fraud detection pipeline"""
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.scaler = RobustScaler()
        self.models = {}
        self.results = {}
        self.best_model = None
        self.best_model_name = None
        
        # Set random seed
        np.random.seed(random_state)
        
    def generate_synthetic_data(self, n_samples=50000, fraud_rate=0.02):
        """Generate synthetic credit card transaction data"""
        print("Generating synthetic credit card transaction data...")
        
        np.random.seed(self.random_state)
        
        # Number of fraudulent transactions
        n_fraud = int(n_samples * fraud_rate)
        n_normal = n_samples - n_fraud
        
        # Generate normal transactions
        normal_data = {
            'Amount': np.random.lognormal(3, 1.5, n_normal),
            'Time': np.random.uniform(0, 86400, n_normal),  # 24 hours in seconds
            'V1': np.random.normal(0, 1, n_normal),
            'V2': np.random.normal(0, 1, n_normal),
            'V3': np.random.normal(0, 1, n_normal),
            'V4': np.random.normal(0, 1, n_normal),
            'V5': np.random.normal(0, 1, n_normal),
            'V6': np.random.normal(0, 1, n_normal),
            'V7': np.random.normal(0, 1, n_normal),
            'V8': np.random.normal(0, 1, n_normal),
            'V9': np.random.normal(0, 1, n_normal),
            'V10': np.random.normal(0, 1, n_normal),
            'Class': np.zeros(n_normal)
        }
        
        # Generate fraudulent transactions (different patterns)
        fraud_data = {
            'Amount': np.random.lognormal(2, 2, n_fraud),
            'Time': np.random.uniform(0, 86400, n_fraud),
            'V1': np.random.normal(2, 1.5, n_fraud),
            'V2': np.random.normal(-1, 1.2, n_fraud),
            'V3': np.random.normal(1.5, 1, n_fraud),
            'V4': np.random.normal(-0.5, 1.3, n_fraud),
            'V5': np.random.normal(0.8, 1.1, n_fraud),
            'V6': np.random.normal(-1.2, 1, n_fraud),
            'V7': np.random.normal(0.5, 1.4, n_fraud),
            'V8': np.random.normal(-0.8, 1, n_fraud),
            'V9': np.random.normal(1.2, 1.2, n_fraud),
            'V10': np.random.normal(-0.3, 1, n_fraud),
            'Class': np.ones(n_fraud)
        }
        
        # Combine data
        data = {}
        for key in normal_data.keys():
            data[key] = np.concatenate([normal_data[key], fraud_data[key]])
        
        # Create DataFrame
        df = pd.DataFrame(data)
        
        # Shuffle the data
        df = df.sample(frac=1, random_state=self.random_state).reset_index(drop=True)
        
        print(f"Dataset created with {len(df)} transactions")
        print(f"Fraud rate: {df['Class'].mean():.2%}")
        
        return df
    
    def load_data(self, filepath=None):
        """Load data from file or generate synthetic data"""
        if filepath and os.path.exists(filepath):
            print(f"Loading data from {filepath}...")
            self.df = pd.read_csv(filepath)
        else:
            self.df = self.generate_synthetic_data()
            # Save the generated data
            self.df.to_csv('credit_card_data.csv', index=False)
            print("Dataset saved as 'credit_card_data.csv'")
        
        return self.df
    
    def explore_data(self):
        """Perform exploratory data analysis"""
        print("\n" + "="*50)
        print("EXPLORATORY DATA ANALYSIS")
        print("="*50)
        
        print(f"Dataset Shape: {self.df.shape}")
        print(f"\nClass Distribution:")
        class_counts = self.df['Class'].value_counts()
        print(class_counts)
        print(f"Fraud percentage: {class_counts[1]/len(self.df)*100:.2f}%")
        
        print(f"\nMissing values:")
        print(self.df.isnull().sum().sum())
        
        print(f"\nBasic statistics:")
        print(self.df.describe())
        
        # Create visualizations
        self._create_eda_plots()
        
    def _create_eda_plots(self):
        """Create EDA visualizations"""
        plt.style.use('default')
        
        # Class distribution
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Bar plot
        class_counts = self.df['Class'].value_counts()
        axes[0,0].bar(['Normal', 'Fraud'], class_counts.values, color=['skyblue', 'salmon'])
        axes[0,0].set_title('Class Distribution')
        axes[0,0].set_ylabel('Count')
        
        # Amount distribution
        axes[0,1].hist(self.df[self.df['Class'] == 0]['Amount'], bins=50, alpha=0.7, 
                      label='Normal', color='skyblue')
        axes[0,1].hist(self.df[self.df['Class'] == 1]['Amount'], bins=50, alpha=0.7, 
                      label='Fraud', color='salmon')
        axes[0,1].set_title('Amount Distribution by Class')
        axes[0,1].set_xlabel('Amount')
        axes[0,1].set_ylabel('Frequency')
        axes[0,1].legend()
        axes[0,1].set_yscale('log')
        
        # Time distribution
        axes[1,0].hist(self.df[self.df['Class'] == 0]['Time'], bins=50, alpha=0.7, 
                      label='Normal', color='skyblue')
        axes[1,0].hist(self.df[self.df['Class'] == 1]['Time'], bins=50, alpha=0.7, 
                      label='Fraud', color='salmon')
        axes[1,0].set_title('Time Distribution by Class')
        axes[1,0].set_xlabel('Time (seconds)')
        axes[1,0].set_ylabel('Frequency')
        axes[1,0].legend()
        
        # Correlation heatmap
        v_features = [col for col in self.df.columns if col.startswith('V')][:5]  # First 5 V features
        corr_matrix = self.df[v_features + ['Class']].corr()
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, ax=axes[1,1])
        axes[1,1].set_title('Correlation Matrix (Sample Features)')
        
        plt.tight_layout()
        plt.savefig('eda_plots.png', dpi=300, bbox_inches='tight')
        plt.show()
        print("EDA plots saved as 'eda_plots.png'")
    
    def preprocess_data(self):
        """Preprocess the data"""
        print("\n" + "="*50)
        print("DATA PREPROCESSING")
        print("="*50)
        
        # Separate features and target
        X = self.df.drop('Class', axis=1)
        y = self.df['Class']
        
        print(f"Features shape: {X.shape}")
        print(f"Target shape: {y.shape}")
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=self.random_state, stratify=y
        )
        
        print(f"Training set shape: {X_train.shape}")
        print(f"Test set shape: {X_test.shape}")
        
        # Scale the features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Convert back to DataFrame for easier handling
        X_train_scaled = pd.DataFrame(X_train_scaled, columns=X.columns)
        X_test_scaled = pd.DataFrame(X_test_scaled, columns=X.columns)
        
        self.X_train, self.X_test = X_train_scaled, X_test_scaled
        self.y_train, self.y_test = y_train, y_test
        
        print("Data preprocessing completed!")
        
    def train_models(self):
        """Train multiple ML models"""
        print("\n" + "="*50)
        print("MODEL TRAINING")
        print("="*50)
        
        # Initialize models
        models = {
            'Logistic Regression': LogisticRegression(random_state=self.random_state, max_iter=1000),
            'Decision Tree': DecisionTreeClassifier(random_state=self.random_state),
            'Random Forest': RandomForestClassifier(random_state=self.random_state, n_estimators=100),
            'XGBoost': xgb.XGBClassifier(random_state=self.random_state, eval_metric='logloss')
        }
        
        # Train and evaluate each model
        for name, model in models.items():
            print(f"\nTraining {name}...")
            
            # Train the model
            model.fit(self.X_train, self.y_train)
            
            # Make predictions
            y_pred = model.predict(self.X_test)
            y_pred_proba = model.predict_proba(self.X_test)[:, 1]
            
            # Calculate metrics
            roc_auc = roc_auc_score(self.y_test, y_pred_proba)
            f1 = f1_score(self.y_test, y_pred)
            
            # Store results
            self.results[name] = {
                'model': model,
                'predictions': y_pred,
                'probabilities': y_pred_proba,
                'roc_auc': roc_auc,
                'f1_score': f1,
                'classification_report': classification_report(self.y_test, y_pred)
            }
            
            print(f"ROC-AUC: {roc_auc:.4f}")
            print(f"F1-Score: {f1:.4f}")
        
        # Find best model
        self.best_model_name = max(self.results.keys(), key=lambda x: self.results[x]['roc_auc'])
        self.best_model = self.results[self.best_model_name]['model']
        
        print(f"\nBest model (by ROC-AUC): {self.best_model_name}")
        
    def train_with_smote(self):
        """Train models with SMOTE to handle class imbalance"""
        print("\n" + "="*50)
        print("TRAINING WITH SMOTE")
        print("="*50)
        
        # Apply SMOTE
        smote = SMOTE(random_state=self.random_state)
        X_train_smote, y_train_smote = smote.fit_resample(self.X_train, self.y_train)
        
        print(f"Original training set shape: {self.X_train.shape}")
        print(f"SMOTE training set shape: {X_train_smote.shape}")
        
        # Train selected models with SMOTE
        smote_models = {
            'Logistic Regression (SMOTE)': LogisticRegression(random_state=self.random_state, max_iter=1000),
            'Random Forest (SMOTE)': RandomForestClassifier(random_state=self.random_state, n_estimators=100)
        }
        
        for name, model in smote_models.items():
            print(f"\nTraining {name}...")
            
            # Train on SMOTE data
            model.fit(X_train_smote, y_train_smote)
            
            # Predict on original test set
            y_pred = model.predict(self.X_test)
            y_pred_proba = model.predict_proba(self.X_test)[:, 1]
            
            # Calculate metrics
            roc_auc = roc_auc_score(self.y_test, y_pred_proba)
            f1 = f1_score(self.y_test, y_pred)
            
            # Store results
            self.results[name] = {
                'model': model,
                'predictions': y_pred,
                'probabilities': y_pred_proba,
                'roc_auc': roc_auc,
                'f1_score': f1,
                'classification_report': classification_report(self.y_test, y_pred)
            }
            
            print(f"ROC-AUC: {roc_auc:.4f}")
            print(f"F1-Score: {f1:.4f}")
        
        # Update best model if SMOTE version is better
        current_best_score = self.results[self.best_model_name]['roc_auc']
        for name, result in self.results.items():
            if result['roc_auc'] > current_best_score:
                self.best_model_name = name
                self.best_model = result['model']
                current_best_score = result['roc_auc']
    
    def evaluate_models(self):
        """Evaluate and compare all models"""
        print("\n" + "="*50)
        print("MODEL EVALUATION")
        print("="*50)
        
        # Create comparison DataFrame
        comparison_data = []
        for model_name, result in self.results.items():
            comparison_data.append({
                'Model': model_name,
                'ROC-AUC': result['roc_auc'],
                'F1-Score': result['f1_score']
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        comparison_df = comparison_df.sort_values('ROC-AUC', ascending=False)
        
        print("Model Comparison:")
        print(comparison_df.round(4))
        
        # Create evaluation plots
        self._create_evaluation_plots()
        
        # Print detailed reports
        print(f"\nDetailed Classification Reports:")
        print("-" * 60)
        for model_name, result in self.results.items():
            print(f"\n{model_name}:")
            print("-" * 30)
            print(result['classification_report'])
    
    def _create_evaluation_plots(self):
        """Create evaluation visualizations"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Model comparison bar plot
        model_names = list(self.results.keys())
        roc_scores = [self.results[name]['roc_auc'] for name in model_names]
        f1_scores = [self.results[name]['f1_score'] for name in model_names]
        
        x = np.arange(len(model_names))
        width = 0.35
        
        axes[0,0].bar(x - width/2, roc_scores, width, label='ROC-AUC', color='skyblue')
        axes[0,0].bar(x + width/2, f1_scores, width, label='F1-Score', color='salmon')
        axes[0,0].set_xlabel('Models')
        axes[0,0].set_ylabel('Score')
        axes[0,0].set_title('Model Performance Comparison')
        axes[0,0].set_xticks(x)
        axes[0,0].set_xticklabels(model_names, rotation=45, ha='right')
        axes[0,0].legend()
        axes[0,0].set_ylim(0, 1)
        
        # ROC Curves
        for model_name, result in self.results.items():
            fpr, tpr, _ = roc_curve(self.y_test, result['probabilities'])
            axes[0,1].plot(fpr, tpr, label=f"{model_name} (AUC = {result['roc_auc']:.3f})")
        
        axes[0,1].plot([0, 1], [0, 1], 'k--', label='Random Classifier')
        axes[0,1].set_xlabel('False Positive Rate')
        axes[0,1].set_ylabel('True Positive Rate')
        axes[0,1].set_title('ROC Curves Comparison')
        axes[0,1].legend()
        axes[0,1].grid(True, alpha=0.3)
        
        # Confusion Matrix for best model
        cm = confusion_matrix(self.y_test, self.results[self.best_model_name]['predictions'])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[1,0])
        axes[1,0].set_title(f'Confusion Matrix - {self.best_model_name}')
        axes[1,0].set_xlabel('Predicted')
        axes[1,0].set_ylabel('Actual')
        
        # Precision-Recall Curves
        for model_name, result in self.results.items():
            precision, recall, _ = precision_recall_curve(self.y_test, result['probabilities'])
            axes[1,1].plot(recall, precision, label=f"{model_name}")
        
        axes[1,1].set_xlabel('Recall')
        axes[1,1].set_ylabel('Precision')
        axes[1,1].set_title('Precision-Recall Curves')
        axes[1,1].legend()
        axes[1,1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('model_evaluation.png', dpi=300, bbox_inches='tight')
        plt.show()
        print("Evaluation plots saved as 'model_evaluation.png'")
    
    def save_model(self):
        """Save the best model and scaler"""
        print("\n" + "="*50)
        print("MODEL SAVING")
        print("="*50)
        
        # Save best model and scaler
        joblib.dump(self.best_model, 'best_fraud_detection_model.pkl')
        joblib.dump(self.scaler, 'feature_scaler.pkl')
        
        print(f"Best model saved: {self.best_model_name}")
        print(f"ROC-AUC Score: {self.results[self.best_model_name]['roc_auc']:.4f}")
        print("Files saved:")
        print("- best_fraud_detection_model.pkl")
        print("- feature_scaler.pkl")
    
    def predict_fraud(self, transaction_data):
        """Predict if a transaction is fraudulent"""
        # Convert to DataFrame if needed
        if isinstance(transaction_data, dict):
            transaction_df = pd.DataFrame([transaction_data])
        else:
            transaction_df = pd.DataFrame([transaction_data])
        
        # Scale the features
        transaction_scaled = self.scaler.transform(transaction_df)
        
        # Make prediction
        prediction = self.best_model.predict(transaction_scaled)[0]
        probability = self.best_model.predict_proba(transaction_scaled)[0, 1]
        
        return {
            'is_fraud': bool(prediction),
            'fraud_probability': float(probability),
            'risk_level': 'High' if probability > 0.7 else 'Medium' if probability > 0.3 else 'Low'
        }
    
    def run_complete_pipeline(self, data_filepath=None):
        """Run the complete fraud detection pipeline"""
        print("CREDIT CARD FRAUD DETECTION PIPELINE")
        print("="*50)
        
        # Load data
        self.load_data(data_filepath)
        
        # Explore data
        self.explore_data()
        
        # Preprocess data
        self.preprocess_data()
        
        # Train models
        self.train_models()
        
        # Train with SMOTE
        self.train_with_smote()
        
        # Evaluate models
        self.evaluate_models()
        
        # Save best model
        self.save_model()
        
        # Demo prediction
        print("\n" + "="*50)
        print("DEMO PREDICTION")
        print("="*50)
        
        # Test with a sample transaction
        sample_transaction = self.X_test.iloc[0].to_dict()
        prediction_result = self.predict_fraud(sample_transaction)
        
        print("Sample prediction:")
        print(f"Prediction result: {prediction_result}")
        print(f"Actual label: {self.y_test.iloc[0]}")
        
        print("\n" + "="*50)
        print("PIPELINE COMPLETED SUCCESSFULLY!")
        print("="*50)


def main():
    """Main function to run the fraud detection pipeline"""
    # Initialize the pipeline
    pipeline = FraudDetectionPipeline(random_state=42)
    
    # Run the complete pipeline
    pipeline.run_complete_pipeline()
    
    # Print summary
    print("\nSUMMARY:")
    print("-" * 30)
    print(f"Dataset Size: {len(pipeline.df):,} transactions")
    print(f"Fraud Rate: {pipeline.df['Class'].mean():.2%}")
    print(f"Best Model: {pipeline.best_model_name}")
    print(f"Best ROC-AUC Score: {pipeline.results[pipeline.best_model_name]['roc_auc']:.4f}")
    
    print("\nFiles Generated:")
    print("- credit_card_data.csv (dataset)")
    print("- best_fraud_detection_model.pkl (trained model)")
    print("- feature_scaler.pkl (data scaler)")
    print("- eda_plots.png (exploratory data analysis)")
    print("- model_evaluation.png (model comparison)")


if __name__ == "__main__":
    main()