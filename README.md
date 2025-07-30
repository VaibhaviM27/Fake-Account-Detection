# Credit Card Fraud Detection Project

A comprehensive machine learning project to detect fraudulent credit card transactions using multiple algorithms including Logistic Regression, Decision Trees, Random Forest, and XGBoost.

## 🎯 Project Overview

This project implements a complete fraud detection pipeline that:
- Generates synthetic credit card transaction data
- Performs comprehensive exploratory data analysis
- Implements multiple machine learning algorithms
- Handles class imbalance using SMOTE
- Provides detailed model evaluation and comparison
- Saves the best performing model for deployment

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- pip package manager
- VSCode (optional) or Jupyter Notebook

### Installation

1. **Clone or download the project files**
2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## 📋 Project Structure

```
fraud-detection/
├── requirements.txt              # Python dependencies
├── fraud_detection_script.py     # Complete Python script for VSCode
├── fraud_detection_notebook.ipynb # Jupyter notebook version
├── README.md                     # This file
└── Generated files (after execution):
    ├── credit_card_data.csv      # Synthetic dataset
    ├── best_fraud_detection_model.pkl # Trained model
    ├── feature_scaler.pkl        # Data scaler
    ├── eda_plots.png            # EDA visualizations
    └── model_evaluation.png     # Model comparison plots
```

## 🔧 Execution Methods

### Method 1: Using VSCode (Python Script)

#### Step 1: Setup Environment
```bash
# Navigate to project directory
cd /path/to/fraud-detection

# Install dependencies
pip install -r requirements.txt
```

#### Step 2: Run the Script
```bash
# Execute the complete pipeline
python fraud_detection_script.py
```

#### Step 3: Monitor Output
The script will:
1. Generate synthetic data (50,000 transactions)
2. Perform exploratory data analysis
3. Train multiple ML models
4. Apply SMOTE for class imbalance
5. Evaluate and compare models
6. Save the best model
7. Generate visualization plots

#### Expected Runtime: 3-5 minutes

### Method 2: Using Jupyter Notebook

#### Step 1: Start Jupyter
```bash
# Navigate to project directory
cd /path/to/fraud-detection

# Start Jupyter Notebook
jupyter notebook
```

#### Step 2: Open the Notebook
- Open `fraud_detection_notebook.ipynb` in Jupyter
- The notebook contains the same functionality as the script but with interactive cells

#### Step 3: Execute Cells
Run cells sequentially or use "Run All":
- **Cell 1**: Import libraries
- **Cell 2**: Generate synthetic data
- **Cell 3-4**: Exploratory data analysis
- **Cell 5**: Data preprocessing
- **Cell 6-8**: Model training and evaluation
- **Cell 9**: SMOTE implementation
- **Cell 10**: Model comparison and saving

## 🧪 What the Code Does

### 1. Data Generation
- Creates 50,000 synthetic transactions (2% fraud rate)
- Features include: Amount, Time, V1-V10 (PCA components), Class (target)
- Fraudulent transactions have different statistical patterns

### 2. Exploratory Data Analysis
- Class distribution analysis
- Amount and time distribution by class
- Feature correlation analysis
- Visualization of key patterns

### 3. Data Preprocessing
- Train/test split (80/20)
- Feature scaling using RobustScaler
- Handles outliers effectively

### 4. Model Training
Implements and compares:
- **Logistic Regression**: Linear classifier with regularization
- **Decision Tree**: Non-linear, interpretable model
- **Random Forest**: Ensemble of decision trees
- **XGBoost**: Gradient boosting algorithm

### 5. Class Imbalance Handling
- Applies SMOTE (Synthetic Minority Oversampling Technique)
- Retrains selected models on balanced data
- Compares original vs SMOTE performance

### 6. Model Evaluation
Comprehensive evaluation using:
- **ROC-AUC Score**: Area under ROC curve
- **F1-Score**: Harmonic mean of precision and recall
- **Confusion Matrix**: True/false positives and negatives
- **Precision-Recall Curves**: Performance across thresholds
- **Classification Reports**: Detailed metrics per class

### 7. Model Deployment Preparation
- Saves best performing model as pickle file
- Saves feature scaler for consistent preprocessing
- Provides prediction function for new transactions

## 📊 Expected Results

### Model Performance (Typical Results)
- **Random Forest**: ROC-AUC ~0.95-0.98
- **XGBoost**: ROC-AUC ~0.94-0.97
- **Logistic Regression**: ROC-AUC ~0.90-0.94
- **Decision Tree**: ROC-AUC ~0.88-0.92

### Generated Files
1. **credit_card_data.csv**: 50,000 synthetic transactions
2. **best_fraud_detection_model.pkl**: Trained model (usually Random Forest)
3. **feature_scaler.pkl**: RobustScaler for preprocessing
4. **eda_plots.png**: EDA visualizations
5. **model_evaluation.png**: Model comparison plots

## 🔍 Key Features

### Advanced Techniques
- **SMOTE**: Handles severe class imbalance (2% fraud rate)
- **RobustScaler**: Handles outliers in financial data
- **Cross-validation**: Ensures robust model evaluation
- **Feature Engineering**: Uses PCA-like transformed features

### Comprehensive Evaluation
- Multiple metrics for imbalanced classification
- ROC curves and Precision-Recall curves
- Confusion matrices for all models
- Feature importance analysis for tree-based models

### Production Ready
- Modular, object-oriented design
- Proper error handling and logging
- Model serialization for deployment
- Prediction function with risk levels

## 🛠️ Customization Options

### Modify Dataset Parameters
```python
# In fraud_detection_script.py, modify:
pipeline = FraudDetectionPipeline(random_state=42)
pipeline.load_data()  # Uses default 50,000 samples, 2% fraud rate

# Or generate custom data:
df = pipeline.generate_synthetic_data(n_samples=100000, fraud_rate=0.01)
```

### Add New Models
```python
# In the train_models method, add:
'New Model': YourModelClass(parameters)
```

### Adjust Evaluation Metrics
```python
# Modify the evaluation methods to include additional metrics
from sklearn.metrics import precision_score, recall_score
```

## 📈 Understanding the Output

### Console Output Sections
1. **Data Generation**: Dataset size and fraud rate
2. **EDA**: Basic statistics and distributions
3. **Preprocessing**: Data shapes and scaling info
4. **Model Training**: Individual model performance
5. **SMOTE Training**: Balanced data results
6. **Evaluation**: Comprehensive model comparison
7. **Model Saving**: Best model information

### Visualization Outputs
1. **EDA Plots**: Class distribution, amount/time patterns, correlations
2. **Model Evaluation**: Performance comparison, ROC curves, confusion matrices

## 🚨 Troubleshooting

### Common Issues

#### 1. Import Errors
```bash
# Solution: Install missing packages
pip install package_name
```

#### 2. Memory Issues (Large Datasets)
```python
# Reduce dataset size in script:
df = generate_synthetic_data(n_samples=10000)  # Smaller dataset
```

#### 3. Plotting Issues
```bash
# For headless servers, add before imports:
import matplotlib
matplotlib.use('Agg')
```

#### 4. Jupyter Kernel Issues
```bash
# Restart kernel and clear output
# Or reinstall jupyter:
pip install --upgrade jupyter
```

## 📚 Learning Objectives

After running this project, you'll understand:

1. **Data Science Pipeline**: Complete ML workflow from data to deployment
2. **Imbalanced Classification**: Techniques for handling rare events
3. **Model Comparison**: How to evaluate and select best algorithms
4. **Feature Engineering**: Importance of data preprocessing
5. **Fraud Detection**: Domain-specific challenges and solutions

## 🔗 Next Steps

### Production Deployment
1. **API Development**: Create REST API using Flask/FastAPI
2. **Real-time Scoring**: Implement streaming predictions
3. **Model Monitoring**: Track performance drift
4. **A/B Testing**: Compare model versions

### Advanced Techniques
1. **Deep Learning**: Neural networks for complex patterns
2. **Ensemble Methods**: Combine multiple models
3. **Feature Selection**: Automated feature engineering
4. **Hyperparameter Tuning**: Optimize model parameters

### Business Integration
1. **Threshold Optimization**: Balance precision vs recall
2. **Cost-Sensitive Learning**: Account for business costs
3. **Explainable AI**: Understand model decisions
4. **Regulatory Compliance**: Meet financial regulations

## 📞 Support

If you encounter issues:
1. Check the troubleshooting section
2. Verify all dependencies are installed
3. Ensure Python version compatibility
4. Review error messages for specific issues

## 📄 License

This project is for educational purposes. Feel free to modify and use for learning.

---

**Happy Learning! 🎉**

This project provides a solid foundation for understanding fraud detection and machine learning pipelines. Experiment with different parameters and techniques to deepen your understanding.