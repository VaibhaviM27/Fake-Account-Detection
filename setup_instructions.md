# Detailed Setup Instructions

## 🖥️ System Requirements

- **Python**: 3.8 or higher
- **RAM**: Minimum 4GB (8GB recommended)
- **Disk Space**: 500MB free space
- **OS**: Windows 10+, macOS 10.14+, or Linux

## 📋 Step-by-Step Setup

### Option A: Using VSCode

#### 1. Install Python
**Windows:**
```bash
# Download from python.org or use Microsoft Store
# Verify installation:
python --version
pip --version
```

**macOS:**
```bash
# Using Homebrew (recommended):
brew install python

# Or download from python.org
# Verify installation:
python3 --version
pip3 --version
```

**Linux (Ubuntu/Debian):**
```bash
sudo apt update
sudo apt install python3 python3-pip
python3 --version
pip3 --version
```

#### 2. Install VSCode
- Download from: https://code.visualstudio.com/
- Install Python extension in VSCode

#### 3. Setup Project
```bash
# Create project directory
mkdir fraud-detection
cd fraud-detection

# Download project files (or copy them)
# Make sure you have:
# - requirements.txt
# - fraud_detection_script.py
# - README.md

# Install dependencies
pip install -r requirements.txt
```

#### 4. Run the Project
```bash
# Execute the script
python fraud_detection_script.py
```

### Option B: Using Jupyter Notebook

#### 1. Install Jupyter
```bash
# Install Jupyter
pip install jupyter

# Or use Anaconda (recommended for beginners):
# Download from: https://www.anaconda.com/products/distribution
```

#### 2. Setup Project
```bash
# Navigate to project directory
cd fraud-detection

# Install dependencies
pip install -r requirements.txt

# Start Jupyter
jupyter notebook
```

#### 3. Run the Notebook
- Open `fraud_detection_notebook.ipynb`
- Run cells sequentially or use "Run All"

### Option C: Using Google Colab (Cloud-based)

#### 1. Upload Files
- Go to: https://colab.research.google.com/
- Upload `fraud_detection_notebook.ipynb`

#### 2. Install Dependencies
Add this cell at the beginning:
```python
!pip install imbalanced-learn xgboost lightgbm plotly
```

#### 3. Run the Notebook
- Execute cells normally
- Files will be saved in Colab environment

## 🔧 Environment Setup Options

### Option 1: Virtual Environment (Recommended)
```bash
# Create virtual environment
python -m venv fraud_env

# Activate it
# Windows:
fraud_env\Scripts\activate
# macOS/Linux:
source fraud_env/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run project
python fraud_detection_script.py

# Deactivate when done
deactivate
```

### Option 2: Conda Environment
```bash
# Create conda environment
conda create -n fraud_env python=3.9

# Activate environment
conda activate fraud_env

# Install dependencies
pip install -r requirements.txt

# Run project
python fraud_detection_script.py
```

### Option 3: Docker (Advanced)
```dockerfile
# Create Dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

CMD ["python", "fraud_detection_script.py"]
```

```bash
# Build and run
docker build -t fraud-detection .
docker run fraud-detection
```

## 🚨 Common Setup Issues

### Issue 1: Permission Errors
**Windows:**
```bash
# Run as administrator or:
pip install --user -r requirements.txt
```

**macOS/Linux:**
```bash
# Use sudo or virtual environment
sudo pip3 install -r requirements.txt
```

### Issue 2: Package Conflicts
```bash
# Create fresh virtual environment
python -m venv fresh_env
source fresh_env/bin/activate  # or fresh_env\Scripts\activate on Windows
pip install -r requirements.txt
```

### Issue 3: Jupyter Not Starting
```bash
# Reinstall jupyter
pip uninstall jupyter
pip install jupyter

# Or use specific port
jupyter notebook --port=8889
```

### Issue 4: Import Errors
```bash
# Check if packages installed correctly
pip list | grep scikit-learn
pip list | grep pandas

# Reinstall specific package
pip install --upgrade scikit-learn
```

## 🔍 Verification Steps

### 1. Test Python Installation
```python
# Run this in Python/Jupyter:
import sys
print(f"Python version: {sys.version}")

import pandas as pd
import numpy as np
import sklearn
print("All core packages imported successfully!")
```

### 2. Test GPU Support (Optional)
```python
# For XGBoost GPU support:
import xgboost as xgb
print(f"XGBoost version: {xgb.__version__}")
```

### 3. Test Plotting
```python
import matplotlib.pyplot as plt
import seaborn as sns

# Simple test plot
plt.plot([1, 2, 3], [1, 4, 2])
plt.show()
print("Plotting works!")
```

## 📊 Performance Optimization

### For Large Datasets
```python
# Reduce dataset size for testing
df = generate_synthetic_data(n_samples=10000, fraud_rate=0.02)
```

### For Faster Training
```python
# Use fewer estimators
RandomForestClassifier(n_estimators=50)  # Instead of 100
```

### For Memory Issues
```python
# Process in chunks or reduce features
X_train = X_train.iloc[:, :5]  # Use first 5 features only
```

## 🎯 Success Indicators

You'll know setup is successful when:
1. ✅ All dependencies install without errors
2. ✅ Script runs and generates synthetic data
3. ✅ Models train and show performance metrics
4. ✅ Plots are generated and saved
5. ✅ Best model is saved as .pkl file

## 📞 Getting Help

### If Setup Fails:
1. **Check Python version**: Must be 3.8+
2. **Update pip**: `pip install --upgrade pip`
3. **Clear pip cache**: `pip cache purge`
4. **Try different Python distribution**: Anaconda, Miniconda
5. **Use cloud platforms**: Google Colab, Kaggle Notebooks

### Resources:
- **Python Installation**: https://www.python.org/downloads/
- **VSCode Setup**: https://code.visualstudio.com/docs/python/python-tutorial
- **Jupyter Guide**: https://jupyter.org/install
- **Anaconda Guide**: https://docs.anaconda.com/anaconda/install/

## 🚀 Quick Test Command

After setup, run this quick test:
```bash
python -c "
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
print('✅ Setup successful! Ready to detect fraud!')
"
```

---

**You're all set! 🎉 Now run the fraud detection project!**