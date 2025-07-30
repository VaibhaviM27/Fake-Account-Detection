#!/usr/bin/env python3
"""
Quick Start Script for SMS Spam Classification
==============================================

This script provides a quick way to test your environment and run a simplified 
version of the SMS spam classification system.

Usage:
    python quick_start.py
"""

import sys
import subprocess
import importlib

def check_package(package_name, import_name=None):
    """Check if a package is installed"""
    if import_name is None:
        import_name = package_name
    
    try:
        importlib.import_module(import_name)
        print(f"✅ {package_name} is installed")
        return True
    except ImportError:
        print(f"❌ {package_name} is not installed")
        return False

def install_package(package_name):
    """Install a package using pip"""
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])
        print(f"✅ Successfully installed {package_name}")
        return True
    except subprocess.CalledProcessError:
        print(f"❌ Failed to install {package_name}")
        return False

def check_environment():
    """Check and set up the environment"""
    print("🔍 Checking Python environment...")
    print(f"Python version: {sys.version}")
    
    # Required packages
    packages = [
        ("pandas", "pandas"),
        ("numpy", "numpy"),
        ("scikit-learn", "sklearn"),
        ("matplotlib", "matplotlib"),
        ("seaborn", "seaborn"),
        ("nltk", "nltk"),
        ("requests", "requests")
    ]
    
    missing_packages = []
    
    for package_name, import_name in packages:
        if not check_package(package_name, import_name):
            missing_packages.append(package_name)
    
    if missing_packages:
        print(f"\n📦 Installing missing packages: {', '.join(missing_packages)}")
        for package in missing_packages:
            install_package(package)
    
    print("\n✅ Environment check completed!")

def run_quick_demo():
    """Run a quick demonstration"""
    print("\n🚀 Running Quick SMS Spam Classification Demo")
    print("=" * 50)
    
    try:
        # Import required libraries
        import pandas as pd
        import numpy as np
        from sklearn.model_selection import train_test_split
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.naive_bayes import MultinomialNB
        from sklearn.metrics import accuracy_score, classification_report
        import re
        
        print("📚 Libraries imported successfully!")
        
        # Create sample data
        print("📝 Creating sample dataset...")
        sample_data = {
            'label': ['ham', 'spam', 'ham', 'spam', 'ham', 'spam'] * 50,
            'message': [
                'Hey, how are you doing today?',
                'URGENT! You have won $1000! Click here now!',
                'Can you pick up some groceries?',
                'FREE! Get your loan approved instantly!',
                'Meeting at 3pm in conference room',
                'Congratulations! You are our winner!'
            ] * 50
        }
        
        df = pd.DataFrame(sample_data)
        print(f"✅ Sample dataset created with {len(df)} messages")
        print(f"📊 Label distribution: {df['label'].value_counts().to_dict()}")
        
        # Simple preprocessing
        def simple_preprocess(text):
            text = text.lower()
            text = re.sub(r'[^a-zA-Z\s]', '', text)
            return text
        
        df['processed'] = df['message'].apply(simple_preprocess)
        print("🔧 Text preprocessing completed!")
        
        # Prepare features and target
        X = df['processed']
        y = df['label'].map({'ham': 0, 'spam': 1})
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )
        
        # Vectorize text
        vectorizer = TfidfVectorizer(max_features=1000)
        X_train_vec = vectorizer.fit_transform(X_train)
        X_test_vec = vectorizer.transform(X_test)
        
        print("📊 TF-IDF vectorization completed!")
        print(f"   Feature matrix shape: {X_train_vec.shape}")
        
        # Train model
        model = MultinomialNB()
        model.fit(X_train_vec, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test_vec)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"\n🤖 Model trained successfully!")
        print(f"📈 Accuracy: {accuracy:.4f}")
        
        # Test with sample messages
        test_messages = [
            "Hi, how are you?",
            "URGENT! Win money now!",
            "Let's meet for coffee",
            "FREE cash prize! Click now!"
        ]
        
        print(f"\n🧪 Testing with sample messages:")
        print("-" * 40)
        
        for msg in test_messages:
            processed_msg = simple_preprocess(msg)
            msg_vec = vectorizer.transform([processed_msg])
            prediction = model.predict(msg_vec)[0]
            probability = model.predict_proba(msg_vec)[0]
            
            result = "SPAM" if prediction == 1 else "HAM"
            confidence = max(probability) * 100
            
            print(f"📱 '{msg}'")
            print(f"🔍 Prediction: {result} (Confidence: {confidence:.1f}%)")
            print()
        
        print("✅ Quick demo completed successfully!")
        print("\n🎯 Next steps:")
        print("   1. Run the full system: python sms_spam_classifier.py")
        print("   2. Try the Jupyter notebook: jupyter notebook sms_spam_classification.ipynb")
        print("   3. Read the guide: SMS_SPAM_CLASSIFICATION_GUIDE.md")
        
    except Exception as e:
        print(f"❌ Error during demo: {e}")
        print("💡 Try installing missing packages or check the troubleshooting guide")

def main():
    """Main function"""
    print("🎯 SMS Spam Classification - Quick Start")
    print("=" * 50)
    
    # Check environment
    check_environment()
    
    # Ask user if they want to run demo
    while True:
        choice = input("\n🎮 Would you like to run the quick demo? (y/n): ").lower().strip()
        if choice in ['y', 'yes']:
            run_quick_demo()
            break
        elif choice in ['n', 'no']:
            print("👋 Skipping demo. You can run it later with: python quick_start.py")
            break
        else:
            print("Please enter 'y' or 'n'")
    
    print("\n🎉 Quick start completed!")

if __name__ == "__main__":
    main()