#!/usr/bin/env python3
"""
Simple test script for the Movie Genre Prediction System
Run this first to verify everything is working before running the full system.
"""

try:
    print("🧪 Testing Movie Genre Prediction System...")
    print("=" * 50)
    
    # Test imports
    print("1. Testing imports...")
    from movie_genre_predictor import MovieGenrePredictor
    import pandas as pd
    import numpy as np
    from sklearn.model_selection import train_test_split
    from sklearn.naive_bayes import MultinomialNB
    print("   ✅ All imports successful")
    
    # Test initialization
    print("\n2. Testing initialization...")
    predictor = MovieGenrePredictor()
    print("   ✅ Predictor initialized successfully")
    
    # Test data loading
    print("\n3. Testing data loading...")
    data = predictor.load_sample_data()
    print(f"   ✅ Loaded {len(data)} movie samples")
    print(f"   📊 Genres: {sorted(data['genre'].unique())}")
    
    # Test text preprocessing
    print("\n4. Testing text preprocessing...")
    sample_text = "A superhero battles evil villains to save the city!"
    processed = predictor.preprocess_text(sample_text)
    print(f"   Original: {sample_text}")
    print(f"   Processed: {processed}")
    print("   ✅ Text preprocessing working")
    
    # Test feature extraction
    print("\n5. Testing feature extraction...")
    X_tfidf, X_count = predictor.prepare_features(data)
    y = predictor.prepare_labels(data)
    print(f"   ✅ TF-IDF features: {X_tfidf.shape}")
    print(f"   ✅ Count features: {X_count.shape}")
    print(f"   ✅ Labels: {y.shape}")
    
    print("\n🎉 All core components working!")
    print("\n📋 Ready to run full system:")
    print("   → python movie_genre_predictor.py")
    
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("💡 Run: pip install -r movie_genre_requirements.txt")
    
except Exception as e:
    print(f"❌ Error: {e}")
    print("💡 Check error message and ensure dependencies are installed")

print("\n" + "=" * 50)