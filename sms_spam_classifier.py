#!/usr/bin/env python3
"""
SMS Spam Classification System
==============================

This script implements a comprehensive SMS spam classification system using:
- TF-IDF vectorization
- Word embeddings (Word2Vec)
- Multiple classifiers: Naive Bayes, Logistic Regression, SVM
- Comprehensive evaluation and comparison

Author: AI Assistant
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (classification_report, confusion_matrix, 
                           accuracy_score, precision_score, recall_score, 
                           f1_score, roc_auc_score, roc_curve)
from sklearn.pipeline import Pipeline
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from wordcloud import WordCloud
import re
import string
import warnings
import os
import requests
from gensim.models import Word2Vec
from gensim.utils import simple_preprocess

warnings.filterwarnings('ignore')

class SMSSpamClassifier:
    """
    A comprehensive SMS spam classification system
    """
    
    def __init__(self):
        """Initialize the classifier"""
        self.data = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.vectorizers = {}
        self.models = {}
        self.results = {}
        self.stemmer = PorterStemmer()
        
        # Download required NLTK data
        self._download_nltk_data()
        
    def _download_nltk_data(self):
        """Download required NLTK datasets"""
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')
            
        try:
            nltk.data.find('corpora/stopwords')
        except LookupError:
            nltk.download('stopwords')
    
    def download_dataset(self):
        """Download the SMS spam dataset"""
        print("📥 Downloading SMS Spam Dataset...")
        
        # URL for the SMS Spam Collection dataset
        url = "https://raw.githubusercontent.com/mohitgupta-omg/Kaggle-SMS-Spam-Collection-Dataset-/master/spam.csv"
        
        try:
            # Download the dataset
            response = requests.get(url)
            response.raise_for_status()
            
            # Save to file
            with open('spam.csv', 'wb') as f:
                f.write(response.content)
            
            print("✅ Dataset downloaded successfully!")
            return True
            
        except Exception as e:
            print(f"❌ Error downloading dataset: {e}")
            print("📝 Creating sample dataset for demonstration...")
            self._create_sample_dataset()
            return False
    
    def _create_sample_dataset(self):
        """Create a sample dataset if download fails"""
        sample_data = {
            'v1': ['ham', 'spam', 'ham', 'spam', 'ham', 'spam', 'ham', 'spam'] * 100,
            'v2': [
                'How are you doing today?',
                'URGENT! You have won $1000! Click here now!',
                'Can you pick up milk on your way home?',
                'FREE! Get your loan approved instantly! Call now!',
                'Meeting at 3pm today in conference room',
                'Congratulations! You are selected for cash prize!',
                'Thanks for the help yesterday',
                'WINNER! Claim your prize now! Limited time offer!'
            ] * 100
        }
        
        df = pd.DataFrame(sample_data)
        df.to_csv('spam.csv', index=False)
        print("✅ Sample dataset created!")
    
    def load_and_explore_data(self):
        """Load and explore the SMS spam dataset"""
        print("📊 Loading and exploring the dataset...")
        
        try:
            # Try different encodings
            encodings = ['latin-1', 'utf-8', 'cp1252']
            for encoding in encodings:
                try:
                    self.data = pd.read_csv('spam.csv', encoding=encoding)
                    break
                except UnicodeDecodeError:
                    continue
            
            if self.data is None:
                raise Exception("Could not read the dataset with any encoding")
            
            # Clean column names and select relevant columns
            if 'v1' in self.data.columns and 'v2' in self.data.columns:
                self.data = self.data[['v1', 'v2']].copy()
                self.data.columns = ['label', 'message']
            elif 'Category' in self.data.columns and 'Message' in self.data.columns:
                self.data = self.data[['Category', 'Message']].copy()
                self.data.columns = ['label', 'message']
            
            # Remove any rows with missing values
            self.data = self.data.dropna()
            
            print(f"✅ Dataset loaded successfully!")
            print(f"📈 Dataset shape: {self.data.shape}")
            print(f"📋 Dataset info:")
            print(self.data.info())
            print(f"\n📊 Label distribution:")
            print(self.data['label'].value_counts())
            
            # Display sample messages
            print(f"\n📝 Sample messages:")
            for i, (label, message) in enumerate(zip(self.data['label'][:5], self.data['message'][:5])):
                print(f"{i+1}. [{label.upper()}] {message[:100]}...")
            
            return True
            
        except Exception as e:
            print(f"❌ Error loading dataset: {e}")
            return False
    
    def visualize_data(self):
        """Create visualizations for the dataset"""
        print("📊 Creating data visualizations...")
        
        # Set up the plotting style
        plt.style.use('default')
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Label distribution
        label_counts = self.data['label'].value_counts()
        axes[0, 0].pie(label_counts.values, labels=label_counts.index, autopct='%1.1f%%', 
                      colors=['lightblue', 'lightcoral'])
        axes[0, 0].set_title('Distribution of SMS Labels')
        
        # 2. Message length distribution
        self.data['message_length'] = self.data['message'].str.len()
        
        spam_lengths = self.data[self.data['label'] == 'spam']['message_length']
        ham_lengths = self.data[self.data['label'] == 'ham']['message_length']
        
        axes[0, 1].hist(ham_lengths, alpha=0.7, label='Ham', bins=50, color='lightblue')
        axes[0, 1].hist(spam_lengths, alpha=0.7, label='Spam', bins=50, color='lightcoral')
        axes[0, 1].set_xlabel('Message Length')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].set_title('Message Length Distribution')
        axes[0, 1].legend()
        
        # 3. Word count distribution
        self.data['word_count'] = self.data['message'].str.split().str.len()
        
        spam_words = self.data[self.data['label'] == 'spam']['word_count']
        ham_words = self.data[self.data['label'] == 'ham']['word_count']
        
        axes[1, 0].hist(ham_words, alpha=0.7, label='Ham', bins=30, color='lightblue')
        axes[1, 0].hist(spam_words, alpha=0.7, label='Spam', bins=30, color='lightcoral')
        axes[1, 0].set_xlabel('Word Count')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].set_title('Word Count Distribution')
        axes[1, 0].legend()
        
        # 4. Box plot for message lengths
        data_for_box = [ham_lengths, spam_lengths]
        axes[1, 1].boxplot(data_for_box, labels=['Ham', 'Ham'])
        axes[1, 1].set_ylabel('Message Length')
        axes[1, 1].set_title('Message Length Box Plot')
        
        plt.tight_layout()
        plt.savefig('sms_data_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Create word clouds
        self._create_wordclouds()
    
    def _create_wordclouds(self):
        """Create word clouds for spam and ham messages"""
        print("☁️ Creating word clouds...")
        
        # Separate spam and ham messages
        spam_text = ' '.join(self.data[self.data['label'] == 'spam']['message'])
        ham_text = ' '.join(self.data[self.data['label'] == 'ham']['message'])
        
        # Create word clouds
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Spam word cloud
        spam_wordcloud = WordCloud(width=400, height=300, background_color='white').generate(spam_text)
        axes[0].imshow(spam_wordcloud, interpolation='bilinear')
        axes[0].set_title('Spam Messages Word Cloud', fontsize=16)
        axes[0].axis('off')
        
        # Ham word cloud
        ham_wordcloud = WordCloud(width=400, height=300, background_color='white').generate(ham_text)
        axes[1].imshow(ham_wordcloud, interpolation='bilinear')
        axes[1].set_title('Ham Messages Word Cloud', fontsize=16)
        axes[1].axis('off')
        
        plt.tight_layout()
        plt.savefig('sms_wordclouds.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def preprocess_text(self, text):
        """Preprocess text data"""
        # Convert to lowercase
        text = text.lower()
        
        # Remove punctuation and digits
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Tokenize
        tokens = word_tokenize(text)
        
        # Remove stopwords and stem
        stop_words = set(stopwords.words('english'))
        tokens = [self.stemmer.stem(token) for token in tokens if token not in stop_words]
        
        return ' '.join(tokens)
    
    def prepare_features(self):
        """Prepare features using different techniques"""
        print("🔧 Preparing features...")
        
        # Preprocess messages
        print("   📝 Preprocessing text...")
        self.data['processed_message'] = self.data['message'].apply(self.preprocess_text)
        
        # Prepare target variable
        y = self.data['label'].map({'ham': 0, 'spam': 1})
        
        # Split the data
        X_train_text, X_test_text, self.y_train, self.y_test = train_test_split(
            self.data['processed_message'], y, test_size=0.2, random_state=42, stratify=y
        )
        
        # 1. TF-IDF Vectorization
        print("   📊 Creating TF-IDF features...")
        tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
        self.X_train_tfidf = tfidf.fit_transform(X_train_text)
        self.X_test_tfidf = tfidf.transform(X_test_text)
        self.vectorizers['tfidf'] = tfidf
        
        # 2. Count Vectorization
        print("   🔢 Creating Count features...")
        count_vec = CountVectorizer(max_features=5000, ngram_range=(1, 2))
        self.X_train_count = count_vec.fit_transform(X_train_text)
        self.X_test_count = count_vec.transform(X_test_text)
        self.vectorizers['count'] = count_vec
        
        # 3. Word2Vec Embeddings
        print("   🧠 Creating Word2Vec embeddings...")
        self._create_word2vec_features(X_train_text, X_test_text)
        
        print("✅ Feature preparation completed!")
    
    def _create_word2vec_features(self, X_train_text, X_test_text):
        """Create Word2Vec embeddings"""
        # Prepare sentences for Word2Vec
        train_sentences = [simple_preprocess(text) for text in X_train_text]
        
        # Train Word2Vec model
        w2v_model = Word2Vec(sentences=train_sentences, vector_size=100, window=5, 
                           min_count=1, workers=4, sg=0)
        
        # Create document vectors by averaging word vectors
        def get_document_vector(text, model, vector_size=100):
            words = simple_preprocess(text)
            word_vectors = []
            for word in words:
                if word in model.wv:
                    word_vectors.append(model.wv[word])
            
            if word_vectors:
                return np.mean(word_vectors, axis=0)
            else:
                return np.zeros(vector_size)
        
        # Create training and testing vectors
        self.X_train_w2v = np.array([get_document_vector(text, w2v_model) for text in X_train_text])
        self.X_test_w2v = np.array([get_document_vector(text, w2v_model) for text in X_test_text])
        
        self.vectorizers['word2vec'] = w2v_model
    
    def train_models(self):
        """Train multiple classification models"""
        print("🤖 Training classification models...")
        
        # Define models
        models_config = {
            'Naive Bayes': MultinomialNB(),
            'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
            'SVM': SVC(random_state=42, probability=True),
            'Random Forest': RandomForestClassifier(random_state=42, n_estimators=100)
        }
        
        # Feature sets
        feature_sets = {
            'TF-IDF': (self.X_train_tfidf, self.X_test_tfidf),
            'Count': (self.X_train_count, self.X_test_count),
            'Word2Vec': (self.X_train_w2v, self.X_test_w2v)
        }
        
        # Train models for each feature set
        for feature_name, (X_train, X_test) in feature_sets.items():
            print(f"\n   📊 Training models with {feature_name} features...")
            
            for model_name, model in models_config.items():
                print(f"      🔄 Training {model_name}...")
                
                # Skip Naive Bayes for Word2Vec (requires non-negative features)
                if model_name == 'Naive Bayes' and feature_name == 'Word2Vec':
                    continue
                
                try:
                    # Train model
                    model.fit(X_train, self.y_train)
                    
                    # Make predictions
                    y_pred = model.predict(X_test)
                    y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
                    
                    # Calculate metrics
                    accuracy = accuracy_score(self.y_test, y_pred)
                    precision = precision_score(self.y_test, y_pred)
                    recall = recall_score(self.y_test, y_pred)
                    f1 = f1_score(self.y_test, y_pred)
                    
                    # Store results
                    key = f"{model_name}_{feature_name}"
                    self.models[key] = model
                    self.results[key] = {
                        'accuracy': accuracy,
                        'precision': precision,
                        'recall': recall,
                        'f1': f1,
                        'y_pred': y_pred,
                        'y_pred_proba': y_pred_proba
                    }
                    
                    print(f"         ✅ {model_name} with {feature_name}: Accuracy = {accuracy:.4f}")
                    
                except Exception as e:
                    print(f"         ❌ Error training {model_name} with {feature_name}: {e}")
        
        print("✅ Model training completed!")
    
    def evaluate_models(self):
        """Evaluate and compare all trained models"""
        print("📊 Evaluating models...")
        
        # Create results DataFrame
        results_data = []
        for key, metrics in self.results.items():
            model_name, feature_type = key.rsplit('_', 1)
            results_data.append({
                'Model': model_name,
                'Features': feature_type,
                'Accuracy': metrics['accuracy'],
                'Precision': metrics['precision'],
                'Recall': metrics['recall'],
                'F1-Score': metrics['f1']
            })
        
        results_df = pd.DataFrame(results_data)
        print("\n📈 Model Performance Comparison:")
        print(results_df.round(4))
        
        # Find best model
        best_model_idx = results_df['F1-Score'].idxmax()
        best_model_info = results_df.iloc[best_model_idx]
        print(f"\n🏆 Best Model: {best_model_info['Model']} with {best_model_info['Features']} features")
        print(f"   F1-Score: {best_model_info['F1-Score']:.4f}")
        
        # Create visualizations
        self._plot_model_comparison(results_df)
        self._plot_confusion_matrices()
        
        return results_df
    
    def _plot_model_comparison(self, results_df):
        """Plot model comparison"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
        
        for i, metric in enumerate(metrics):
            ax = axes[i//2, i%2]
            
            # Create pivot table for heatmap
            pivot_data = results_df.pivot(index='Model', columns='Features', values=metric)
            
            sns.heatmap(pivot_data, annot=True, cmap='YlOrRd', ax=ax, fmt='.3f')
            ax.set_title(f'{metric} Comparison')
        
        plt.tight_layout()
        plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def _plot_confusion_matrices(self):
        """Plot confusion matrices for best models"""
        # Get top 4 models by F1-score
        top_models = sorted(self.results.items(), key=lambda x: x[1]['f1'], reverse=True)[:4]
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.ravel()
        
        for i, (model_key, metrics) in enumerate(top_models):
            model_name, feature_type = model_key.rsplit('_', 1)
            
            cm = confusion_matrix(self.y_test, metrics['y_pred'])
            
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[i])
            axes[i].set_title(f'{model_name}\n({feature_type} features)')
            axes[i].set_xlabel('Predicted')
            axes[i].set_ylabel('Actual')
        
        plt.tight_layout()
        plt.savefig('confusion_matrices.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def predict_message(self, message, model_key=None):
        """Predict if a message is spam or ham"""
        if model_key is None:
            # Use best model (highest F1-score)
            model_key = max(self.results.keys(), key=lambda x: self.results[x]['f1'])
        
        model_name, feature_type = model_key.rsplit('_', 1)
        model = self.models[model_key]
        
        # Preprocess message
        processed_message = self.preprocess_text(message)
        
        # Transform message based on feature type
        if feature_type == 'TF-IDF':
            message_features = self.vectorizers['tfidf'].transform([processed_message])
        elif feature_type == 'Count':
            message_features = self.vectorizers['count'].transform([processed_message])
        elif feature_type == 'Word2Vec':
            words = simple_preprocess(processed_message)
            word_vectors = []
            for word in words:
                if word in self.vectorizers['word2vec'].wv:
                    word_vectors.append(self.vectorizers['word2vec'].wv[word])
            
            if word_vectors:
                message_features = np.mean(word_vectors, axis=0).reshape(1, -1)
            else:
                message_features = np.zeros((1, 100))
        
        # Make prediction
        prediction = model.predict(message_features)[0]
        probability = model.predict_proba(message_features)[0] if hasattr(model, 'predict_proba') else None
        
        result = 'SPAM' if prediction == 1 else 'HAM'
        
        print(f"\n📱 Message: '{message}'")
        print(f"🔍 Prediction: {result}")
        if probability is not None:
            spam_prob = probability[1] * 100
            ham_prob = probability[0] * 100
            print(f"📊 Confidence: Ham: {ham_prob:.1f}%, Spam: {spam_prob:.1f}%")
        print(f"🤖 Model used: {model_name} with {feature_type} features")
        
        return result, probability
    
    def run_interactive_demo(self):
        """Run interactive demo for testing messages"""
        print("\n🎮 Interactive SMS Spam Detection Demo")
        print("=" * 50)
        print("Enter SMS messages to classify them as spam or ham.")
        print("Type 'quit' to exit.\n")
        
        while True:
            message = input("📱 Enter SMS message: ").strip()
            
            if message.lower() == 'quit':
                print("👋 Goodbye!")
                break
            
            if message:
                try:
                    self.predict_message(message)
                except Exception as e:
                    print(f"❌ Error: {e}")
            else:
                print("⚠️ Please enter a valid message.")
            
            print("-" * 50)

def main():
    """Main function to run the SMS spam classification system"""
    print("🚀 SMS Spam Classification System")
    print("=" * 50)
    
    # Initialize classifier
    classifier = SMSSpamClassifier()
    
    # Download and load dataset
    classifier.download_dataset()
    if not classifier.load_and_explore_data():
        print("❌ Failed to load dataset. Exiting...")
        return
    
    # Visualize data
    classifier.visualize_data()
    
    # Prepare features
    classifier.prepare_features()
    
    # Train models
    classifier.train_models()
    
    # Evaluate models
    results_df = classifier.evaluate_models()
    
    # Test with sample messages
    print("\n🧪 Testing with sample messages:")
    test_messages = [
        "Hi, how are you doing today?",
        "URGENT! You have won $1000! Click here now!",
        "Can you pick up milk on your way home?",
        "FREE! Get your loan approved instantly! Call now!",
        "Meeting scheduled for tomorrow at 2 PM",
        "Congratulations! You are our lucky winner! Claim now!"
    ]
    
    for msg in test_messages:
        classifier.predict_message(msg)
        print()
    
    # Run interactive demo
    while True:
        choice = input("\n🎮 Would you like to try the interactive demo? (y/n): ").lower()
        if choice == 'y':
            classifier.run_interactive_demo()
            break
        elif choice == 'n':
            break
        else:
            print("Please enter 'y' or 'n'")
    
    print("\n✅ SMS Spam Classification System completed successfully!")
    print("📊 Generated files:")
    print("   - sms_data_analysis.png")
    print("   - sms_wordclouds.png") 
    print("   - model_comparison.png")
    print("   - confusion_matrices.png")

if __name__ == "__main__":
    main()